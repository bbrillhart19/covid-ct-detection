import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from data import CTDataLoader, CTSliceDataset
from model import ResUnet
from utils import *

SPLIT_DATA = False
IN_SIZE = 128
DATA_PATH = 'data'
BATCH_SIZE= 16
LR = 0.0001
PATIENCE = 1
GPU = True
TRAIN = {'lung':True,'inf':False}
MODEL_LOGS = ensure(os.path.join('model_logs'))
MODEL_CKPTS = {'lung':os.path.join(MODEL_LOGS,'unet_lung.pt'),'inf':os.path.join(MODEL_LOGS,'unet_infection.pt')} 
FROM_SAVE = {'lung':False,'inf':False}

class ModelTrainer():
    def __init__(self, model, device, ckpt_path, criterion, optim, optim_args, lr_sched, lr_sched_args):
       
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.ckpt_path = ckpt_path

        self.criterion = criterion
        self.optim = optim([p for p in self.model.parameters() if p.requires_grad],**optim_args)
        self.lr_sched = lr_sched(self.optim,**lr_sched_args)
        
        self.train_losses = []
        self.val_losses = []

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(self.ckpt_path)
            print('Loading checkpoint from',self.ckpt_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.lr_sched.load_state_dict(checkpoint['lr_state_dict'])
            return checkpoint['epoch'], checkpoint['loss']
        except FileNotFoundError:
            print(self.ckpt_path,'not found, continuing >>>')
            return 0, np.Inf        

    def save_checkpoint(self, epoch, loss):
        checkpoint = dict(
            model_state_dict = self.model.state_dict(),
            optim_state_dict = self.optim.state_dict(),
            lr_state_dict = self.lr_sched.state_dict(),
            epoch = epoch,
            loss = loss
        )
        torch.save(checkpoint, self.ckpt_path)
        print('Saved',self.ckpt_path)

    def train_step(self, in_data, target):
        self.model.train()
        pred = self.model(in_data)
        loss = self.criterion(pred, target)
        self.train_losses.append(loss.item())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return pred

    def get_train_loss(self):
        loss = float(torch.mean(torch.as_tensor(self.train_losses)))
        self.train_losses = []
        return loss

    def val_step(self, in_data, target=None):
        self.model.eval()
        pred = self.model(in_data)
        if target:
            loss = self.criterion(pred, target)
            self.val_losses.append(loss.item())
        return pred  

    def get_val_loss(self):
        loss = float(torch.mean(torch.as_tensor(self.val_losses)))
        self.val_losses = []
        return loss 

def train(lung_trainer, infection_trainer, train_dataloader, val_dataloader):
    epochs = dict(lung=0,inf=0)    
    train_losses = dict(lung=[],inf=[])
    val_losses = dict(lung=[],inf=[])    
    min_loss = dict(lung=np.Inf,inf=np.Inf)

    if FROM_SAVE['lung']:
        epochs['lung'], min_loss['lung'] = lung_trainer.load_checkpoint()
    if FROM_SAVE['inf']:
        epochs['inf'], min_loss['inf'] = infection_trainer.load_checkpoint()
    

    epoch = max(epochs[model] for model in epochs.keys())
    patience = 0
    while patience <= PATIENCE:
        epoch += 1

        # Train one epoch
        for train_data in tqdm(train_dataloader,desc='Epoch ['+str(epoch)+']'):
            ct_image, lung_mask, inf_mask, image_type = \
                train_data['ct_scan'], train_data['lung'], train_data['inf'], train_data['id']

            if TRAIN['lung']:
                # Make lung prediction and backprop
                ct_image = ct_image.to(lung_trainer.device)
                lung_mask = lung_mask.to(lung_trainer.device)   
                lung_pred = lung_trainer.train_step(ct_image, lung_mask)
            else:
                lung_pred = lung_trainer.val_step(ct_image, None)

            if infection_trainer:
                # Stack with ct_image for infection model
                ct_image = ct_image.detach()
                lung_pred = lung_pred.detach()
                r_lung = lung_pred[:,0].unsqueeze(1)
                l_lung = lung_pred[:,1].unsqueeze(1)
                lungs = r_lung+l_lung
                inf_input = torch.cat((ct_image,lungs),dim=1).to(infection_trainer.device)

                # Make infection prediction
                inf_mask = inf_mask.to(infection_trainer.device)
                inf_pred = infection_trainer.train_step(inf_input, inf_mask)

        if TRAIN['lung']:    
            # Output and append losses
            lung_train_loss = lung_trainer.get_train_loss()
            train_losses['lung'].append(lung_train_loss)
            print('Lung Train Loss:',lung_train_loss)
            lung_trainer.lr_sched.step()

        if infection_trainer:
            inf_train_loss = infection_trainer.get_train_loss()
            train_losses['inf'].append(inf_train_loss)
            print('Infection Train Loss:',inf_train_loss)            
            infection_trainer.lr_sched.step()

        # Validate
        for val_data in tqdm(val_dataloader,desc='Validating'):
            ct_image, lung_mask, inf_mask, image_type = \
                val_data['ct_scan'], val_data['lung'], val_data['inf'], val_data['id']
            ct_image = ct_image.to(lung_trainer.device)
            lung_mask = lung_mask.to(lung_trainer.device)
            inf_mask = inf_mask.to(infection_trainer.device)

            # Make lung prediction and backprop
            lung_pred = lung_trainer.val_step(ct_image, lung_mask)

            # Stack with ct_image for infection model
            ct_image = ct_image.detach()
            lung_pred = lung_pred.detach()
            r_lung = lung_pred[:,0].unsqueeze(1)
            l_lung = lung_pred[:,1].unsqueeze(1)
            lungs = r_lung+l_lung
            inf_input = torch.cat((ct_image,lungs),dim=1).to(infection_trainer.device)

            # Make infection prediction
            inf_pred = infection_trainer.val_step(inf_input, inf_mask)
            
        # Output and append losses
        lung_val_loss = lung_trainer.get_val_loss()
        val_losses['lung'].append(lung_val_loss)
        if lung_val_loss < min_loss['lung']:
            print('Lung Val Loss:',lung_val_loss,'\nimproved from',min_loss['lung'])
            lung_trainer.save_checkpoint(epoch, lung_val_loss)
            min_loss['lung'] = lung_val_loss
        else:
            print('Lung Val Loss:',lung_val_loss,'\ndid not improve from',min_loss['lung'])
        inf_val_loss = infection_trainer.get_val_loss()
        val_losses['inf'].append(inf_val_loss)
        # Infection model results determine improvement
        if inf_val_loss < min_loss['inf']:
            print('Infection Val Loss',inf_val_loss,'\nimproved from',min_loss['inf'])
            infection_trainer.save_checkpoint(epoch, inf_val_loss)
            min_loss['inf'] = inf_val_loss
            patience = 0
        else:
            print('Infection Val Loss',inf_val_loss,'\ndid not improve from',min_loss['inf'])
            patience += 1

        print('Patience:',patience,'Remaining:',PATIENCE-patience)
    
    print('Training Completed')        

def main():
    # Split data to train, val, test if desired
    if SPLIT_DATA:
        ct_dataloader = CTDataLoader(DATA_PATH)
        ct_dataloader.split_data()

    # Get train dataloader #TODO: Transforms
    train_dataset = CTSliceDataset('train', IN_SIZE, transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Get val dataloader
    val_dataset = CTSliceDataset('val', IN_SIZE, transform=None)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Set lung model to train
    lung_trainer = ModelTrainer(
        model = ResUnet(in_channels=1, out_channels=2),
        device = get_default_device(gpu=GPU),
        ckpt_path=MODEL_CKPTS['lung'],
        criterion = nn.BCELoss(reduction='mean'),
        optim = torch.optim.Adam,
        optim_args = dict(lr=LR),
        lr_sched = torch.optim.lr_scheduler.StepLR,
        lr_sched_args = dict(step_size=20,gamma=0.1)
    )
    print('Lung Model >>>')
    print('Device:',lung_trainer.device)
    summary(lung_trainer.model,(1,IN_SIZE,IN_SIZE))
    
    if TRAIN['inf']:
        # Set infection model to train
        infection_trainer = ModelTrainer(
            model = ResUnet(in_channels=2, out_channels=1),
            device = get_default_device(gpu=GPU),
            ckpt_path=MODEL_CKPTS['inf'],
            criterion = nn.BCELoss(reduction='mean'),
            optim = torch.optim.Adam,
            optim_args = dict(lr=LR),
            lr_sched = torch.optim.lr_scheduler.StepLR,
            lr_sched_args = dict(step_size=20,gamma=0.1)
        )
    else:
        infection_trainer = None

    print('Infection Model >>>')
    print('Device:',infection_trainer.device)
    summary(infection_trainer.model,(2,IN_SIZE,IN_SIZE))
    
    # Train lung and infection models
    train(lung_trainer, infection_trainer, train_dataloader, val_dataloader)

if __name__=="__main__":
    main()
