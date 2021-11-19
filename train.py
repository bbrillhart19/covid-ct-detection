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
BATCH_SIZE= 2
LR = 0.0001
PATIENCE = 1
GPU = True
MODEL_CKPTS = {'lung':'unet_lung.pt','inf':'unet_infection.pt'} 

class ModelTrainer():
    def __init__(self, model, device, criterion, optim, optim_args, lr_sched, lr_sched_args):
       
        self.model = model
        self.device = device
        self.model.to(self.device)

        self.criterion = criterion
        self.optim = optim([p for p in self.model.parameters() if p.requires_grad],**optim_args)
        self.lr_sched = lr_sched(self.optim,**lr_sched_args)
        
        self.train_losses = []
        self.val_losses = []

    def load_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            return None
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
        self.lr_sched.load_state_dict(checkpoint['lr_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

    def save_checkpoint(self, ckpt_path, epoch, loss):
        checkpoint = dict(
            model_state_dict = self.model.state_dict(),
            optim_state_dict = self.optim.state_dict(),
            lr_state_dict = self.lr_sched.state_dict(),
            epoch = epoch,
            loss = loss
        )
        torch.save(checkpoint, ckpt_path)

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

    def val_step(self, in_data):
        self.model.eval()
        pred = self.model(in_data)
        loss = self.criterion(pred, in_data)
        self.val_losses.append(loss.item())
        return pred  

    def get_val_loss(self):
        loss = float(torch.mean(torch.as_tensor(self.val_losses)))
        self.val_losses = []
        return loss 

def train(lung_trainer, infection_trainer, train_dataloader, val_dataloader):
    epoch = 0
    not_improved = 1
    train_losses = dict(lung=[],inf=[])
    val_losses = dict(lung=[],inf=[])
    while not_improved <= PATIENCE:
        epoch += 1

        # Train one epoch
        for train_data in tqdm(train_dataloader,desc='Epoch ['+str(epoch)+']'):
            ct_image, lung_mask, inf_mask, image_type = train_data['ct_scan'], train_data['lung'], train_data['inf'], train_data['id']
            ct_image = ct_image.to(lung_trainer.device)
            lung_mask = lung_mask.to(lung_trainer.device)
            inf_mask = inf_mask.to(infection_trainer.device)

            # Make lung prediction and backprop
            lung_pred = lung_trainer.train_step(ct_image, lung_mask)

            # Stack with ct_image for infection model
            ct_image = ct_image.detach()
            lung_pred = lung_pred.detach()
            r_lung = lung_pred[:,0].unsqueeze(1)
            l_lung = lung_pred[:,1].unsqueeze(1)
            inf_input = torch.cat((ct_image,r_lung,l_lung),dim=1).to(infection_trainer.device)

            # Make infection prediction
            inf_pred = infection_trainer.train_step(inf_input, inf_mask)
            break
            
        # Output and append losses
        lung_train_loss = lung_trainer.get_train_loss()
        train_losses['lung'].append(lung_train_loss)
        print('Lung Train Loss:',lung_train_loss)

        inf_train_loss = infection_trainer.get_train_loss()
        train_losses['inf'].append(inf_train_loss)
        print('Infection Train Loss:',inf_train_loss)            

        # Step lr_schedulers
        lung_trainer.lr_sched.step()
        infection_trainer.lr_sched.step()

        not_improved += 1

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
        criterion = nn.BCELoss(reduction='mean'),
        optim = torch.optim.Adam,
        optim_args = dict(lr=LR),
        lr_sched = torch.optim.lr_scheduler.StepLR,
        lr_sched_args = dict(step_size=20,gamma=0.1)
    )
    print('Lung Model >>>')
    print('Device:',lung_trainer.device)
    summary(lung_trainer.model,(1,IN_SIZE,IN_SIZE))
    

    # Set infection model to train
    infection_trainer = ModelTrainer(
        model = ResUnet(in_channels=3, out_channels=1),
        device = get_default_device(gpu=GPU),
        criterion = nn.BCELoss(reduction='mean'),
        optim = torch.optim.Adam,
        optim_args = dict(lr=LR),
        lr_sched = torch.optim.lr_scheduler.StepLR,
        lr_sched_args = dict(step_size=20,gamma=0.1)
    )

    print('Infection Model >>>')
    print('Device:',infection_trainer.device)
    summary(infection_trainer.model,(3,IN_SIZE,IN_SIZE))
    
    # Train lung and infection models
    train(lung_trainer, infection_trainer, train_dataloader, val_dataloader)

if __name__=="__main__":
    main()
