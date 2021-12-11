import os
import torch
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from data import *
from model import ResUnet
from utils import *

SPLIT_DATA = False
IN_SIZE = 128
DATA_PATH = 'data'
BATCH_SIZE = 8
LR = 0.0001
PATIENCE = 10
GPU = True
EXP_NAME = 'cropped_roi'
MODEL_LOGS = ensure(os.path.join('model_logs',EXP_NAME))
MODEL_CKPTS = {'lung':os.path.join(MODEL_LOGS,'unet_lung.pt'),'inf':os.path.join(MODEL_LOGS,'unet_infection.pt')} 
FROM_SAVE = {'lung':False,'inf':False}
# TRAIN = {'lung':False,'inf':False}
RESULTS_FOLDER = ensure(os.path.join('results','train',EXP_NAME))

class ModelTrainer():
    def __init__(self, model, device, ckpt_path, metrics, criterion=None, optim=None, optim_args=None, lr_sched=None, lr_sched_args=None):
       
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.ckpt_path = ckpt_path
        self.metrics = metrics

        self.criterion = criterion
        self.optim = optim([p for p in self.model.parameters() if p.requires_grad],**optim_args)
        self.lr_sched = lr_sched(self.optim,**lr_sched_args)
        
        self.train_losses = []
        self.val_losses = []
        self.metric_scores = {}

    def load_checkpoint(self):
        try:
            checkpoint = torch.load(self.ckpt_path,map_location=self.device)
            print('Loading checkpoint from',self.ckpt_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.lr_sched.load_state_dict(checkpoint['lr_state_dict'])
            return checkpoint['epoch'], checkpoint['loss']
        except FileNotFoundError:
            print(self.ckpt_path,'not found, continuing >>>')
            return 0, np.Inf        

    def save_checkpoint(self, epoch, loss, metrics):
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
        if target is not None:
            loss = self.criterion(pred, target)
            self.val_losses.append(loss.item())
            scores = self.metrics.get_metrics(target, pred)
            for score in scores.keys():
                if score not in self.metric_scores.keys():
                    self.metric_scores[score] = []
                self.metric_scores[score].append(scores[score])
        return pred  

    def get_val_loss(self):
        loss = float(torch.mean(torch.as_tensor(self.val_losses)))
        self.val_losses = []
        return loss 

    def get_metric_scores(self):
        mean_scores = {}
        for score in self.metric_scores.keys():
            mean_scores[score] = float(torch.mean(torch.as_tensor(self.metric_scores[score])))
            self.metric_scores[score] = []
        return mean_scores


def train_lung_model(lung_trainer, train_dataloader, val_dataloader):
    epoch = 0
    losses = dict(train=[], val=[])
    min_loss = np.Inf
    if FROM_SAVE['lung']:
        epoch, min_loss = lung_trainer.load_checkpoint()
    
    patience = 0
    while patience < PATIENCE:
        epoch += 1

        # Train one epoch
        for train_data in tqdm(train_dataloader, desc='Epoch ['+str(epoch)+']'):
            ct_image, lung_mask, image_type = \
                train_data['ct_scan'], train_data['lung'], train_data['id']
            # Train step on batch
            ct_image = ct_image.to(lung_trainer.device)
            lung_mask = lung_mask.to(lung_trainer.device)   
            lung_pred = lung_trainer.train_step(ct_image, lung_mask)

        # Output and append train loss
        train_loss = lung_trainer.get_train_loss()
        losses['train'].append(train_loss)
        print('Lung Train Loss:',train_loss)

        # Step LR Scheduler
        lung_trainer.lr_sched.step()

        # Validate
        for val_data in tqdm(val_dataloader,desc='Validating'):
            ct_image, lung_mask, image_type = \
                val_data['ct_scan'], val_data['lung'], val_data['id']

            # Val step on batch
            ct_image = ct_image.to(lung_trainer.device)
            lung_mask = lung_mask.to(lung_trainer.device)   
            lung_pred = lung_trainer.val_step(ct_image, lung_mask)

        # Output and append val losses
        val_loss = lung_trainer.get_val_loss()
        losses['val'].append(val_loss)
        metrics = lung_trainer.get_metric_scores()
        print('Metrics >>>')
        for score in metrics.keys():
            print('['+score+']',metrics[score])
        if val_loss < min_loss:
            print('Lung val loss improved from:',min_loss,'->',val_loss)
            lung_trainer.save_checkpoint(epoch, val_loss, metrics)
            min_loss = val_loss
            patience = 0
        else:
            print('Lung val loss:',val_loss,'did not improve from',min_loss)
            patience += 1        
        print('Patience:',patience,'Remaining:',PATIENCE-patience)

        save_loss(losses, os.path.join(RESULTS_FOLDER,'losses'), 'lung_model.png')

def train_infection_model(lung_trainer, infection_trainer, train_dataloader, val_dataloader):
    epoch = 0
    losses = dict(train=[],val=[])
    min_loss = np.Inf
    if FROM_SAVE['inf']:
        epoch, min_loss = infection_trainer.load_checkpoint()
    
    patience = 0
    while patience < PATIENCE:
        epoch += 1

        # Train one epoch
        for train_data in tqdm(train_dataloader, desc='Epoch ['+str(epoch)+']'):
            ct_image, inf_mask, image_type = \
                train_data['ct_scan'], train_data['inf'], train_data['id']

            # Get pred from lung model
            ct_image = ct_image.to(lung_trainer.device)
            lung_pred = lung_trainer.val_step(ct_image, None)

            # Stack ct_image with lung pred for infection model input
            inf_input = torch.concat([ct_image,lung_pred],dim=1).to(infection_trainer.device)
            
            # Train step on batch
            ct_image = ct_image.to(infection_trainer.device)
            inf_mask = inf_mask.to(infection_trainer.device)
            inf_pred = infection_trainer.train_step(inf_input, inf_mask)
            
        # Output and append train loss
        train_loss = infection_trainer.get_train_loss()
        losses['train'].append(train_loss)
        print('Infection Train Loss:',train_loss)

        # Step LR Scheduler
        infection_trainer.lr_sched.step()

        # Validate
        for val_data in tqdm(val_dataloader,desc='Validating'):
            ct_image, inf_mask, image_type = \
                val_data['ct_scan'], val_data['inf'], val_data['id']
            
            # Get pred from lung model
            ct_image = ct_image.to(lung_trainer.device)
            lung_pred = lung_trainer.val_step(ct_image, None)

            # Stack ct_image with lung pred for infection model input
            inf_input = torch.concat([ct_image,lung_pred],dim=1).to(infection_trainer.device)
            
            # Val step on batch
            ct_image = ct_image.to(infection_trainer.device)
            inf_mask = inf_mask.to(infection_trainer.device)
            inf_pred = infection_trainer.val_step(inf_input, inf_mask)
            
        # Output and append val losses
        val_loss = infection_trainer.get_val_loss()
        losses['val'].append(val_loss)
        metrics = infection_trainer.get_metric_scores()
        print('Metrics >>>')
        for score in metrics.keys():
            print('['+score+']',metrics[score])
        if val_loss < min_loss:
            print('Infection val loss improved from:',min_loss,'->',val_loss)
            infection_trainer.save_checkpoint(epoch, val_loss, metrics)
            min_loss = val_loss
            patience = 0
        else:
            print('Infection val loss:',val_loss,'did not improve from',min_loss)
            patience += 1
        
        print('Patience:',patience,'Remaining:',PATIENCE-patience)

        save_loss(losses, os.path.join(RESULTS_FOLDER,'losses'), 'infection_model.png')

def eval_models(lung_trainer, infection_trainer, val_dataloader):
    lung_trainer.load_checkpoint()
    infection_trainer.load_checkpoint()
    for val_data in tqdm(val_dataloader,desc='Evaluating'):
        ct_image, lung_mask, inf_mask, image_type = \
                val_data['ct_scan'], val_data['lung'], val_data['inf'], val_data['id']
            
        # Get pred from lung model
        ct_image = ct_image.to(lung_trainer.device)
        lung_mask = lung_mask.to(lung_trainer.device)
        lung_pred = lung_trainer.val_step(ct_image, lung_mask)

        # Stack ct_image with lung pred for infection model input
        inf_input = torch.concat([ct_image,lung_pred],dim=1).to(infection_trainer.device)
        
        # Val step on batch
        ct_image = ct_image.to(infection_trainer.device)
        inf_mask = inf_mask.to(infection_trainer.device)
        inf_pred = infection_trainer.val_step(inf_input, inf_mask)

        save_sample(
            ct_image.detach().cpu().numpy(),
            lung_mask.detach().cpu().numpy(),
            lung_pred.detach().cpu().numpy(),
            inf_mask.detach().cpu().numpy(),
            inf_pred.detach().cpu().numpy(),
            os.path.join(RESULTS_FOLDER,'samples'),
            num_samples=ct_image.size()[0]
        )
            
    # Output test results
    val_loss = lung_trainer.get_val_loss()
    print('Lung Loss:',val_loss)
    metrics = lung_trainer.get_metric_scores()
    print('Metrics >>>')
    for score in metrics.keys():
        print('['+score+']',metrics[score])
    save_metrics(metrics, os.path.join(RESULTS_FOLDER,'metrics'), 'lung_model.png')

    val_loss = infection_trainer.get_val_loss()
    print('Inf Loss:',val_loss)
    metrics = infection_trainer.get_metric_scores()
    print('Metrics >>>')
    for score in metrics.keys():
        print('['+score+']',metrics[score])
    save_metrics(metrics, os.path.join(RESULTS_FOLDER,'metrics'), 'infection_model.png')

def main():
    # Split data to train, val, test if desired
    if SPLIT_DATA:
        ct_dataloader = CTDataLoader(DATA_PATH)
        ct_dataloader.split_data()

    # Get train dataloader #TODO: Transforms
    aug_transform = T.Compose([
        ToTensor(),
        RandomVerticalFlip(0.4),
        RandomHorizontalFlip(0.4),
        RandomRot90(0.2),
        RandomRotate(0.3, 30)
    ])
    train_dataset = CTSliceDataset('train', IN_SIZE, transform=aug_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Get val dataloader
    transform = T.Compose([ToTensor()])
    val_dataset = CTSliceDataset('val', IN_SIZE, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Set lung model to train    
    lung_trainer = ModelTrainer(
        model = ResUnet(in_channels=1, out_channels=1),
        device = get_default_device(gpu=GPU),
        ckpt_path=MODEL_CKPTS['lung'],
        metrics = BinaryMetrics(),
        criterion = nn.BCELoss(reduction='mean'),
        optim = torch.optim.Adam,
        optim_args = dict(lr=LR),
        lr_sched = torch.optim.lr_scheduler.StepLR,
        lr_sched_args = dict(step_size=20,gamma=0.1),
    )

    print('Lung Model >>>')
    print('Device:',lung_trainer.device)
    summary(lung_trainer.model,(1,IN_SIZE,IN_SIZE))

    # Train lung model
    train_lung_model(lung_trainer, train_dataloader, val_dataloader)

    # Load best lung model from checkpoint
    lung_trainer.load_checkpoint()
    
    # Set infection model to train
    infection_trainer = ModelTrainer(
        model = ResUnet(in_channels=2, out_channels=1),
        device = get_default_device(gpu=GPU),
        ckpt_path=MODEL_CKPTS['inf'],
        metrics = BinaryMetrics(),
        criterion = nn.BCELoss(reduction='mean'),
        optim = torch.optim.Adam,
        optim_args = dict(lr=LR),
        lr_sched = torch.optim.lr_scheduler.StepLR,
        lr_sched_args = dict(step_size=20,gamma=0.1),
    )

    print('Infection Model >>>')
    print('Device:',infection_trainer.device)
    summary(infection_trainer.model,(2,IN_SIZE,IN_SIZE))

    # Train Infection model
    train_infection_model(lung_trainer, infection_trainer, train_dataloader, val_dataloader)

    # Evaluate models
    eval_models(lung_trainer, infection_trainer, val_dataloader)

if __name__=="__main__":
    main()
