import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from data import CTDataLoader, CTSliceDataset
from model import get_infection_model, get_lung_model
from utils import *

SPLIT_DATA = False
IN_SIZE = 128
DATA_PATH = 'data'
BATCH_SIZE= 16
LR = 0.0001
PATIENCE = 1
GPU = True

def main():
    # Split data to train, val, test if desired
    if SPLIT_DATA:
        ct_dataloader = CTDataLoader(DATA_PATH)
        ct_dataloader.split_data()

    # Set device
    device = get_default_device(gpu=GPU)
    print('Device:',device)

    # Get train dataloader #TODO: Transforms
    train_dataset = CTSliceDataset('train', IN_SIZE, transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Get val dataloader
    val_dataset = CTSliceDataset('val', IN_SIZE, transform=None)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Get models
    lung_model = get_lung_model()
    lung_model.to(device)
    print('Lung Model >>>')
    summary(lung_model,(1,IN_SIZE,IN_SIZE))

    infection_model = get_infection_model()
    infection_model.to(device)
    print('Infection Model >>>')
    summary(infection_model,(3,IN_SIZE,IN_SIZE))

    # Set optimizers
    lung_optimizer = torch.optim.Adam([p for p in lung_model.parameters() if p.requires_grad], lr=LR)
    infection_optimizer = torch.optim.Adam([p for p in infection_model.parameters() if p.requires_grad], lr=LR)

    # decay LR
    lung_lr_scheduler = torch.optim.lr_scheduler.StepLR(lung_optimizer, step_size=20, gamma=0.1)
    infection_lr_scheduler = torch.optim.lr_scheduler.StepLR(infection_optimizer, step_size=20, gamma=0.1)

    # Loss functions
    lung_criterion = nn.BCELoss(reduction='mean')
    infection_criterion = nn.BCELoss(reduction='mean')

    epoch = 0
    not_improved = 0
    train_losses = dict(lung=[],inf=[])
    val_losses = dict(lung=[],inf=[])
    while epoch < PATIENCE:
        epoch += 1

        # Train one epoch
        lung_model.train()
        infection_model.train()
        lung_losses = []
        inf_losses = []
        for train_data in tqdm(train_dataloader,desc='Epoch ['+str(epoch)+']'):
            ct_image, lung_mask, inf_mask, image_type = train_data['ct_scan'], train_data['lung'], train_data['inf'], train_data['id']
            ct_image = ct_image.to(device)
            lung_mask = lung_mask.to(device)
            inf_mask = inf_mask.to(device)

            # Make lung prediction
            lung_pred = lung_model(ct_image)
            # Backprop lung model
            lung_loss = lung_criterion(lung_pred,lung_mask)
            lung_losses.append(lung_loss.item())
            lung_optimizer.zero_grad()
            lung_loss.backward()
            lung_optimizer.step()

            # Stack with ct_image for infection model
            ct_image = ct_image.detach()
            lung_pred = lung_pred.detach()
            r_lung = lung_pred[:,0].unsqueeze(1)
            l_lung = lung_pred[:,1].unsqueeze(1)
            inf_input = torch.cat((ct_image,r_lung,l_lung),dim=1).to(device)

            # Make infection prediction
            inf_pred = infection_model(inf_input)
            # Backprop infection model
            inf_loss = infection_criterion(inf_pred,inf_mask)
            inf_losses.append(inf_loss.item())
            infection_optimizer.zero_grad()
            inf_loss.backward()
            infection_optimizer.step()
            
        # Output and append losses
        lung_epoch_train_loss = sum(lung_losses)/len(lung_losses)
        train_losses['lung'].append(lung_epoch_train_loss)
        print('Lung Train Loss:',lung_epoch_train_loss)
        inf_epoch_train_loss = sum(inf_losses)/len(inf_losses)
        train_losses['inf'].append(inf_epoch_train_loss)
        print('Inf Train Loss:',inf_epoch_train_loss)            

        # Step lr_schedulers
        lung_lr_scheduler.step()
        infection_lr_scheduler.step()


if __name__=="__main__":
    main()
