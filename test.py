import os
import torch
from torchvision import transforms as T
from torchsummary import summary
from tqdm import tqdm
from data import *
from model import ResUnet
from utils import *

GPU = True
EXP_NAME = 'cropped_roi'
MODEL_LOGS = ensure(os.path.join('model_logs',EXP_NAME))
MODEL_CKPTS = {'lung':os.path.join(MODEL_LOGS,'unet_lung.pt'),'inf':os.path.join(MODEL_LOGS,'unet_infection.pt')}
RESULTS_FOLDER = ensure(os.path.join('results','test',EXP_NAME))
BATCH_SIZE = 4
IN_SIZE = 128

def test(lung_model, inf_model, test_dataloader, device):
    # Predict test set with lung and inf model
    # NOTE: patching not implemented since it didn't work anyway
    lung_model.eval()
    inf_model.eval()
    
    # Init metrics
    metrics = BinaryMetrics()
    lung_metrics = {}
    inf_metrics = {}

    # Run batched test loop on slices
    for test_data in tqdm(test_dataloader, desc='Predicting'):
        ct_image, lung_mask, inf_mask = \
            test_data['ct_scan'], test_data['lung'], test_data['inf']

        ct_image = ct_image.to(device)

        # Make lung prediction
        lung_pred = lung_model(ct_image)

        # Stack lung_pred to inf_input
        inf_input = torch.concat([ct_image,lung_pred],dim=1).to(device)

        # Make infection prediction
        inf_pred = inf_model(inf_input)

        # Add lung metrics
        lung_scores = metrics.get_metrics(lung_mask, lung_pred)
        for score in lung_scores.keys():
            if score not in lung_metrics.keys():
                lung_metrics[score] = []
            lung_metrics[score].append(lung_scores[score])

        # Add infection metrics
        inf_scores = metrics.get_metrics(inf_mask, inf_pred)
        for score in inf_scores.keys():
            if score not in inf_metrics.keys():
                inf_metrics[score] = []
            inf_metrics[score].append(inf_scores[score])

        # Save samples
        save_sample(
            ct_image.detach().cpu().numpy(),
            lung_mask.detach().cpu().numpy(),
            lung_pred.detach().cpu().numpy(),
            inf_mask.detach().cpu().numpy(),
            inf_pred.detach().cpu().numpy(),
            os.path.join(RESULTS_FOLDER,'samples'),
            num_samples=ct_image.size()[0]
        )
    
    # Output and save lung metrics
    print('Lung Metrics >>>')
    for score in lung_metrics.keys():
        lung_metrics[score] = float(torch.mean(torch.as_tensor(lung_metrics[score])))
        print('['+score+']', lung_metrics[score])
    save_metrics(lung_metrics, os.path.join(RESULTS_FOLDER,'metrics'), 'lung_model.png')

    # Output and save infection metrics
    print('Infection Metrics >>>')
    for score in inf_metrics.keys():
        inf_metrics[score] = float(torch.mean(torch.as_tensor(inf_metrics[score])))
        print('['+score+']', inf_metrics[score])
    save_metrics(inf_metrics, os.path.join(RESULTS_FOLDER,'metrics'), 'inf_model.png')

def main():
    # Init dataloaders with test data
    # NOTE: patching not implemented since it didn't work anyway
    transform = T.Compose([ToTensor()])
    test_dataset = CTSliceDataset('test', IN_SIZE, transform=transform,crop=True,patching=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Set device
    device = get_default_device(GPU)

    # Init models
    lung_model = ResUnet(in_channels=1,out_channels=1).to(device)
    inf_model = ResUnet(in_channels=2,out_channels=1).to(device)

    # Load checkpoints from save
    lung_ckpt = torch.load(MODEL_CKPTS['lung'],map_location=device)
    lung_model.load_state_dict(lung_ckpt['model_state_dict'])

    inf_ckpt = torch.load(MODEL_CKPTS['inf'],map_location=device)
    inf_model.load_state_dict(inf_ckpt['model_state_dict'])

    # Print summaries
    print('Lung Model >>>')
    print('Device:',device)
    summary(lung_model,(1,IN_SIZE,IN_SIZE))

    print('Infection Model >>>')
    print('Device:',device)
    summary(inf_model,(2,IN_SIZE,IN_SIZE))

    # Send to test
    test(lung_model, inf_model, test_dataloader, device)

if __name__=="__main__":
    main()
