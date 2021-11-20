import torch
import numpy as np
import os
from matplotlib import pyplot as plt

def ensure(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

def get_default_device(gpu=True):
    return torch.device('cuda') if torch.cuda.is_available() and gpu else torch.device('cpu')

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def stack_infection_input(ct_image, lung_pred):
    # Stack with ct_image for infection model
    r_lung = lung_pred[:,0].unsqueeze(1)
    l_lung = lung_pred[:,1].unsqueeze(1)
    lungs = r_lung+l_lung
    return torch.cat((ct_image,lungs),dim=1)

def save_loss(losses, folder, fn):
    path = os.path.join(ensure(folder),fn)
    plt.plot(losses['train'], label='Train Loss')
    plt.plot(losses['val'],label='Val Loss')
    plt.legend()
    plt.savefig(path)
    plt.close()


