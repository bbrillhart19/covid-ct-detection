import torch
import numpy as np
import os

def ensure(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

def get_default_device(gpu=True):
    return torch.device('cuda') if torch.cuda.is_available() and gpu else torch.device('cpu')

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
    


