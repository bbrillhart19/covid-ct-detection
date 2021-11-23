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
    min_train_loss = round(min(losses['train']),5)
    min_val_loss = round(min(losses['val']),5)
    plt.plot(losses['train'], label='Train Loss')
    plt.plot(losses['val'],label='Val Loss')
    plt.title('Min Train Loss: '+str(min_train_loss)+'\nMin Val Loss: '+str(min_val_loss))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path)
    plt.close()

class BinaryMetrics():
    r"""Calculate common metrics in binary cases.
    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion 
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative 
    rate, as specificity/TPR is meaningless in multiclass cases.
    """
    def __init__(self, eps=1e-5, activation='0-1'):
        self.eps = eps
        self.activation = activation

    def get_metrics(self, gt, pred):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        metrics = dict(
            pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps),
            dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps),
            precision = (tp + self.eps) / (tp + fp + self.eps),
            recall = (tp + self.eps) / (tp + fn + self.eps),
            specificity = (tn + self.eps) / (tn + fp + self.eps)
        )

        return metrics


