import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation
import cv2

def ensure(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

def get_default_device(gpu=True):
    return torch.device('cuda') if torch.cuda.is_available() and gpu else torch.device('cpu')

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def get_roi(image, thresh):
    data = image.copy()
    data = data*255
    data = data.astype(np.uint8)

    mask = np.zeros_like(data)    
    mask[data>thresh] = 255
    mask_dil = binary_dilation(mask, iterations=5)
    mask[mask_dil] = 255

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)

def crop_roi(image, roi):
    x, y, w, h = roi
    return image[y:y+h,x:x+w]

def stack_infection_input(ct_image, lung_pred):
    # Stack with ct_image for infection model
    r_lung = lung_pred[:,0].unsqueeze(1)
    l_lung = lung_pred[:,1].unsqueeze(1)
    lungs = r_lung+l_lung
    return torch.cat((ct_image,lungs),dim=1)

def save_sample(ct_image, lung_mask, lung_pred, inf_mask, inf_pred, folder, num_samples=4):
    path = ensure(folder)
    sample_num = len(os.listdir(path))
    path = os.path.join(path, 'sample_'+str(sample_num).zfill(4)+'.png')
    fig, axes = plt.subplots(3, num_samples, figsize=(18,10))
    row = 0
    for i, ax in enumerate(axes.flatten()):
        if i % num_samples == 0:
            row += 1
        ax.set_title('CT Image')
        ax.imshow(ct_image[i%num_samples,0],cmap='bone')
        if row == 2:
            ax.set_title('Lung')
            ax.imshow(lung_mask[i%num_samples,0],cmap='binary',alpha=0.5,vmin=0,vmax=1)
            ax.imshow(lung_pred[i%num_samples,0],cmap='jet',alpha=0.3,vmin=0,vmax=1)
        elif row == 3:
            ax.set_title('Infection')
            ax.imshow(inf_mask[i%num_samples,0],cmap='binary',alpha=0.4,vmin=0,vmax=1)
            ax.imshow(inf_pred[i%num_samples,0],cmap='jet',alpha=0.2,vmin=0,vmax=1)
    # fig.tight_layout()
    plt.savefig(path)
    plt.close()

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

def save_metrics(metrics, folder, fn):

    def autolabel(bars):
        for bar in bars:
            height = round(bar.get_height(),4)
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    path = os.path.join(ensure(folder), fn)
    title = fn.split('.')[0]
    print('Saving',title,'metrics >>>')
    x = np.arange(len(metrics.items()))
    fig, ax = plt.subplots()
    bars = ax.bar(x, [m[1] for m in metrics.items()])
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics.items()])
    ax.set_title(title+' Binary Metrics')
    autolabel(bars)
    fig.tight_layout()
    plt.savefig(path)
    plt.close()
    


class BinaryMetrics():
    def __init__(self, eps=1e-5):
        self.eps = eps

    def get_metrics(self, gt, pred):
        output = pred.detach().cpu().view(-1, )
        target = gt.detach().cpu().view(-1, ).float()

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


