import os
from skimage.transform import resize
from skimage import img_as_bool
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, dataloader
import torch.nn.functional as F
import nibabel as nib 
import pandas as pd 
import matplotlib.pyplot as plt 
from tqdm import tqdm


from utils import *

IMAGE_TYPES = ['corona','radio']
R_LUNG = 1
L_LUNG = 2
INFECTION = 3

def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array

class CTDataLoader():
    def __init__(self, data_path, *args, **kwargs):

        # TODO: move to config
        # Define dataset folder
        self.data_path = data_path

        # TODO: Move to config
        # Define metadata path and load into raw_data
        self.metadata_fn = self.data_path + os.sep + 'metadata.csv'
        self.metadata_df = pd.read_csv(self.metadata_fn)

        # Update metadata paths
        self.update_metadata_paths()

    # Function to change metadata values to data_path
    def update_metadata_paths(self):
        for idx, row in self.metadata_df.iterrows():
            for key in self.metadata_df.keys():
                new_value = row[key].replace('../input/covid19-ct-scans',self.data_path).replace('/',os.sep) 
                self.metadata_df.iloc[idx] = row.replace([row[key]], new_value)

    def save_all(self, path):
        '''Save All Data to path'''
        for idx, row in self.metadata_df.iterrows():
            self.save_npy_slices(idx, path)

    def split_data(self, train=0.8, val=0.1):
        '''
        Split all data into train/test
        Make sure even data of corona and radio image_types
        Save slices as <train/test>/<dtype>/<fn>_slice<slice_idx>.npy
        '''
        image_types = {image_type:[] for image_type in IMAGE_TYPES}

        for image_type in image_types.keys():
            print('Image Type:',image_type)
            for idx in self.metadata_df.index[self.metadata_df['ct_scan'].str.contains(image_type)]:
                image_types[image_type].append(idx)

            train_num = int(train*len(image_types[image_type]))
            for train_idx in image_types[image_type][:train_num]:
                self.save_npy_slices(train_idx, 'train')
            val_num = train_num+int(val*len(image_types[image_type]))
            for val_idx in image_types[image_type][train_num:val_num]:
                self.save_npy_slices(val_idx, 'val')  
            for test_idx in image_types[image_type][val_num:]:
                self.save_npy_slices(test_idx, 'test')          

    def save_npy_slices(self, idx, path):
        # NOTE: Renames masks with ct_scan name for convenience in CTSliceDataset
        for key in self.metadata_df.keys():
            print('Saving',path,key,'>>>')
            fn = os.path.basename(self.metadata_df.loc[idx,'ct_scan'])
            base_path = ensure(os.path.join(path,key))
            data = self.load_data(idx, key)
            for slice_num in tqdm(range(data.shape[-1]),desc=fn):
                npy_file = os.path.join(base_path,fn.replace('.nii','_slice'+str(slice_num)+'.npy'))
                if not os.path.exists(npy_file):
                    np.save(npy_file,data[...,slice_num])

    def load_data_all(self, idx):
        # Load all data types for single image
        data_list = [self.load_data(idx, key) for key in self.metadata_df.keys()]
        return data_list

    def load_data(self, idx, key=None):
        # Load a single data type by key for single image
        assert key is not None
        data = read_nii(self.metadata_df.loc[idx,key])
        return data
    
    def display_all(self, idx, slice_num=0, color_map = 'nipy_spectral'):
        '''
        Plots and a slice with all available annotations
        '''
        data_list = self.load_data_all(idx)

        fig = plt.figure(figsize=(18,5))

        plt.subplot(1,4,1)
        plt.imshow(data_list[0][..., slice_num], cmap='bone')
        plt.title('Original Image')

        plt.subplot(1,4,2)
        plt.imshow(data_list[0][..., slice_num], cmap='bone')
        plt.imshow(data_list[1][..., slice_num], alpha=0.5, cmap=color_map)
        plt.title('Lung Mask')

        plt.subplot(1,4,3)
        plt.imshow(data_list[0][..., slice_num], cmap='bone')
        plt.imshow(data_list[2][..., slice_num], alpha=0.5, cmap=color_map)
        plt.title('Infection Mask')

        plt.subplot(1,4,4)
        plt.imshow(data_list[0][..., slice_num], cmap='bone')
        plt.imshow(data_list[3][..., slice_num], alpha=0.5, cmap=color_map)
        plt.title('Lung and Infection Mask')

        plt.show()

    def display(self, fn, slice_num=0, key=None, color_map = 'nipy_spectral'):
        # display single data type for single image
        assert key is not None
        data = self.load_data(fn, key)
        plt.imshow(data[..., slice_num], cmap='bone')
        plt.title(key + ' image')
        plt.show()

class CTSliceDataset(Dataset):
    def __init__(self, data_path, size, transform=None):
        # Lung mode for training SSD, infection mode for Unet (not available yet)
        self.data_path = data_path
        self.transform = transform
        self.size = (size, size)

        ct_path = os.path.join(self.data_path,'ct_scan')
        self.ct_images = [os.path.join(ct_path,ct_image) for ct_image in os.listdir(ct_path)]     

    def __getitem__(self, idx):
        # Set image type id
        for i, image_type in enumerate(IMAGE_TYPES):
            if image_type in self.ct_images[idx]:
                image_type_id = i   

        # Load ct scan and resize
        ct_image = np.load(self.ct_images[idx])
        ct_image = resize(ct_image, self.size)
        ct_image = np.expand_dims(ct_image, axis=0)
        ct_image = normalize(ct_image).astype(np.float32)

        # Get mask
        mask_id = self.ct_images[idx].replace('ct_scan','lung_and_infection_mask')
        mask = np.load(mask_id)
 
        # 1-> RLung 2-> LLung 3 -> Infection
        # Lung mask 2xHxW
        r_lung_mask = mask.copy()
        r_lung_mask[mask != R_LUNG] = 0
        r_lung_mask[mask == R_LUNG] = 1
        l_lung_mask = mask.copy()
        l_lung_mask[mask != L_LUNG] = 0
        l_lung_mask[mask == L_LUNG] = 1
        
        # Resize lung masks and stack
        r_lung_mask = img_as_bool(resize(r_lung_mask, self.size)).astype(np.float32)
        l_lung_mask = img_as_bool(resize(l_lung_mask, self.size)).astype(np.float32)
        lung_mask = np.stack([r_lung_mask,l_lung_mask],axis=0)

        # Infection mask 1xHxW
        inf_mask = mask.copy()
        inf_mask[mask != INFECTION] = 0
        inf_mask[mask == INFECTION] = 1
        inf_mask = img_as_bool(resize(inf_mask, self.size)).astype(np.float32)
        inf_mask = np.expand_dims(inf_mask, axis=0)

        x = {'ct_scan':ct_image,'lung':lung_mask,'inf':inf_mask,'id':image_type_id}

        if self.transform is not None:
            x = self.transform(x)
        
        return x    

    def __len__(self):
        return len(self.ct_images)

# Transforms
class RandomVerticalFlip(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, x):
        if np.random.rand() <= self.p:
            for img in x.keys():
                if img == 'id':
                    continue
                for c in range(x[img].shape[0]):
                    x[img][c] = np.flipud(x[img][c])        
        return x
    
# Distribution utilities
def calculate_mean_offset(base_image_type='corona'):
    image_types = {image_type:[] for image_type in IMAGE_TYPES}
    for dataset_type in ['train','test']:
        
        dataset = CTSliceDataset(dataset_type, 256, transform=None)
        dataloader = DataLoader(
            dataset, batch_size=4, shuffle=False, num_workers=4
        )
        for x in tqdm(dataloader,desc='Calculating '+dataset_type+' means'):
            ct_images = x['ct_scan']
            ids = x['id']
            for i in range(ct_images.shape[0]):
                image_types[IMAGE_TYPES[int(ids[i])]].append(torch.mean(ct_images[i])) 
        for image_type in image_types.keys():
            total = len(image_types[image_type])
            mean = sum(image_types[image_type]) / total if total > 0 else 'N/A'
            print(dataset_type,image_type,'total:',total)
            print(dataset_type,image_type,'mean:',mean)

# Visualization
def display_sample(batch):
    batch_size = batch['ct_scan'].size()[0]
    ct_images = batch['ct_scan']
    lung_masks = batch['lung']
    inf_masks = batch['inf']
    fig, axes = plt.subplots(batch_size, batch_size, figsize=(16,10))
    for i, ax in enumerate(axes.flatten()):
        # First Row ct_image
        if i < batch_size:
            ax.imshow(ct_images[i,0],cmap='bone')
        elif i < batch_size*2:
            ax.imshow(lung_masks[i%batch_size,0],cmap='gray')
        elif i < batch_size*3:
            ax.imshow(lung_masks[i%batch_size,1],cmap='gray')
        else:
            ax.imshow(inf_masks[i%batch_size,0],cmap='gray')
    plt.show()


if __name__=="__main__":
    # calculate_mean_offset()
    # dataloader = CTDataLoader('data')
    # dataloader.split_data()
    test_transform = T.Compose([RandomVerticalFlip(0.8)])
    test_dataset = CTSliceDataset('test', 128, transform=test_transform)
    test_dataloader = DataLoader(
        test_dataset, batch_size=4, shuffle=True, num_workers=4,) #collate_fn=collate_fn
    for x in test_dataloader:
        display_sample(x)
    # dataloader.display_all(0,slice_num=5)
    # for key in dataloader.metadata_df.keys():
    #     print(key)
    #     data0 = dataloader.load_data(0,key)
    #     print(data0.shape)
    #     print('Max:',torch.max(data0),'Min:',torch.min(data0))
    # dataloader.display_all(11,100)



