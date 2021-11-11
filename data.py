import os
from torch.utils.data import Dataset, DataLoader
import nibabel as nib 
import pandas as pd 
import matplotlib.pyplot as plt 
from tqdm import tqdm

from utils import *

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

    def split_data(self, train=0.8):
        '''
        Split all data into train/test
        Make sure even data of corona and radio image_types
        Save slices as <train/test>/<dtype>/<fn>_slice<slice_idx>.npy
        '''
        # TODO: Move to config???
        image_types = dict(
            corona = [],
            radio = [],
        )

        for image_type in image_types.keys():
            print('Image Type:',image_type)
            for idx in self.metadata_df.index[self.metadata_df['ct_scan'].str.contains(image_type)]:
                image_types[image_type].append(idx)

            train_num = int(train*len(image_types[image_type]))
            for train_idx in image_types[image_type][:train_num]:
                self.save_npy_slices(train_idx, 'train')
            for test_idx in image_types[image_type][train_num:]:
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
    def __init__(self, data_path, mode=None, transform=None):
        # Lung mode for training SSD, infection mode for Unet (not available yet)
        assert mode in ['lung','infection']
        self.mode = mode
        self.data_path = data_path
        self.transform = transform

        ct_path = os.path.join(self.data_path,'ct_scan')
        self.ct_images = [os.path.join(ct_path,ct_image) for ct_image in os.listdir(ct_path)]     

    def __getitem__(self, idx):
        # Get mask based on mode
        mask_id = self.ct_images[idx].replace('ct_scan',self.mode+'_mask')
        # Load ct scan and mask 
        ct_image = np.load(self.ct_images[idx])
        mask = np.load(mask_id)
        if self.mode == 'lung':
            mask[mask==2] = 1
            
        if self.transform is not None:
            ct_image, mask = self.transform(ct_image, mask)
        
        return ct_image, mask     

    def __len__(self):
        return len(self.ct_images)

if __name__=="__main__":
    # dataloader = CTDataLoader('data')
    # dataloader.split_data()
    test_dataset = CTSliceDataset('test',mode='lung')
    test_dataloader = DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn
    )
    for ct_image, mask in test_dataloader:
        continue
    # dataloader.display_all(0,slice_num=5)
    # for key in dataloader.metadata_df.keys():
    #     print(key)
    #     data0 = dataloader.load_data(0,key)
    #     print(data0.shape)
    #     print('Max:',torch.max(data0),'Min:',torch.min(data0))
    # dataloader.display_all(11,100)



