import glob
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from utils import *
from tqdm import tqdm

class CTDataLoader():
    def __init__(self, data_path, device, *args, **kwargs):

        # Define dataset folder
        self.data_path = data_path

        # Define metadata path and load into raw_data
        self.metadata_fn = data_path + os.sep + 'metadata.csv'
        self.metadata_df = pd.read_csv(self.metadata_fn)

        # Set device
        self.device = device

        # Update metadata paths
        self.update_metadata_paths()

    # Function to change metadata values to data_path
    def update_metadata_paths(self):
        for idx, row in self.metadata_df.iterrows():
            for key in self.metadata_df.keys():
                new_value = row[key].replace('../input/covid19-ct-scans',self.data_path).replace('/',os.sep) 
                self.metadata_df.iloc[idx] = row.replace([row[key]], new_value)

    def split_data(self, train=0.8, val=0.1, test=0.1):
        '''
        Split all data into train/val/test
        Make sure even data of corona and radio images
        Train = 80% corona/radio Val=10% corona/radio Test=10% corona/radio
        Save slices as <train/val/test>/<dtype>/<fn>_slice<slice_idx>.npy
        '''
        images = dict(
            corona = [],
            radio = [],
        )

        for idx, image in enumerate(self.metadata_df['ct_scan'].values):
            if 'corona' in image:
                images['corona'].append(idx)
            elif 'radio' in image:
                images['radio'].append(idx)

        for image_key in images.keys():
            print('Image Type:',image_key)
            train_num = int(train*len(images[image_key]))
            for train_image in tqdm(images[image_key][:train_num],desc='Saving Train Data as slices'):
                self.save_npy_slices(train_image, 'train')
            
            val_num = train_num+int(val*len(images[image_key]))
            for val_image in tqdm(images[image_key][train_num:val_num],desc='Saving Val Data as slices'):
                self.save_npy_slices(val_image, 'val')

            test_num = val_num+int(test*len(images[image_key]))
            for test_image in tqdm(images[image_key][val_num:min(test_num,len(images[image_key]))],desc='Saving Test Data as slices'):
                self.save_npy_slices(test_image, 'test')        

    def save_npy_slices(self, idx, path):
        for key in self.metadata_df.keys():
            fn = self.metadata_df.loc[idx,key]
            base_path = ensure(os.path.join(*fn.replace(self.data_path,path).split(os.sep)[:-1]))
            name = fn.split(os.sep)[-1]
            data = self.load_data(idx, key)
            for slice_num in range(data.shape[-1]):
                npy_file = name.replace('.nii','_slice'+str(slice_num)+'.npy')
                np.save(os.path.join(base_path,npy_file),data[...,slice_num])

    def load_data_all(self, idx=0):
        # Load all data types for single image
        data_list = [self.load_data(idx, key) for key in self.metadata_df.keys()]
        return data_list

    def load_data(self, idx=0, key=None):
        # Load a single data type by key for single image
        assert key is not None
        data = read_nii(self.metadata_df.loc[idx,key])
        return data
    
    def display_all(self, idx=0, slice_num=0, color_map = 'nipy_spectral'):
        '''
        Plots and a slice with all available annotations
        '''
        data_list = self.load_data_all(idx)

        fig = plt.figure(figsize=(18,15))

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

    def display(self, idx=0, slice_num=0, key=None, color_map = 'nipy_spectral'):
        # display single data type for single image
        assert key is not None
        data = self.load_data(idx, key)
        plt.imshow(data[..., slice_num], cmap='bone')
        plt.title(key + ' image')
        plt.show()

if __name__=="__main__":
    device = get_default_device()
    dataloader = CTDataLoader('data', device)
    dataloader.split_data()
    # for key in dataloader.metadata_df.keys():
    #     print(key)
    #     data0 = dataloader.load_data(0,key)
    #     print(data0.shape)
    #     print('Max:',torch.max(data0),'Min:',torch.min(data0))
    # dataloader.display_all(11,100)



