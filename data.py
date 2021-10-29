import nibabel as nib 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from utils import *
from tqdm import tqdm

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

        # Define dataset folder
        self.data_path = data_path

        # Define metadata path and load into raw_data
        self.metadata_fn = data_path + os.sep + 'metadata.csv'
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
        # TODO: Move to config
        image_types = dict(
            corona = [],
            radio = [],
        )

        for image_type in image_types.keys():
            print('Image Type:',image_type)
            for image in [img for img in self.metadata_df['ct_scan'].values if image_type in img]:
                image_types[image_type].append(image)

            train_num = int(train*len(image_types[image_type]))
            for train_image in image_types[image_type][:train_num]:
                self.save_npy_slices(train_image, 'train')
            for test_image in image_types[image_type][train_num:len(image_types[image_type])]:
                self.save_npy_slices(test_image, 'test')        

    def save_npy_slices(self, image, path):
        for key in self.metadata_df.keys():
            print('Saving',path,key,'>>>')
            fn = self.metadata_df.loc[self.metadata_df['ct_scan']==image][key][0]
            base_path = ensure(os.path.join(*fn.replace(self.data_path,path).split(os.sep)[:-1]))
            name = fn.split(os.sep)[-1]
            data = self.load_data(fn, key)
            for slice_num in tqdm(range(data.shape[-1]),desc=name):
                npy_file = name.replace('.nii','_slice'+str(slice_num)+'.npy')
                np.save(os.path.join(base_path,npy_file),data[...,slice_num])

    def load_data_all(self, fn):
        # Load all data types for single image
        data_list = [self.load_data(fn, key) for key in self.metadata_df.keys()]
        return data_list

    def load_data(self, fn, key=None):
        # Load a single data type by key for single image
        assert key is not None
        data = read_nii(self.metadata_df.loc[self.metadata_df[key]==fn][key][0])
        return data
    
    def display_all(self, fn, slice_num=0, color_map = 'nipy_spectral'):
        '''
        Plots and a slice with all available annotations
        '''
        data_list = self.load_data_all(fn)

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

    def display(self, fn, slice_num=0, key=None, color_map = 'nipy_spectral'):
        # display single data type for single image
        assert key is not None
        data = self.load_data(fn, key)
        plt.imshow(data[..., slice_num], cmap='bone')
        plt.title(key + ' image')
        plt.show()

if __name__=="__main__":
    dataloader = CTDataLoader('data')
    dataloader.split_data()
    # for key in dataloader.metadata_df.keys():
    #     print(key)
    #     data0 = dataloader.load_data(0,key)
    #     print(data0.shape)
    #     print('Max:',torch.max(data0),'Min:',torch.min(data0))
    # dataloader.display_all(11,100)



