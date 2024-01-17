import os
import glob
import json

import torch
from torch.utils.data import Dataset
import torchio as tio
import pandas as pd
import nrrd

from dataset.utils import show_item


class AllCTsDataset(Dataset):
    def __init__(self, root_dir='data/AllCTs_nrrd_global', split='train', augmentation=False,
                 resize_d=1, resize_h=1, resize_w=1):
        
        assert split in ['all', 'train', 'val', 'test'], 'Invalid split: {}'.format(split)

        self.root_dir = root_dir

        # Read the CSV file as a DataFrame
        self.df = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
        self.df['name'] = self.df['name'].astype(str)

        # Take only the required split
        if split != 'all':
            self.df = self.df[self.df['split'] == split]

        # Read one 3d image and define sizes
        img, _ = nrrd.read(f'{self.root_dir}/{self.df["name"].iloc[0]}.nrrd')
        d, h, w = img.shape
      
        self.resize_d = resize_d
        self.resize_h = resize_h
        self.resize_w = resize_w

        # Update sizes based on resize
        self.d, self.h, self.w, = d//self.resize_d, h//self.resize_h, w//self.resize_w

        # Resize transform
        self.resize = tio.Resample((self.resize_d, self.resize_h, self.resize_w))

        # Augmentation transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.df['name'].iloc[index] + '.nrrd')
        img, _ = nrrd.read(path)

        img = torch.from_numpy(img)

        #  min-max normalized to the range between -1 and 1
        img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

        img = img.unsqueeze(0).float()
        img = self.resize(img)
       
        return {'data': img}

    def get_named_item(self, item_name, vmin=0, vmax=1, path=None, show=True):
        if path is None:
            path = os.path.join(self.root_dir, item_name + '.nrrd')
        else:
            path = os.path.join(path, item_name + '.nrrd')
        
        img, _ = nrrd.read(path)

        if show:
            show_item(img, vmin, vmax)
        else:
            return img
        
    def get_named_item_from_dataset(self, item_name):
        index = self.df[self.df['name'] == item_name].index[0]
        
        return self.__getitem__(index)

    def save_to_nrrd(self, item_name, item, save_path=None):
        # Transform the item to numpy array
        item = item.numpy()

        # Remove channel dimension if present
        if len(item.shape) == 4:
            item = item.squeeze(0)

        #  min-max normalized to the range between 0 and 1
        item = (item - item.min()) / (item.max() - item.min())
        
        if save_path is None:
            save_path = os.path.join(self.root_dir, item_name + '.nrrd')
        
        nrrd.write(save_path, item)

if __name__ == '__main__':
    dataset = AllCTsDataset(root_dir='data/allcts-global-gen-01', split='all')
    print(len(dataset))
    item = dataset.__getitem__(0)

    dataset.show_item(item['data'].squeeze(0).numpy())
    