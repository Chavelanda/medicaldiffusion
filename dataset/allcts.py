import os

import torch
from torch.utils.data import Dataset
import torchio as tio
import pandas as pd
import numpy as np
import nrrd

from dataset.utils import show_item

import matplotlib.image

class AllCTsDataset(Dataset):
    def __init__(self, root_dir='data/AllCTs_nrrd_global', split='train', binarize=False,
                 resize_d=1, resize_h=1, resize_w=1, conditioned=True, metadata_name='metadata.csv'):
        
        assert split in ['all', 'train', 'val', 'test', 'train-val'], 'Invalid split: {}'.format(split)

        self.root_dir = root_dir

        # Read the CSV file as a DataFrame
        self.df = pd.read_csv(os.path.join(root_dir, metadata_name))
        self.df['name'] = self.df['name'].astype(str)

        # Take only the required split
        if split == 'train-val':
            self.df = self.df[self.df['split'] != 'test']
        elif split != 'all':
            self.df = self.df[self.df['split'] == split]

        self.input_df = self.df.copy()

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

        # Binarize
        self.binarize = binarize

        # Condition
        self.conditioned = conditioned
        self.cond_dim = self.df['quality'].nunique()

        # One-hot encoding of the condition
        self.df = pd.get_dummies(self.df, columns=['quality'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        path = os.path.join(self.root_dir, entry['name'] + '.nrrd')
        
        img, _ = nrrd.read(path)
        img = torch.from_numpy(img)

        if self.binarize:
            img = (img > 0.5).float()

        #  min-max normalized to the range between -1 and 1
        img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

        img = img.unsqueeze(0).float()
        img = self.resize(img)

        if self.conditioned:
            quality_items = entry[entry.index.str.startswith('quality')]
            cond = quality_items.to_numpy().astype(float)
            cond = torch.tensor(cond).float()
        else:
            cond = None
       
        return {'data': img, 'cond': cond}
    
    def get_cond(self, batch_size=1, random=True, class_idx=None):
        # Tensor of zeros with one in a random position for each element of the batch
        cond = torch.zeros((batch_size, self.cond_dim))
        if random:
            cond[torch.arange(batch_size), torch.randint(0, self.cond_dim, (batch_size,))] = 1
        else:
            assert class_idx is not None, 'If random is False, class_idx must be specified'
            cond[:, class_idx] = 1
        return cond
    
    def get_class_name_from_cond(self, cond):
        quality = torch.argmax(cond, dim=1).cpu().numpy() + 2
        return [f'{q}' for q in quality]
        
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

    def get_header(self):
        return ['name', 'split', 'quality']

    def get_row(self, name, split, cond):
        cond_name = self.get_class_name_from_cond(cond)[0]
        return [name, split, cond_name]
    
    @staticmethod
    def save(item_name, item, save_path):
        # Transform the item to numpy array
        if isinstance(item, torch.Tensor):
            item = item.numpy()

        # Remove channel dimension if present
        if len(item.shape) > 3:
            item = np.squeeze(item)

        #  min-max normalize to the range between 0 and 1 and binarize
        item = (item - item.min()) / (item.max() - item.min())
        item = (item > 0.5).astype(float)
        
        save_path = os.path.join(save_path, item_name + '.nrrd')
        
        nrrd.write(save_path, item)

class AllCTsDatasetSS(AllCTsDataset):
    
    def __init__(self, root_dir='data/AllCTs_nrrd_global', split='train', binarize=False,
                 resize_d=1, resize_h=1, resize_w=1, metadata_name='metadata.csv'):
        
        super().__init__(root_dir=root_dir, split=split, binarize=binarize, resize_d=resize_d, resize_h=resize_h, resize_w=resize_w, metadata_name=metadata_name)

        # Define transforms
        self.transforms = tio.Compose([
        # tio.RandomAffine(scales=(0.03, 0.03, 0), degrees=(
        # 0, 0, 3), translation=(4, 4, 0)),
        tio.RandomFlip(axes=(0), flip_probability=1),
        ])

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        path = os.path.join(self.root_dir, entry['name'] + '.nrrd')
        
        img, _ = nrrd.read(path)
        img = torch.from_numpy(img)

        if self.binarize:
            img = (img > 0.5).float()

        #  min-max normalized to the range between -1 and 1
        img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

        img = img.unsqueeze(0).float()
        img = self.resize(img)
        
        flipped_img = self.transforms(img)

        data = torch.stack((img, flipped_img), dim=0)
        
        return {'data': data}

if __name__ == '__main__':
    dataset = AllCTsDataset(root_dir='data/allcts-global-128', split='all')
    print(len(dataset))
    img = dataset.__getitem__(100)['data'].numpy()
    print(dataset.df.head())
    
    print(img.shape)
    # matplotlib.image.imsave('foo.png', img[0, 64])
    