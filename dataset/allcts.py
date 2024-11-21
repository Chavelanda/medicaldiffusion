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
    def __init__(self, root_dir='data/AllCTs_nrrd_global', split='train', qs=None, 
                binarize=False, rescale=True, resample=1, conditioned=True, metadata_name='metadata.csv'):
        
        assert split in ['all', 'train', 'val', 'test', 'train-val'], 'Invalid split: {}'.format(split)

        self.root_dir = root_dir

        # Read the CSV file as a DataFrame
        self.df = pd.read_csv(os.path.join(root_dir, metadata_name))
        self.df['name'] = self.df['name'].astype(str)
        self.df = self.df.sort_values('name')

        # Take only the required split
        if split == 'train-val':
            self.df = self.df[self.df['split'] != 'test']
        elif split != 'all':
            self.df = self.df[self.df['split'] == split]

        self.input_df = self.df.copy()

        self.qs = qs
        if self.qs is not None:
            self.df = self.df.loc[self.df['quality'].isin(self.qs)]

        # Read one 3d image and define sizes
        img, _ = nrrd.read(f'{self.root_dir}/{self.df["name"].iloc[0]}.nrrd')
        d, h, w = img.shape
      
        self.resample = resample

        # Update sizes based on resample
        self.d, self.h, self.w, = d//self.resample, h//self.resample, w//self.resample

        # Resample transform
        self.resample_transform = tio.Resample(self.resample)

        # Binarize
        self.binarize = binarize

        # Rescale
        self.rescale = rescale

        # Condition
        self.conditioned = conditioned
        self.cond_dim = self.input_df['quality'].nunique()

        # One-hot encoding of the condition
        self.df = pd.get_dummies(self.df, columns=['quality'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        path = os.path.join(self.root_dir, entry['name'] + '.nrrd')
        
        img, _ = nrrd.read(path)
        img = torch.from_numpy(img)

        img = img.unsqueeze(0).float()
        img = self.resample_transform(img)

        if self.binarize:
            img = (img > 0.5).float()

        if self.rescale:
            #  min-max normalized to the range between -1 and 1
            img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

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
        cond = torch.squeeze(cond)
        cond = torch.unsqueeze(cond, 0)
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


class AllCts_MSSSIM(AllCTsDataset):

    def __init__(self, root_dir, split, resample=1, binarize=False, rescale=True, samples=1000, metadata_name='metadata.csv'):
        super().__init__(root_dir=root_dir, split=split, resample=resample, binarize=binarize, rescale=rescale, conditioned=False, metadata_name=metadata_name)

        self.samples = samples

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        index1 = np.random.randint(len(self.df))
        index2 = np.random.randint(len(self.df))

        img1 = super().__getitem__(index1)['data']
        img2 = super().__getitem__(index2)['data']

        return img1, img2


class AllCTsDatasetSS(AllCTsDataset):
    
    def __init__(self, root_dir='data/AllCTs_nrrd_global', split='train', binarize=False, rescale=True,
                 resample=1, metadata_name='metadata.csv',
                 recon_root_dir=None, recon_metadata_name='metadata.csv'):
        
        super().__init__(root_dir=root_dir, split=split, binarize=binarize, rescale=rescale, resample=resample, metadata_name=metadata_name)

        # Define transforms
        self.transforms = tio.Compose([
        # tio.RandomAffine(scales=(0.03, 0.03, 0), degrees=(
        # 0, 0, 3), translation=(4, 4, 0)),
        tio.RandomFlip(axes=(0,), flip_probability=1),
        ])

        # Prepare recon data
        if recon_root_dir is not None:
            self.recon_root_dir = recon_root_dir
            recon_dataset = AllCTsDataset(recon_root_dir, metadata_name=recon_metadata_name, split=split, binarize=binarize, resample=resample)
            self.recon_df = recon_dataset.df
            self.p_a = 1 # Probability to load the reconstructed image
            self.p_b = 0 # Probability to flip the image
        else:
            # If reconstructions are not available, use input df (no transformation happening)
            self.recon_root_dir = root_dir
            self.recon_df = self.df
            self.p_a = 0.
            self.p_b = 1.

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        path = os.path.join(self.root_dir, entry['name'] + '.nrrd')
        
        img, _ = nrrd.read(path)
        img = torch.from_numpy(img)

        img = img.unsqueeze(0).float()
        img = self.resample_transform(img)

        if self.binarize:
            img = (img > 0.5).float()

        if self.rescale:
            #  min-max normalized to the range between -1 and 1
            img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

        if np.random.rand() < self.p_a:       
            # Load and process as for normal image the reconstructed one
            recon_entry = self.recon_df.iloc[index]
            recon_path = os.path.join(self.recon_root_dir, recon_entry['name'] + '.nrrd')
            second_img = torch.from_numpy(nrrd.read(recon_path)[0])
            second_img = self.resample_transform(second_img.unsqueeze(0).float())
            if self.binarize:
                second_img = (second_img > 0.5).float()
            second_img = (second_img - second_img.min()) / (second_img.max() - second_img.min()) * 2 - 1
            
            # Possibly flip the image
            if np.random.rand() < self.p_b:
                second_img = self.transforms(second_img)
        else:
            second_img = self.transforms(img)

        data = torch.stack((img, second_img), dim=0)
        
        return {'data': data}

if __name__ == '__main__':
    # dataset = AllCTsDatasetSS(root_dir='data/allcts-global-128', split='all', recon_root_dir='data/allcts-vqgan-07')
    # print(len(dataset))
    # img = dataset.__getitem__(0)['data']
    # print(img.shape)
    # matplotlib.image.imsave('foo1.png', img[0, 0, 64, :,:])
    # matplotlib.image.imsave('foo2.png', img[1, 0, 64, :,:])
    # print('fooed')

    dataset = AllCTsDataset(root_dir='data/allcts-global-128', split='train-val', resample=1, qs=[2,3,4,5,6], rescale=False)
    print(len(dataset))
    img = dataset.__getitem__(10)['data']

    print(torch.max(img), torch.min(img))
    print(img.shape)
    # matplotlib.image.imsave('foo.png', img[0, 65])

    # print('fooed again')
    