import os
import torchio as tio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
from scipy import signal
import random
import math
import argparse

import matplotlib

# TODO: Normalize to -1 and 1


def reformat_label(label):
    if label == 1:
        label = torch.FloatTensor([1])
    elif label == 0:
        label = torch.FloatTensor([0])
    return label


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.CropOrPad(target_shape=(256, 256, 32))
])

TRAIN_TRANSFORMS = tio.Compose([
    # tio.RandomAffine(scales=(0.03, 0.03, 0), degrees=(
    # 0, 0, 3), translation=(4, 4, 0)),
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

VAL_TRANSFORMS = None


class MRNetDataset(Dataset):
    def __init__(self, 
    root_dir, 
    # task, 
    # plane, 
    split='train',
    conditioned=True,
    metadata_name='metadata.csv',
    # fold=0
    ):
        assert split in ['all', 'train', 'val'], 'Invalid split: {}'.format(split)
        
        super().__init__()
        # self.task = task
        # self.plane = plane
        self.root_dir = root_dir
        self.split = split
        self.conditioned = conditioned
        self.preprocessing_transforms = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS if split == 'train' else VAL_TRANSFORMS
        # self.fold = fold
        self.d = 32
        self.h = 256
        self.w = 256

        # Load metadata
        self.df = pd.read_csv(os.path.join(root_dir, metadata_name), 
            dtype={'name': str, 'abnormal': int, 'acl': int, 'meniscus': int, 'split': str})

         # Take only the required split
        if split != 'all':
            self.df = self.df[self.df['split'] == split]

        # Set dimension of cond
        self.cond_dim = 3

    def __len__(self):
        return len(self.df)

    @property
    def sample_weight(self):
        class_sample_count = np.unique(self.labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[self.labels]
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        data_path = os.path.join(self.root_dir, entry['name'] + '.npy')

        img = np.load(data_path)
        img = torch.from_numpy(img)

        # Add channel dimension
        img = img.unsqueeze(0)

        img = torch.permute(img, (0, 2, 3, 1))  # Use C, H, W, D for TorchIO
        if self.preprocessing_transforms:
            img = self.preprocessing_transforms(img)
        if self.transforms:
            img = self.transforms(img)
        # Revert to C, D, H, W for PyTorch
        img = torch.permute(img, (0, 3, 1, 2))

        if self.conditioned:
            cond = torch.from_numpy(entry[1:4].astype(np.float32).to_numpy())
        else:
            cond = None

        return {'data': img, 'cond': cond}

    def get_header(self):
        return self.df.columns

    def get_row(self, name, split, cond):
        cond = torch.squeeze(cond).type(torch.int8)
        return [name] + cond.numpy().tolist() + [split]

    def get_cond(self, batch_size=1, random=True, class_idx=None):
        # Tensor of zeros with one in a random position for each element of the batch
        cond = torch.zeros((batch_size, self.cond_dim))
        if random:
            cond = torch.ranint(0, 2, (batch_size, self.cond_dim))
        else:
            assert class_idx is not None, 'If random is False, class_idx must be specified'
            cond[:, class_idx] = 1
        return cond

    @staticmethod
    def save(name, item, path):
        # Transform the item to numpy array
        if isinstance(item, torch.Tensor):
            item = item.numpy()

        # Remove channel dimension if present
        if len(item.shape) > 3:
            item = np.squeeze(item)

        save_path = os.path.join(path, name)

        np.save(save_path, item)


class MRNetDatasetMSSSIM(MRNetDataset):
    def __init__(self, 
    root_dir, 
    # task, 
    # plane, 
    split='train',
    metadata_name='metadata.csv',
    # fold=0,
    samples=1000,
    ):
        assert split in ['all', 'train', 'val'], 'Invalid split: {}'.format(split)
        
        super().__init__(root_dir, split=split, conditioned=False, metadata_name=metadata_name)

        self.samples = samples
    
    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        index1 = np.random.randint(len(self.df))
        index2 = np.random.randint(len(self.df))

        img1 = super().__getitem__(index1)['data']
        img2 = super().__getitem__(index2)['data']

        return img1, img2


class MRNetDatasetSS(MRNetDataset):
    
    def __init__(self, 
    root_dir, 
    # task, 
    # plane, 
    split='train',
    metadata_name='metadata.csv',
    # fold=0
    ):
        super().__init__(root_dir, split=split, metadata_name=metadata_name)

        # Define transforms
        self.ss_transforms = tio.Compose([
        # tio.RandomAffine(scales=(0.03, 0.03, 0), degrees=(
        # 0, 0, 3), translation=(4, 4, 0)),
        tio.RandomFlip(axes=(1), flip_probability=1),
        ])

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        data_path = os.path.join(self.root_dir, entry['name'] + '.npy')

        img = np.load(data_path)
        img = torch.from_numpy(img)

        # Add channel dimension
        img = img.unsqueeze(0)

        img = torch.permute(img, (0, 2, 3, 1))  # Use C, H, W, D for TorchIO
        if self.preprocessing_transforms:
            img = self.preprocessing_transforms(img)

        flipped_img = self.ss_transforms(img)

        # Revert to C, D, H, W for PyTorch
        img = torch.permute(img, (0, 3, 1, 2))
        flipped_img = torch.permute(flipped_img, (0, 3, 1, 2))

        data = torch.stack((img, flipped_img), dim=0)

        return {'data': data}



if __name__ == '__main__':
    dataset = MRNetDatasetSS(root_dir='data/mrnet', split='val')
    print(len(dataset))
    img = dataset.__getitem__(0)['data']
    print(img.shape)
    matplotlib.image.imsave('foo1.png', img[0, 0, 6])
    matplotlib.image.imsave('foo2.png', img[1, 0, 6])
    print('fooed')