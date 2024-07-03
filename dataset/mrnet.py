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

from matplotlib import pyplot as plt

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
    preprocessing_transforms=None, 
    transforms=None,
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

        # Load metadata
        self.df = pd.read_csv(os.path.join(root_dir, metadata_name), 
            dtype={'name': str, 'abnormal': int, 'acl': int, 'meniscus': int, 'split': str})

         # Take only the required split
        if split != 'all':
            self.df = self.df[self.df['split'] == split]

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


if __name__ == '__main__':
    dataset = MRNetDataset(root_dir='data/mrnet', split='all')
    print(len(dataset))