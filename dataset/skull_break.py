""" Taken and adapted from https://github.com/cyclomon/3dbraingen """

from torch.utils.data.dataset import Dataset
import torch
import torchio as tio
import matplotlib.pyplot as plt
import os
import glob
import nrrd


class SKULLBREAKDataset(Dataset):
    def __init__(self, root_dir='data/skull-break/train/nrrd/complete_skull'):
        self.root_dir = root_dir
        self.file_names = glob.glob(os.path.join(
            root_dir, './**/*.nrrd'), recursive=True)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]
        img, header = nrrd.read(path)

        img = torch.from_numpy(img)

        img = img.unsqueeze(0).float()

        return {'data': img}
    
class SKULLBREAKDatasetTriplet(Dataset):
    def __init__(self, root_dir='data/skull-break/train/nrrd/complete_skull', 
                 resize_d=1, resize_h=1, resize_w=1):
        self.root_dir = root_dir
        self.file_names = glob.glob(os.path.join(
            root_dir, './**/*.nrrd'), recursive=True)
        
        self.d = resize_d
        self.h = resize_h
        self.w = resize_w
        

        self.resize = tio.Resample((self.d, self.h, self.w))
        self.augment = tio.Compose([
            tio.RandomFlip(axes=(0, 2,), flip_probability=0.5), # 0 for depth, 1 for vertical, 2 for horizontal
            ])
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        path = self.file_names[index]
        img, _ = nrrd.read(path)
        
        img = torch.from_numpy(img)
        img = img.unsqueeze(0).float()
        img = self.resize(img)

        img2 = self.augment(img)
        img3 = self.augment(img)
        

        data = torch.stack((img, img2, img3), dim=0)

        
        return {'data': data}


if __name__ == '__main__':
    print('Testing SKULLBREAKDataset')
    # Instantiate the dataset
    dataset = SKULLBREAKDataset()
    print('Dataset length: ', len(dataset))

    # Retrieve an item
    item = dataset[0]
    print("Shape of the item:", item['data'].shape)

    print(item['data'].max(), item['data'].min())
    # Display a channel of the item as an image
    plt.imshow(item['data'].squeeze(0)[200], cmap='gray')
    plt.show()

    print('Testing SKULLBREAKDatasetTriplet')
    # Instantiate the dataset
    dataset = SKULLBREAKDatasetTriplet(d=4, h=4, w=4)
    print('Dataset length: ', len(dataset))

    # Retrieve an item
    item = dataset[0]
    print("Shape of the item:", item['data'].shape)

