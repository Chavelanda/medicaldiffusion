""" Taken and adapted from https://github.com/cyclomon/3dbraingen """

from torch.utils.data.dataset import Dataset
import torch
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

        img = img.unsqueeze(0)

        return {'data': img}


if __name__ == '__main__':
    print('Testing SKULLBREAKDataset')
    # Instantiate the dataset
    dataset = SKULLBREAKDataset()
    print('Dataset length: ', len(dataset))

    # Retrieve an item
    item = dataset[0]
    print("Shape of the item:", item['data'].shape)

    # Display a channel of the item as an image
    plt.imshow(item['data'].squeeze(0)[200], cmap='gray')
    plt.show()