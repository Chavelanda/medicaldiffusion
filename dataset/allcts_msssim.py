import numpy as np
from dataset.allcts import AllCTsDataset


class AllCts_MSSSIM(AllCTsDataset):

    def __init__(self, root_dir, split, resize_d=1, resize_h=1, resize_w=1, binarize=False, samples=1000, metadata_name='metadata.csv'):
        super().__init__(root_dir=root_dir, split=split, resize_d=resize_d, resize_h=resize_h, resize_w=resize_w, binarize=binarize, conditioned=False, metadata_name=metadata_name)

        self.samples = samples

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        index1 = np.random.randint(len(self.df))
        index2 = np.random.randint(len(self.df))

        img1 = super().__getitem__(index1)['data']
        img2 = super().__getitem__(index2)['data']

        return img1, img2
    

if __name__ == "__main__":
    dataset = AllCts_MSSSIM(root_dir='data/AllCTs_nrrd_global', split='train', resize_d=1, resize_h=1, resize_w=1)
    print(dataset[0][0].shape)