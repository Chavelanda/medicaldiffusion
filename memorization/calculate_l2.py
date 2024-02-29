from dataset.allcts import AllCTsDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def run():
    name = 'allcts-global-128'
    ds1 = AllCTsDataset(f'data/{name}', split='test')
    ds2 = AllCTsDataset('data/allcts-global-128', split='train')

    print('dataset fatto')

    dl1 = DataLoader(ds1, batch_size=1, shuffle=False, num_workers=2)
    bs = 20
    dl2 = DataLoader(ds2, batch_size=bs, shuffle=False, num_workers=24)

    print('dataloader creato')

    distance_matrix = np.zeros((len(ds1), len(ds2)))

    print('matrix creata')

    with torch.no_grad():
        for i, b1 in enumerate(tqdm(dl1)):
            for j, b2 in enumerate(dl2):
                x1 = b1['data'].cuda()
                x2 = b2['data'].cuda()
                x1 = torch.flatten(x1, start_dim=1)
                x2 = torch.flatten(x2, start_dim=1)
                d = torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
                
                column_start = bs*j
                column_end = column_start + x2.shape[0]
                
                distance_matrix[i, column_start:column_end] = d.cpu()
            
    np.save(f'{name}-l2.npy', distance_matrix)

if __name__ == '__main__':
    run()