from dataset.allcts import AllCTsDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def run():
    print('si vola')
    ds1 = AllCTsDataset('data/gen-02-em', split='all')
    ds2 = AllCTsDataset('data/allcts-global-128', split='train')

    print('dataset fatto')

    dl1 = DataLoader(ds1, batch_size=1, shuffle=False, num_workers=2)
    dl2 = DataLoader(ds2, batch_size=1, shuffle=False, num_workers=24)

    print('dataloader creato')

    coefficents_matrix = np.zeros((len(ds1), len(ds2)))

    print('matrix creata')

    with torch.no_grad():
        for i, b1 in enumerate(tqdm(dl1)):
            for j, b2 in enumerate(dl2):
                x1 = b1['data'].cuda()
                x2 = b2['data'].cuda()
                input = torch.cat((x1, x2), dim=0)
                input = input.flatten(start_dim=1)
                coeff = torch.corrcoef(input).cpu().numpy()
                #print(coeff)
                #print(coeff.shape)
                coefficents_matrix[i, j] = coeff[0, 1]

    np.save('gen-02-em-corr.npy', coefficents_matrix)

if __name__ == '__main__':
    run()