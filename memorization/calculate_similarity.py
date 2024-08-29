import os
import csv
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.get_dataset import get_dataset
import dataset.utils as utils
from self_supervised.ss import SS

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    print('Calculating similarity:\n{}'.format(OmegaConf.to_yaml(cfg)))

    dataset1, dataset2, _ = get_dataset(cfg)
    
    dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=False, num_workers=cfg.model.num_workers)
    dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=False, num_workers=cfg.model.num_workers)

    accelerator = cfg.model.accelerator

    if cfg.model.checkpoint_path:
        model = SS.load_from_checkpoint(cfg.model.checkpoint_path).to(accelerator)
    else:
        model = Extractor().to(accelerator)

    save_path = cfg.model.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filepath = os.path.join(save_path, cfg.model.name)

    coefficents_matrix = np.zeros((len(dataset1), len(dataset2)))

    with torch.no_grad():
        for i, b1 in enumerate(tqdm(dataloader1)):
            for j, b2 in enumerate(dataloader2):
                b1['data'] = b1['data'].to(accelerator)
                b2['data'] = b2['data'].to(accelerator)

                x1 = model.test_step(b1)
                x2 = model.test_step(b2)

                x = torch.cat((x1, x2), dim=0)
                x = x.flatten(start_dim=1)
                coeff = torch.corrcoef(x).cpu().numpy()
                #print(coeff)
                #print(coeff.shape)
                coefficents_matrix[i, j] = coeff[0, 1]

    np.save(filepath, coefficents_matrix)
    

class Extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, b):
        return b['data']

    def test_step(self, batch):
        return self.forward(batch)


if __name__ == '__main__':
    run()