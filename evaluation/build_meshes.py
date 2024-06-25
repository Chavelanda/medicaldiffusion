import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.get_dataset import get_dataset
import dataset.utils as utils

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    print('Reconstructing images with the following config:\n{}'.format(OmegaConf.to_yaml(cfg.dataset)))

    dataset, *_ = get_dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.model.num_workers)

    save_path = os.path.join(cfg.dataset.root_dir, 'meshes/')

    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    spacing = cfg.dataset.spacing

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            name = dataset.df.iloc[i, :]['name']
            
            image = batch['data']

            utils.build_mesh(torch.transpose(torch.squeeze(image), 0, 1).detach().cpu().numpy(), threshold=0, name=name, output_folder=save_path, spacing=spacing)

    


if __name__ == '__main__':
    run()