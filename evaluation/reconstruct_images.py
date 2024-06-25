import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.get_dataset import get_dataset
import dataset.utils as utils
from vq_gan_3d.model import VQGAN

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    print('Reconstructing images with the following config:\n{}'.format(OmegaConf.to_yaml(cfg)))

    accelerator = cfg.model.accelerator

    dataset, *_ = get_dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.model.num_workers)

    # load the checkpoint file
    assert os.path.isfile(cfg.model.checkpoint_path), "Checkpoint file for VQGAN must be specified"
    
    ckpt_path = cfg.model.checkpoint_path
    vqgan = VQGAN.load_from_checkpoint(ckpt_path).to(accelerator)
    vqgan.eval()

    save_path = cfg.dataset.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            name = dataset.df.iloc[i, :]['name'] + '-recon'

            batch['data'] = batch['data'].to(accelerator)
            reconstructed_batch = vqgan.test_step(batch, 0).cpu()
            print(reconstructed_batch.shape, name)
            dataset.save(name, reconstructed_batch, save_path)
            break

    


if __name__ == '__main__':
    run()