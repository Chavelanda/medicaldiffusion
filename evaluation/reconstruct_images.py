import os
import csv
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.get_dataset import get_dataset
import dataset.utils as utils
from vq_gan_3d.model import VQGAN
from vq_gan_3d.model.vqvae_upsampling import VQVAE_Upsampling

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    print('Reconstructing images with the following config:\n{}'.format(OmegaConf.to_yaml(cfg)))

    accelerator = cfg.model.accelerator

    dataset, *_ = get_dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.model.num_workers)

    # load the checkpoint file
    assert os.path.isfile(cfg.model.checkpoint_path), "Checkpoint file for VQGAN must be specified"
    
    ckpt_path = cfg.model.checkpoint_path
    vqgan = VQVAE_Upsampling.load_from_checkpoint(ckpt_path).to(accelerator)
    vqgan.eval()

    save_path = cfg.dataset.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create metadata csv if not existing
    metadata_path = os.path.join(save_path, 'metadata.csv')
    with open(metadata_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # If the file is empty, write the header
        if os.stat(metadata_path).st_size == 0:
            # get metadata header from dataset
            writer.writerow(dataset.get_header())

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            name = dataset.df.iloc[i, :]['name'] + '-recon'
            
            batch['data'] = batch['data'].to(accelerator)
            # class_name = dataset.get_class_name_from_cond(batch['cond'])[0]
            split = dataset.df.iloc[i, :]['split']

            reconstructed_batch = vqgan.test_step(batch, 0).cpu()

            # Upsampling 
            # reconstructed_batch = torch.nn.functional.interpolate(reconstructed_batch, size=(456,352,512), mode='trilinear')

            dataset.save(name, reconstructed_batch, save_path)

            # Append metadata to csv
            with open(metadata_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(dataset.get_row(name, split, batch['cond']))
                # writer.writerow([name, split, class_name])
    


if __name__ == '__main__':
    run()