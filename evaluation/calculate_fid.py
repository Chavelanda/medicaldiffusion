import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from dataset.get_dataset import get_dataset
from evaluation.fid.get_extractor import get_extractor


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    print('Calculating FID with the following config:\n{}'.format(OmegaConf.to_yaml(cfg)))

    wandb.init(project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, name=cfg.model.run_name)

    wandb.config.update(OmegaConf.to_container(cfg.dataset))
    wandb.config.update(OmegaConf.to_container(cfg.model))

    dataset_real, dataset_gen, _ = get_dataset(cfg)
    dl_real = DataLoader(dataset_real, batch_size=cfg.model.batch_size, shuffle=True, num_workers=cfg.model.num_workers)
    dl_gen = DataLoader(dataset_gen, batch_size=cfg.model.batch_size, shuffle=True, num_workers=cfg.model.num_workers)

    extractor = get_extractor(cfg).eval()

    device = 'cuda' if cfg.model.cuda else 'cpu'

    input_img_size = (cfg.dataset.image_channels, cfg.dataset.d, cfg.dataset.h, cfg.dataset.w)

    fid = FrechetInceptionDistance(feature=extractor, normalize=True, compute_on_cpu=True, input_img_size=input_img_size).to(device)

    kid = KernelInceptionDistance(feature=extractor, compute_on_cpu=True, subsets=150, subset_size=min(len(dataset_gen), 500)).to(device)

    epochs = cfg.model.epochs

    pbar = trange(epochs)

    with torch.no_grad():
        for e in pbar:
            for batch in tqdm(dl_real, leave=True):
                # Update FID with real features
                batch = batch['data'].to(device)       
                fid.update(batch, real=True)
                kid.update(batch, real=True)

            for batch in tqdm(dl_gen, leave=True):
                # Update FID with gen features
                batch = batch['data'].to(device)
                fid.update(batch, real=False)
                kid.update(batch, real=False)

            # Compute FID
            fid_score = fid.compute()
            kid_mean, kid_std = kid.compute()
            
            wandb.log({'fid': fid_score, 'kid_mean': kid_mean, 'kid_std': kid_std, 'epoch': e})

    print('Computation ended!')


if __name__ == '__main__':
    run()