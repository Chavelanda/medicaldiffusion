import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance

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

    extractor.to(device)

    fid = FrechetInceptionDistance(feature=extractor, normalize=True, compute_on_cpu=True, input_img_size=(cfg.dataset.image_channels, dataset_real.d, dataset_real.h, dataset_real.w))
    fid.reset()

    epochs = cfg.model.epochs

    pbar = trange(epochs)

    with torch.no_grad():
        for e in pbar:
            for batch in tqdm(dl_real, leave=True):
                # Update FID with real features
                batch = batch['data'].to(device)        
                fid.update(batch, real=True)

            for batch in tqdm(dl_gen, leave=True):
                # Update FID with gen features
                batch = batch['data'].to(device)#.to(torch.float64)
                fid.update(batch, real=False)

            # Compute FID
            fid_score = fid.compute()
            
            wandb.log({'fid': fid_score, 'epoch': e})
            pbar.set_description(f'FID score at epoch {e}: {fid_score}')


if __name__ == '__main__':
    run()