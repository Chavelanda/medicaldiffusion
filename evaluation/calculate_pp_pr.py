import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from dataset.get_dataset import get_dataset
from evaluation.fid.get_extractor import get_extractor
from evaluation.pp_pr import compute_pprecision_precall
from evaluation.prdc import compute_prdc

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    print('Calculating Probabilistic Precision and Probabilistic Recall with the following config:\n{}'.format(OmegaConf.to_yaml(cfg)))

    wandb.init(project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, name=cfg.model.run_name)

    wandb.config.update(OmegaConf.to_container(cfg.dataset))
    wandb.config.update(OmegaConf.to_container(cfg.model))

    dataset_real, dataset_gen, _ = get_dataset(cfg)
    dl_real = DataLoader(dataset_real, batch_size=cfg.model.batch_size, shuffle=True, num_workers=cfg.model.num_workers)
    dl_gen = DataLoader(dataset_gen, batch_size=cfg.model.batch_size, shuffle=True, num_workers=cfg.model.num_workers)

    extractor = get_extractor(cfg).eval()

    device = 'cuda' if cfg.model.cuda else 'cpu'

    extractor.to(device)

    real = []

    fake = []

    with torch.no_grad():
        for batch in tqdm(dl_real, leave=True):
            # Update FID with real features
            batch = batch['data'].to(device)

            feat = extractor(batch)
            real.append(feat.cpu().numpy())

        for batch in tqdm(dl_gen, leave=True):
            # Update FID with gen features
            batch = batch['data'].to(device)
            
            feat = extractor(batch)
            fake.append(feat.cpu().numpy())


        real = np.concatenate(real, axis=0)
        fake = np.concatenate(fake, axis=0)
        
        print('Computing pp and pr')
        pp, pr = compute_pprecision_precall(real, fake)

        print('Computing prdc')
        prdc = compute_prdc(real, fake)

        wandb.log({'probabilistic_precision': pp, 'probabilistic_recall': pr, 'precision': prdc['precision'], 'recall': prdc['recall'], 'density': prdc['density'], 'coverage': prdc['coverage']})

    print('Computation ended!')


if __name__ == '__main__':
    run()