import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from tqdm import tqdm
import wandb
import torch

from evaluation.pytorch_ssim.ssim import MSSSIM_3d
from train.get_dataset import get_dataset

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    print('Running MSSSIM evaluation with following config: {}'.format(cfg)) 
    wandb.init(project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, name=cfg.model.run_name)

    wandb.config.update(OmegaConf.to_container(cfg.dataset))
    wandb.config.update(OmegaConf.to_container(cfg.model))

    dataset, *_ = get_dataset(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.model.num_workers, pin_memory=True)
    
    device = 'cuda' if cfg.model.cuda else 'cpu'
    model = MSSSIM_3d(window_size=cfg.model.window_size, size_average=cfg.model.size_average, channel=cfg.dataset.image_channels, device=device).to(device)

    sum = 0.0
    i = 1

    print('Starting MSSSIM evaluation...')
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img1, img2 = batch
            img1, img2 = img1.to(device), img2.to(device)

            msssim = model(img1, img2)
            if torch.isnan(msssim):
                print('NaN encountered, skipping batch...')
            else:
                sum += msssim.item()
                i += 1
            
            wandb.log({'msssim': sum/(i), 'samples': (i)})


if __name__ == "__main__":
    with hydra.initialize(version_base=None, config_path="../config/"):
        cfg = hydra.compose(config_name='base_cfg', overrides=['model=msssim', 'dataset=allcts-msssim'])
    print('Running MSSSIM evaluation with following config:')
    print(cfg)

    run(cfg)

