import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from tqdm import tqdm
import wandb
import torch

from evaluation.pytorch_ssim.ssim import MSSSIM_3d
from dataset.allcts_msssim import AllCts_MSSSIM


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig): 
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.model.run_name)

    if not os.path.exists(cfg.model.results_folder):
        os.makedirs(cfg.model.results_folder)

    wandb.init(project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, name=cfg.model.run_name)

    wandb.config.update(OmegaConf.to_container(cfg.dataset))
    wandb.config.update(OmegaConf.to_container(cfg.model))

    dataset = AllCts_MSSSIM(root_dir=cfg.dataset.root_dir, split=cfg.dataset.split, samples=cfg.dataset.samples, resize_d=cfg.dataset.resize_d, resize_h=cfg.dataset.resize_h, resize_w=cfg.dataset.resize_w)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=True, num_workers=cfg.model.num_workers, pin_memory=True)

    if cfg.dataset.samples % cfg.model.batch_size != 0:
        raise ValueError('The number of samples must be divisible by the batch size.')

    model = MSSSIM_3d(window_size=cfg.model.window_size, size_average=cfg.model.size_average, channel=cfg.dataset.image_channels)
 
    sum = 0.0

    i = 1

    print('Starting MSSSIM evaluation...')
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img1, img2 = batch
            msssim = model(img1, img2)
            if torch.isnan(msssim):
                print('NaN encountered, skipping batch...')
            else:
                sum += msssim.item()
                i += 1
            
            wandb.log({'msssim': sum/(i), 'samples': (i)*cfg.model.batch_size})


if __name__ == "__main__":
    with hydra.initialize(version_base=None, config_path="../config/"):
        cfg = hydra.compose(config_name='base_cfg', overrides=['model=msssim', 'dataset=allcts-msssim'])
    print('Running MSSSIM evaluation with following config:')
    print(cfg)

    run(cfg)

