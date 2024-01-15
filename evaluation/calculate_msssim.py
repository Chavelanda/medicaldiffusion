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
    if cfg.model.gpus != -1:
        torch.cuda.set_device(cfg.model.gpus)
    
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

    model = MSSSIM_3d(window_size=cfg.model.window_size, size_average=cfg.model.size_average, channel=cfg.dataset.image_channels).to(cfg.model.device)
 
    sum = 0.0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            img1, img2 = batch
            img1 = img1.to(cfg.model.device)
            img2 = img2.to(cfg.model.device)
            
            msssim = model(img1, img2)
            sum += msssim.item()
            wandb.log({'msssim': sum/(i+1), 'samples': (i+1)*cfg.model.batch_size})


if __name__ == "__main__":
    run()
