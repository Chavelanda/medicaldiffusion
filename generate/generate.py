import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from tqdm import tqdm
import wandb
import torch

from ddpm import GaussianDiffusion, Unet3D
from train.get_dataset import get_dataset

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    with open_dict(cfg):
        cfg.model.data_folder = os.path.join(
            cfg.model.data_folder, cfg.model.run_name)

    if not os.path.exists(cfg.model.data_folder):
        os.makedirs(cfg.model.data_folder)

    wandb.init(project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, name=cfg.model.run_name)

    wandb.config.update(OmegaConf.to_container(cfg.dataset))
    wandb.config.update(OmegaConf.to_container(cfg.model))

    model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
        ).cuda()
    
    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
    ).cuda()

    data = torch.load(cfg.model.milestone)
    diffusion.load_state_dict(data['ema'])

    n_samples = cfg.model.n_samples
    steps = n_samples // cfg.model.batch_size

    conditioned = cfg.model.conditioned
    class_idx = cfg.model.class_idx
    if class_idx:
        random = False
        class_name = f'_{class_idx}'
    else:
        class_name = ''

    ds, *_ = get_dataset(cfg) 

    name_prefix = cfg.model.name_prefix if name_prefix else ''

    with torch.no_grad():
        for i in tqdm(range(steps), desc='Generating samples'):
            
            cond = ds.get_cond(batch_size=cfg.model.batch_size, random=random, class_idx=class_idx).cuda() if conditioned else None    
            samples = diffusion.sample(cond=cond, batch_size=cfg.model.batch_size).cpu()
            
            for j, sample in enumerate(samples):
                filename = f'{name_prefix}{i*cfg.model.batch_size + j}{class_name}'
                save_path = os.path.join(cfg.model.data_folder, filename + '.nrrd')
                ds.save(filename, sample, save_path=save_path)
            
            wandb.log({'step': i})
    

if __name__ == "__main__":
    run()
