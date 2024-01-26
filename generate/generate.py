import os
import csv
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np

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

    ds, *_ = get_dataset(cfg) 

    # Define conditioning parameters
    if cfg.model.cond:
        cond_dim = ds.cond_dim 
        use_class_cond = cfg.model.use_class_cond
    else:
        cond_dim = None
        use_class_cond = False

    model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            cond_dim=cond_dim,
            use_class_cond=use_class_cond,
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

    conditioned = cfg.model.cond
    class_idx = cfg.model.class_idx
    random = False if class_idx is not None else True
    print('RANDOM', random)

    name_prefix = cfg.model.name_prefix if cfg.model.name_prefix else ''

    # Create metadata csv if not existing
    metadata_path = os.path.join(cfg.model.data_folder, 'metadata.csv')
    with open(metadata_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # If the file is empty, write the header
        if os.stat(metadata_path).st_size == 0:
            writer.writerow(['name', 'split', 'quality'])

    with torch.no_grad():
        for i in tqdm(range(steps), desc='Generating samples'):
            
            cond = ds.get_cond(batch_size=cfg.model.batch_size, random=random, class_idx=class_idx).cuda() if conditioned else None
            print('COND',cond)
            class_names = ds.get_class_name_from_cond(cond) if conditioned else ['null' for _ in range(cfg.model.batch_size)]

            samples = diffusion.sample(cond=cond, batch_size=cfg.model.batch_size).cpu()
            
            for j, sample in enumerate(samples):
                filename = f'{name_prefix}_{i*cfg.model.batch_size + j}_q{class_names[j]}'
                
                ds.save(filename, sample, save_path=cfg.model.data_folder)

                # Append metadata to csv
                with open(metadata_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([filename, 'train', class_names[j]])
            
            wandb.log({'step': i})
    

if __name__ == "__main__":
    run()
