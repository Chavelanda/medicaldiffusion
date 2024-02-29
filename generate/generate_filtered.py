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
import subprocess

 

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
    conditioned = cfg.model.cond
    if conditioned:
        cond_dim = ds.cond_dim 
        use_class_cond = cfg.model.use_class_cond
        class_idx = cfg.model.class_idx
        random = False if class_idx is not None else True
        print(f'Random class: {random}')
    else:
        cond_dim = None
        use_class_cond = False

    # Create model
    image_size = cfg.model.diffusion_img_size
    channels = cfg.model.diffusion_num_channels
    num_frames = cfg.model.diffusion_depth_size

    model = Unet3D(
            dim=image_size,
            dim_mults=cfg.model.dim_mults,
            channels=channels,
            cond_dim=cond_dim,
            use_class_cond=use_class_cond,
        ).cuda()
    
    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=image_size,
        num_frames=num_frames,
        channels=channels,
        timesteps=cfg.model.timesteps,
    ).cuda()

    data = torch.load(cfg.model.milestone)
    diffusion.load_state_dict(data['ema'])
    
    # Create metadata csv if not existing
    name_prefix = cfg.model.name_prefix if cfg.model.name_prefix else ''
    metadata_path = os.path.join(cfg.model.data_folder, 'metadata.csv')
    with open(metadata_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # If the file is empty, write the header
        if os.stat(metadata_path).st_size == 0:
            writer.writerow(['name', 'split', 'quality'])

    # Set up generation 
    n_samples = cfg.model.n_samples
    batch_size = cfg.model.batch_size
    steps = n_samples // batch_size

    # Set up filtered generation
    m = cfg.model.m
    assert batch_size % m == 0, f'Batch size must be divisible by m! You have {batch_size} and {m}'
    ex_step = int(batch_size / m)
    steps = int(-(n_samples // -ex_step)) # ceiling of a / b
    if n_samples % ex_step == 0:
        last_cond = batch_size
    else:
        last_cond = int(n_samples % ex_step * m)

    train_latents = torch.from_numpy(np.load(f'generate/latents/q{class_idx+2}_latents.npy')).float().cuda()

    print(f'Number of samples to generate: {n_samples}')
    print(f'Batch size, m, generated per batch: {batch_size}, {m}, {ex_step}')
    print(f'Steps {steps}')
    print(f'Generated last step, last step batch size: {n_samples - (steps - 1)*ex_step}, {last_cond}')
    print(f'Generating for class idx {class_idx}')
    
    with torch.no_grad():
        for i in tqdm(range(steps), desc='Generating samples'):
            if i == steps - 1:
                batch_size = last_cond

            cond = ds.get_cond(batch_size=batch_size, random=random, class_idx=class_idx).cuda() if conditioned else None
            # print('COND',cond)
            class_names = ds.get_class_name_from_cond(cond) if conditioned else ['null' for _ in range(cfg.model.batch_size)]
            
            # Generate bs latents
            latents = diffusion.p_sample_loop((batch_size, channels, num_frames, image_size, image_size), cond=cond, cond_scale=1)
            
            latents = (((latents + 1.0) / 2.0) * (diffusion.vqgan.codebook.embeddings.max() -
                                                  diffusion.vqgan.codebook.embeddings.min())) + diffusion.vqgan.codebook.embeddings.min()
            
            # Quantize latents
            vq_output = diffusion.vqgan.codebook(latents)
            latents, encodings = torch.flatten(vq_output['embeddings'], start_dim=1), vq_output['encodings'].view((-1, m, num_frames, image_size, image_size))

            # Compute distance with train latents
            distance_matrix = torch.cdist(latents, train_latents, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')

            # Select farthest latent every m samples
            nn, _ = torch.min(distance_matrix.view((-1, m, train_latents.shape[0])), 2)
            selected_idx = torch.argmax(nn, dim=1)

            encoding = encodings[torch.arange(encodings.shape[0]), selected_idx]
            
            # Decode the latent
            samples = diffusion.vqgan.decode(encoding, quantize=False).cpu()

            # Save the images
            for j, sample in enumerate(samples):
                filename = f'{name_prefix}_{i*ex_step + j}_q{class_names[j]}'
                
                ds.save(filename, sample, save_path=cfg.model.data_folder)

                # Append metadata to csv
                with open(metadata_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([filename, 'train', class_names[j]])
            
            wandb.log({'step': i})
    

if __name__ == "__main__":
    run()
