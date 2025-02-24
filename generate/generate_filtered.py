import os
import csv
from collections import OrderedDict
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np

from tqdm import tqdm
import wandb
import torch

from ddpm import GaussianDiffusion, Unet3D
from dataset.get_dataset import get_dataset


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
        cond_scale = cfg.model.cond_scale
        print(f'Random class: {random}')     
    else:
        cond_dim = None
        use_class_cond = False
        cond_scale = 1

    channels=cfg.model.diffusion_num_channels
    d=cfg.model.diffusion_d
    h=cfg.model.diffusion_h
    w=cfg.model.diffusion_w

    model = Unet3D(
            dim=cfg.model.dim,
            dim_mults=cfg.model.dim_mults,
            channels=channels,
            cond_dim=cond_dim,
            use_class_cond=use_class_cond,
        ).cuda()
    
    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        d=d,
        h=h,
        w=w,
        channels=channels,
        timesteps=cfg.model.timesteps,
    ).cuda()

    data = torch.load(cfg.model.milestone)
    
    # Remove 'module.' prefix. Necessary if trained with data parallel
    ema_state_dict = OrderedDict()
    for k, v in data['ema'].items():
        new_key = k.replace("module.", "")  
        ema_state_dict[new_key] = v
    
    diffusion.load_state_dict(ema_state_dict)
    
    # Create metadata csv if not existing
    name_prefix = cfg.model.name_prefix if cfg.model.name_prefix else ''
    metadata_path = os.path.join(cfg.model.data_folder, 'metadata.csv')
    with open(metadata_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # If the file is empty, write the header
        if os.stat(metadata_path).st_size == 0:
            # get metadata header from dataset
            writer.writerow(ds.get_header())

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

    assert (m == 1 and cfg.model.train_latents is None) or (m > 1 and cfg.model.train_latents), "If m > 1, path to train latents must be specified. Otherwise train_latents must be left empty!"
    if m > 1:
        train_latents = torch.from_numpy(np.load(cfg.model.train_latents)).float().cuda()

    print(f'Number of samples to generate: {n_samples}')
    print(f'Batch size, m, generated per batch: {batch_size}, {m}, {ex_step}')
    print(f'Steps {steps}')
    print(f'Generated last step, last step batch size: {n_samples - (steps - 1)*ex_step}, {last_cond}')
    print(f'Generating for class idx {class_idx}')
    print(f'Generating with conditional scale {cond_scale}')
    
    with torch.no_grad():
        for i in tqdm(range(steps), desc='Generating samples'):
            if i == steps - 1:
                batch_size = last_cond

            cond = ds.get_cond(batch_size=batch_size, random=random, class_idx=class_idx).cuda() if conditioned else None
            # print('COND',cond)
            # class_names = ds.get_class_name_from_cond(cond) if conditioned else ['null' for _ in range(cfg.model.batch_size)]
            
            # Generate bs latents
            latents = diffusion.p_sample_loop((batch_size, channels, d, h, w), cond=cond, cond_scale=cond_scale)
            
            latents = (((latents + 1.0) / 2.0) * (diffusion.vqgan.codebook.embeddings.max() -
                                                  diffusion.vqgan.codebook.embeddings.min())) + diffusion.vqgan.codebook.embeddings.min()
            
            # Quantize latents
            vq_output = diffusion.vqgan.codebook(latents)
            latents, encodings = torch.flatten(vq_output['embeddings'], start_dim=1), vq_output['encodings'].view((-1, m, d, h, w))

            if m > 1:
                # Compute distance with train latents
                distance_matrix = torch.cdist(latents, train_latents, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')

                # # FARE MAESTRIE PER CAPIRE DI CHE QS SONO I NEAREST LATENTS
                # print('WOOOOOOOOOOOOOOOO')
                # print('Shape latents ', latents.shape)
                # print('Shape distance matrix ', distance_matrix.shape)
                # print('shape index min ', torch.argmin(distance_matrix, axis=1).shape)
                # min_distance_matrix = torch.min(distance_matrix, axis=1)
                # print('Min distance tuple ', min_distance_matrix)
                # sorted_idx = torch.argsort(min_distance_matrix[0])
                # print('Sorted idx for min distance tuple ', sorted_idx)
                # sorted_min_distance_indexes = min_distance_matrix[1][sorted_idx]
                # print('Sorted min distance indexes ', sorted_min_distance_indexes)

                # dataset = AllCTsDataset(
                #     root_dir='./data/allcts-global-128/',
                #     metadata_name='metadata.csv',
                #     split='train-val',
                #     binarize=True,
                # )

                # print(dataset.df.iloc[sorted_min_distance_indexes.cpu(), :])

                # return None
                # # FINITE MAESTRIE

                # Select farthest latent every m samples
                nn, _ = torch.min(distance_matrix.view((-1, m, train_latents.shape[0])), 2)
                selected_idx = torch.argmax(nn, dim=1)
            else:
                selected_idx = 0
            
            encoding = encodings[torch.arange(encodings.shape[0]), selected_idx]
            
            # Decode the latent
            samples = diffusion.vqgan.decode(encoding, quantize=False).cpu()

            # Save the images
            for j, sample in enumerate(samples):
                filename = f'{name_prefix}_{i*ex_step + j}'
                
                ds.save(filename, sample, cfg.model.data_folder)

                # Append metadata to csv
                with open(metadata_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(ds.get_row(filename, 'train', cond[j].cpu()))
            
            wandb.log({'step': i})
    

if __name__ == "__main__":
    run()
