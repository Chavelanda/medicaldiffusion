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
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix, cfg.model.run_name)

    if not os.path.exists(cfg.model.results_folder):
        os.makedirs(cfg.model.results_folder)

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
        # sampling_timesteps=cfg.model.sampling_timesteps,
        loss_type=cfg.model.loss_type,
        # objective=cfg.objective
    ).cuda()

    if cfg.model.load_milestone:
        data = torch.load(cfg.model.load_milestone)
        diffusion.load_state_dict(data['model'])

    dataset, *_ = get_dataset(cfg)

    n_samples = cfg.model.n_samples
    steps = n_samples // cfg.model.batch_size

    with torch.no_grad():
        for i in tqdm(range(steps), desc='Generating samples'):
            samples = diffusion.sample(batch_size=cfg.model.batch_size).cpu()
            for j, sample in enumerate(samples):
                filename = f'{i*cfg.model.batch_size + j}'
                save_path = os.path.join(cfg.model.results_folder, filename + '.nrrd')
                dataset.save_to_nrrd(filename, sample, save_path=save_path)
            wandb.log({'step': i})
    

if __name__ == "__main__":
    run()
