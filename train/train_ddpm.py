import os
from re import I
from ddpm import Unet3D, GaussianDiffusion, Trainer
import torch
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from dataset.get_dataset import get_dataset

import wandb

# NCCL_P2P_DISABLE=1 accelerate launch train/train_ddpm.py

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(cfg.model.gpus)
    with open_dict(cfg):
        cfg.model.results_folder = os.path.join(
            cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix, cfg.model.run_name)

    wandb.init(project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, name=cfg.model.run_name)

    wandb.config.update(OmegaConf.to_container(cfg.dataset))
    wandb.config.update(OmegaConf.to_container(cfg.model))

    train_dataset, val_dataset, _ = get_dataset(cfg)

    # Define conditioning parameters
    if cfg.model.cond:
        cond_dim = train_dataset.cond_dim 
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
        # sampling_timesteps=cfg.model.sampling_timesteps,
        loss_type=cfg.model.loss_type,
        # objective=cfg.objective
    ).cuda()

    wandb.watch(diffusion)

    trainer = Trainer(
        diffusion,
        cfg=cfg,
        dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=cfg.model.batch_size,
        validate_save_and_sample_every=cfg.model.validate_save_and_sample_every,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        ema_decay=cfg.model.ema_decay,
        amp=cfg.model.amp,
        num_sample_rows=cfg.model.num_sample_rows,
        results_folder=cfg.model.results_folder,
        num_workers=cfg.model.num_workers,
        conditioned=cfg.model.cond,
    )

    if cfg.model.load_milestone:
        trainer.load(cfg.model.load_milestone)

    trainer.train(log_fn=wandb.log)
    wandb.finish()

if __name__ == '__main__':
    run()

    # wandb.finish()

    # Incorporate GAN loss in DDPM training?
    # Incorporate GAN loss in UNET segmentation?
    # Maybe better if I don't use ema updates?
    # Use with other vqgan latent space (the one with more channels?)
