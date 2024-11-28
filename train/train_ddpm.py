import os

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.get_dataset import get_dataset
from ddpm import Unet3D, GaussianDiffusion, Trainer


def setup(rank, world_size):
    # Init distributed training
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class myDDP(torch.nn.parallel.DistributedDataParallel):
   def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def train(rank, world_size, cfg: DictConfig):
    # Setup distributed training
    setup(rank, world_size)

    results_folder = os.path.join(cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix, cfg.model.run_name)
    print("Setting results_folder to {}".format(results_folder))

    # automatically adjust learning rate following https://arxiv.org/abs/1706.02677
    batch_rate = world_size*cfg.model.gradient_accumulate_every
    actual_batch_size = cfg.model.batch_size*batch_rate
    actual_lr = cfg.model.train_lr*batch_rate
    print(f'Batch size is {cfg.model.batch_size}\tActual batch size is {actual_batch_size}')
    print(f'Learning rate is {cfg.model.train_lr}\tActual learning rate is {actual_lr}')

    with open_dict(cfg):
        cfg.model.actual_batch_size = actual_batch_size
        cfg.model.base_lr = cfg.model.train_lr
        cfg.model.train_lr = actual_lr

        cfg.model.results_folder = results_folder

    # Init wandb only in first process
    if rank == 0:
        wandb.init(project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, name=cfg.model.run_name)
        wandb.config.update(OmegaConf.to_container(cfg.dataset))
        wandb.config.update(OmegaConf.to_container(cfg.model))

    train_dataset, val_dataset, _ = get_dataset(cfg)

    # Define dataloaders
    sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    dl = DataLoader(train_dataset, batch_size=cfg.model.batch_size, pin_memory=True, num_workers=cfg.model.num_workers, sampler=sampler)
    val_dl = DataLoader(val_dataset, batch_size=cfg.model.batch_size, shuffle=False, pin_memory=True, num_workers=cfg.model.num_workers)

    # Define conditioning parameters
    if cfg.model.cond:
        cond_dim = train_dataset.cond_dim 
        use_class_cond = cfg.model.use_class_cond
    else:
        cond_dim = None
        use_class_cond = False

    unet3d = Unet3D(
            dim=cfg.model.dim, # It is the channel dimension after init_conv. Why do we use w?
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            cond_dim=cond_dim,
            use_class_cond=use_class_cond,
        ).to(rank)

    diffusion = GaussianDiffusion(
        unet3d,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        d=cfg.model.diffusion_d,
        h=cfg.model.diffusion_h,
        w=cfg.model.diffusion_w,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        # sampling_timesteps=cfg.model.sampling_timesteps,
        loss_type=cfg.model.loss_type,
    ).to(rank)

    ddp_diffusion = myDDP(diffusion, device_ids=[rank], find_unused_parameters=True)

    trainer = Trainer(
        ddp_diffusion,
        dl=dl,
        val_dl=val_dl,
        ema_decay=cfg.model.ema_decay,
        train_batch_size=cfg.model.batch_size,
        base_lr=cfg.model.base_lr,
        train_lr=cfg.model.train_lr,
        train_num_steps=cfg.model.train_num_steps,
        gradient_accumulate_every=cfg.model.gradient_accumulate_every,
        amp=cfg.model.amp,
        validate_save_and_sample_every=cfg.model.validate_save_and_sample_every,
        results_folder=cfg.model.results_folder,
        num_sample_rows=cfg.model.num_sample_rows,
        num_workers=cfg.model.num_workers,
        conditioned=cfg.model.cond,
        rank=rank,
    )

    if cfg.model.load_milestone:
        dist.barrier()
        trainer.load(cfg.model.load_milestone, map_location={'cuda:%d' % 0: 'cuda:%d' % rank})

    if rank == 0:
        trainer.train(log_fn=wandb.log)
        wandb.finish()
    else:
        trainer.train()

    cleanup()


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    world_size = cfg.model.n_gpus
    mp.spawn(train, args=(world_size, cfg), nprocs=world_size, join=True)


if __name__ == '__main__':
    run()

    # wandb.finish()

    # Incorporate GAN loss in DDPM training?
    # Incorporate GAN loss in UNET segmentation?
    # Maybe better if I don't use ema updates?
    # Use with other vqgan latent space (the one with more channels?)
