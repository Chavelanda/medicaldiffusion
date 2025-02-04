import os

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from diffusers import DDPMScheduler

import torch
from torch.utils.data import DataLoader

from dataset.get_dataset import get_dataset
from ddpm import Diffuser
from ddpm.callbacks import SampleAndSaveCallback


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    results_folder = os.path.join(cfg.model.results_folder, cfg.dataset.name, cfg.model.results_folder_postfix, cfg.model.run_name)
    os.makedirs(results_folder, exist_ok=True)
    print("Setting results_folder to {}".format(results_folder))

    # automatically adjust learning rate following https://arxiv.org/abs/1706.02677
    batch_rate = cfg.model.devices
    actual_batch_size = cfg.model.batch_size*batch_rate
    actual_lr = cfg.model.train_lr*batch_rate
    print(f'Batch size is {cfg.model.batch_size}\tActual batch size is {actual_batch_size}')
    print(f'Learning rate is {cfg.model.train_lr}\tActual learning rate is {actual_lr}')

    with open_dict(cfg):
        cfg.model.actual_batch_size = actual_batch_size
        cfg.model.base_lr = cfg.model.train_lr
        cfg.model.train_lr = actual_lr

        cfg.model.results_folder = results_folder

    train_dataset, val_dataset, sampler = get_dataset(cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,
                                  num_workers=cfg.model.num_workers, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
                                shuffle=False, num_workers=cfg.model.num_workers)

    # Define conditioning parameters
    if cfg.model.cond:
        cond_dim = train_dataset.cond_dim 
        use_class_cond = cfg.model.use_class_cond
    else:
        cond_dim = None
        use_class_cond = False

    # model checkpointing callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor=f'val/loss_l1',
                     save_top_k=1, mode='min', dirpath=results_folder, filename='best_val_l1-{epoch}-{step}'))
    callbacks.append(ModelCheckpoint(monitor=f'val/loss_l2',
                     save_top_k=1, mode='min', dirpath=results_folder, filename='best_val_l2-{epoch}-{step}'))
    callbacks.append(ModelCheckpoint(every_n_epochs=50, save_top_k=-1,
                     dirpath=results_folder, filename='train-{epoch}-{step}'))
    callbacks.append(ModelCheckpoint(every_n_epochs=1, save_top_k=1,
                     dirpath=results_folder, filename='last-train-{epoch}-{step}'))
    # progress bar callback
    callbacks.append(TQDMProgressBar(refresh_rate=50))
    # log lr callback
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    callbacks.append(SampleAndSaveCallback(results_folder=results_folder, sample_every_n_epochs=1, save_gif=True, save_image=False, save_func=train_dataset.save))

    noise_scheduler_class = DDPMScheduler

    # Resume training if needed, otherwise start from scratch
    ckpt_path = cfg.model.load_milestone
    if ckpt_path is not None:
        assert os.path.isfile(ckpt_path), f'Checkpoint is not None, but it is not a file: {ckpt_path}'
        print(f'Will resume from the ckpt {ckpt_path}')
    
    diffuser = Diffuser(
        vqvae_ckpt=cfg.model.vqvae_ckpt,
        noise_scheduler_class=noise_scheduler_class,
        in_channels=cfg.model.diffusion_num_channels,
        sample_d=cfg.model.diffusion_d,
        sample_h=cfg.model.diffusion_h,
        sample_w=cfg.model.diffusion_w,
        dim=cfg.model.dim,
        dim_mults=cfg.model.dim_mults,
        use_class_cond=use_class_cond,
        cond_dim=cond_dim,
        null_cond_prob=cfg.model.null_cond_prob,
        ema_decay=cfg.model.ema_decay,
        loss=cfg.model.loss_type,
        lr=cfg.model.train_lr,
        training_timesteps=cfg.model.timesteps,
    )
    
    # create wandb logger
    wandb_logger = pl.loggers.WandbLogger(name=cfg.model.run_name, project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, log_model=False)

    trainer = pl.Trainer(
        accelerator=cfg.model.accelerator,
        devices=cfg.model.devices,
        accumulate_grad_batches=cfg.model.gradient_accumulate_every,
        callbacks=callbacks,
        max_steps=cfg.model.train_num_steps,
        max_epochs=-1,
        precision=cfg.model.precision,
        logger=wandb_logger,
        strategy='ddp_find_unused_parameters_true',
        log_every_n_steps=50,
        check_val_every_n_epoch=cfg.model.check_val_every_n_epoch,
        fast_dev_run=False, 
    )

    # Updating wandb configs
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg.dataset))
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg.model))
        wandb_logger.experiment.config["ckpt_path"] = ckpt_path

    torch.set_float32_matmul_precision('medium')

    trainer.fit(diffuser, train_dataloader, val_dataloader, ckpt_path=ckpt_path)

if __name__ == '__main__':
    run()

    # wandb.finish()

