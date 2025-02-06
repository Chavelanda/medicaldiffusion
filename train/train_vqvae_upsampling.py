"Adapted from https://github.com/SongweiGe/TATS"

import os

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
import torch
from torch.utils.data import DataLoader

from vq_gan_3d.model.vqvae_upsampling import VQVAEUpsampling
from dataset.get_dataset import get_dataset


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    
    pl.seed_everything(cfg.model.seed)

    train_dataset, val_dataset, sampler = get_dataset(cfg)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,
                                  num_workers=cfg.model.num_workers, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
                                shuffle=False, num_workers=cfg.model.num_workers)
    
    base_dir = os.path.join(cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix, cfg.model.run_name)
    print("Setting default_root_dir to {}".format(base_dir))

    # automatically adjust learning rate following https://arxiv.org/abs/1706.02677
    batch_rate = cfg.model.devices*cfg.model.accumulate_grad_batches
    actual_batch_size = cfg.model.batch_size*batch_rate
    actual_lr = cfg.model.lr*batch_rate
    print(f'Batch size is {cfg.model.batch_size}\tActual batch size is {actual_batch_size}')
    print(f'Learning rate is {cfg.model.lr}\tActual learning rate is {actual_lr}')

    with open_dict(cfg):
        cfg.model.actual_batch_size = actual_batch_size
        cfg.model.base_lr = cfg.model.lr
        cfg.model.lr = actual_lr
        
        cfg.model.default_root_dir = base_dir

        # Set dataset sizes
        cfg.dataset.d = train_dataset.d
        cfg.dataset.h = train_dataset.h
        cfg.dataset.w = train_dataset.w

    # model checkpointing callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                     save_top_k=1, mode='min', dirpath=base_dir, filename='best_val-{epoch}-{step}'))
    callbacks.append(ModelCheckpoint(every_n_epochs=30, save_top_k=-1,
                     dirpath=base_dir, filename='train-{epoch}-{step}'))
    callbacks.append(ModelCheckpoint(every_n_epochs=1, save_top_k=1,
                     dirpath=base_dir, filename='last-{epoch}-{step}'))
    # progress bar callback
    callbacks.append(TQDMProgressBar(refresh_rate=50))
    # log lr callback
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    # load the most recent checkpoint file
    ckpt_path = None
    if cfg.model.resume and os.path.isfile(cfg.model.checkpoint_path):
        assert os.path.isfile(cfg.model.checkpoint_path), f'Resume is {cfg.model.resume} but {cfg.model.checkpoint_path} is not a checkpoint file!'
        ckpt_path = cfg.model.checkpoint_path
        print(f'Will resume from the recent ckpt {ckpt_path}')
    
    model = VQVAEUpsampling(embedding_dim=cfg.model.embedding_dim,
                            n_codes=cfg.model.n_codes,
                            n_hiddens=cfg.model.n_hiddens,
                            downsample=cfg.model.downsample,
                            image_channels=cfg.dataset.image_channels,
                            norm_type=cfg.model.norm_type,
                            padding_type=cfg.model.padding_type,
                            num_groups=cfg.model.num_groups,
                            no_random_restart=cfg.model.no_random_restart,
                            restart_thres=cfg.model.restart_thres,
                            d=cfg.dataset.d,
                            h=cfg.dataset.h,
                            w=cfg.dataset.w,
                            gan_feat_weight=cfg.model.gan_feat_weight,
                            disc_channels=cfg.model.disc_channels,
                            disc_layers=cfg.model.disc_layers,
                            disc_loss_type=cfg.model.disc_loss_type,
                            image_gan_weight=cfg.model.image_gan_weight,
                            video_gan_weight=cfg.model.video_gan_weight,
                            perceptual_weight=cfg.model.perceptual_weight,
                            l1_weight=cfg.model.l1_weight,
                            gradient_clip_val=cfg.model.gradient_clip_val,
                            discriminator_iter_start=cfg.model.discriminator_iter_start,
                            lr=cfg.model.lr,
                            base_lr=cfg.model.base_lr, 
                            original_d=train_dataset.original_d, 
                            original_h=train_dataset.original_h, 
                            original_w=train_dataset.original_w, 
                            architecture=cfg.model.architecture, 
                            architecture_down=cfg.model.architecture_down,
                            model_parallelism=cfg.model.model_parallelism)
        
    # Setup model parallelism
    if cfg.model.model_parallelism:
        # Check there are enough available GPUs
        assert torch.cuda.device_count() >= cfg.model.devices * 2, f"When model parallelism is active, the number of available GPUs should be at least double the number of processes. Instead {torch.cuda.device_count()} < {cfg.model.devices}*2"
        devices = [i * 2 for i in range(cfg.model.devices)]
    else:
        devices = cfg.model.devices

    # create wandb logger
    wandb_logger = pl.loggers.WandbLogger(name=cfg.model.run_name, project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, log_model=False)

    trainer = pl.Trainer(
        accelerator=cfg.model.accelerator,
        devices=devices,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        logger=wandb_logger,
        strategy='ddp',
        log_every_n_steps=50,
        # test
        fast_dev_run=False
    )

    # Updating wandb configs
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg.dataset))
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg.model))
        wandb_logger.experiment.config["ckpt_path"] = ckpt_path
        # torch.cuda.memory._record_memory_history()

    torch.set_float32_matmul_precision('medium')

    # try:
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    # except Exception as error:
    #     print("An exception occurred:", error)
    # finally:
    #     if trainer.global_rank == 0:
    #         # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
    #         pass


if __name__ == '__main__':
    run()
