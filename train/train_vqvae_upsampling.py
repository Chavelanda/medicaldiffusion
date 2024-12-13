"Adapted from https://github.com/SongweiGe/TATS"

import os

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
import torch
from torch.utils.data import DataLoader

from vq_gan_3d.model.vqvae_upsampling import VQVAE_Upsampling
from dataset.get_dataset import get_dataset


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    # torch.cuda.memory._record_memory_history()
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

    model = VQVAE_Upsampling(cfg, original_d=train_dataset.original_d, original_h=train_dataset.original_h, original_w=train_dataset.original_w, architecture=cfg.model.architecture)

    # model checkpointing callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/loss_ae',
                     save_top_k=1, mode='min', dirpath=base_dir, filename='best_val-{epoch}-{step}'))
    callbacks.append(ModelCheckpoint(every_n_epochs=30, save_top_k=-1,
                     dirpath=base_dir, filename='train-{epoch}-{step}'))
    # progress bar callback
    callbacks.append(TQDMProgressBar(refresh_rate=50))
    # log lr callback
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    # load the most recent checkpoint file
    ckpt_path = None
    if cfg.model.resume and os.path.isfile(cfg.model.checkpoint_path):
        assert os.path.isfile(cfg.model.checkpoint_path), f'Resume is {cfg.model.resume} but {cfg.model.checkpoint_path} is not a checkpoint file!'
        model = VQVAE_Upsampling.load_from_checkpoint(cfg.model.checkpoint_path)
        ckpt_path = cfg.model.checkpoint_path
        print(f'Will resume from the recent ckpt {ckpt_path}')
        
    # create wandb logger
    wandb_logger = pl.loggers.WandbLogger(name=cfg.model.run_name, project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, log_model=False)

    trainer = pl.Trainer(
        accelerator=cfg.model.accelerator,
        devices=cfg.model.devices,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        logger=wandb_logger,
        strategy='ddp_find_unused_parameters_true',
        log_every_n_steps=50,
    )

    # Updating wandb configs
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg.dataset))
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg.model))
        wandb_logger.experiment.config["ckpt_path"] = ckpt_path

    torch.set_float32_matmul_precision('medium')

    try:
        trainer.fit(model, train_dataloader, val_dataloader)
    except Exception as error:
        print("An exception occurred:", error)
    finally:
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        pass
        

if __name__ == '__main__':
    run()
