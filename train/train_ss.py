import os
import shutil
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import DataLoader
from dataset.get_dataset import get_dataset

from self_supervised.ss import SS

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    train_dataset, val_dataset, sampler = get_dataset(cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,
                                  num_workers=cfg.model.num_workers, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
                                shuffle=False, num_workers=cfg.model.num_workers)

    with open_dict(cfg):
        cfg.model.default_root_dir = os.path.join(cfg.model.default_root_dir, cfg.dataset.name, cfg.model.run_name)
        print("Setting default_root_dir to {}".format(cfg.model.default_root_dir))
        base_dir = cfg.model.default_root_dir
    
    model = SS(cfg.model.n_hiddens, cfg.model.downsample, image_channels=cfg.dataset.image_channels, temperature=cfg.model.temperature,
    norm_type=cfg.model.norm_type, padding_type=cfg.model.padding_type, num_groups=cfg.model.num_groups, lr=cfg.model.lr)

    # Defining callbacks for checkpointing and early stopping
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/loss',
                     save_top_k=1, mode='min', dirpath=base_dir, filename='best_val-{epoch}-{step}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=5000, save_top_k=-1,
                     dirpath=base_dir, filename='train-{epoch}-{step}'))
    callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=15))
    

    # load the most recent checkpoint file
    ckpt_path = None

    if cfg.model.resume and os.path.exists(base_dir):
        print('Will resume from the recent ckpt')

        # Copy and rename the latest checkpoint file
        if cfg.model.checkpoint_path:
            ckpt_path = cfg.model.checkpoint_path
            print(f'Will resume from the recent ckpt {ckpt_path}')
        elif 'latest_checkpoint.ckpt' in os.listdir(base_dir):
            src_file = os.path.join(base_dir, 'latest_checkpoint.ckpt')
            ckpt_file = 'latest_checkpoint_prev.ckpt'
            ckpt_path = os.path.join(base_dir, ckpt_file)
            shutil.copy(src_file, ckpt_path)
            print(f'Will resume from the recent ckpt {ckpt_path}')
        else:
            print('No latest_checkpoint.ckpt found in {}.'.format(base_dir))
            return None
                
    # create wandb logger
    wandb_logger = pl.loggers.WandbLogger(name=cfg.model.run_name, project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, log_model="all")
    
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg.dataset))
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg.model))
    wandb_logger.experiment.config["ckpt_path"] = ckpt_path

    trainer = pl.Trainer(
        accelerator=cfg.model.accelerator,
        devices=cfg.model.devices,
        default_root_dir=cfg.model.default_root_dir,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        logger=wandb_logger,
    )

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    run()