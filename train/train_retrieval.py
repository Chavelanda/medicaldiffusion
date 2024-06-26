import os
import shutil
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from train.get_dataset import get_dataset

from retrieval.retrieval import RetrievalResnet

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    train_dataset, val_dataset, sampler = get_dataset(cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,
                                  num_workers=cfg.model.num_workers, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
                                shuffle=False, num_workers=cfg.model.num_workers)

    with open_dict(cfg):
        cfg.model.default_root_dir = os.path.join(cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix, cfg.model.run_name)
        print("Setting default_root_dir to {}".format(cfg.model.default_root_dir))
        base_dir = cfg.model.default_root_dir
    
    model = RetrievalResnet(cfg, input_d=train_dataset.d, input_h=train_dataset.h, input_w=train_dataset.w)

    # Defining callbacks for checkpointing
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/triplet_loss',
                     save_top_k=3, mode='min', dirpath=base_dir, filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=100,
                     save_top_k=-1, dirpath=base_dir, filename='train-{epoch}-{step}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                     dirpath=base_dir, filename='train-{epoch}-{step}'))
    

    # load the most recent checkpoint file
    ckpt_path = None

    if cfg.model.resume and os.path.exists(base_dir):
        print('Will resume from the recent ckpt')

        # Copy and rename the latest checkpoint file
        if 'latest_checkpoint.ckpt' in os.listdir(base_dir):
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
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        gradient_clip_val=cfg.model.gradient_clip_val,
        logger=wandb_logger,
    )

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    run()