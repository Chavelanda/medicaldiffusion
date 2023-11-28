import os
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from train.get_dataset import get_dataset

from filtering.retrieval import RetrievalResnet

@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    print("Starting run...")
    pl.seed_everything(cfg.model.seed)

    train_dataset, val_dataset, sampler = get_dataset(cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,
                                  num_workers=cfg.model.num_workers, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
                                shuffle=False, num_workers=cfg.model.num_workers)

    with open_dict(cfg):
        cfg.model.default_root_dir = os.path.join(cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)
        print("Setting default_root_dir to {}".format(cfg.model.default_root_dir))
    
    model = RetrievalResnet(cfg, input_d=train_dataset.d, input_h=train_dataset.h, input_w=train_dataset.w)

    # Defining callbacks for checkpointing
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    
    # load the most recent checkpoint file
    base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    if cfg.model.resume and os.path.exists(base_dir):
        print('Will resume from the recent ckpt')
        log_folder = ckpt_file = ''
        version_id_used = step_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            for fn in os.listdir(ckpt_folder):
                if fn == 'latest_checkpoint.ckpt':
                    ckpt_file = 'latest_checkpoint_prev.ckpt'
                    os.rename(os.path.join(ckpt_folder, fn),
                              os.path.join(ckpt_folder, ckpt_file))
            if len(ckpt_file) > 0:
                cfg.model.resume_from_checkpoint = os.path.join(
                    ckpt_folder, ckpt_file)
                print('will start from the recent ckpt %s' %
                      cfg.model.resume_from_checkpoint)
                
    # create wandb logger
    wandb_logger = pl.loggers.WandbLogger(name=cfg.model.run_name, project=cfg.model.wandb_project, entity=cfg.model.wandb_entity, log_model="all")
    
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg.dataset))
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg.model))

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

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=None)


if __name__ == '__main__':
    run()