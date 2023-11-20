from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn

import medicalnet as mednet

class RetrievalResnet(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        model_params = {
            'sample_input_D': cfg.model.input_D,
            'sample_input_H': cfg.model.input_H,
            'sample_input_W': cfg.model.input_W,
            'num_seg_classes': 1, # not used
            'no_cuda': True if cfg.model.gpus == 0 else False
        }

        assert cfg.model.resnet_type in mednet.model_names, f"Resnet type {cfg.model.resnet_type} not found in {mednet.__all__}"

        self.resnet = mednet.resnet_gap(cfg.model.resnet_type, pretrain_path=
                                        cfg.model.pretrain_path, **model_params)
        self.criterion = nn.TripletMarginLoss()


    def forward(self, batch) -> Any:
        x1, x2, x3 = batch['anchor'], batch['positive'], batch['negative']

        x1, x2, x3 = self.resnet(x1), self.resnet(x2), self.resnet(x3)

        loss = self.criterion(x1, x2, x3)

        return loss



    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)

        self.log('train/triplet_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)

        # Logging to be understood
        self.log('val/triplet_loss', loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        lr = self.cfg.model.lr
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        return optimizer