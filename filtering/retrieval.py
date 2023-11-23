from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


import medicalnet as mednet

class RetrievalResnet(pl.LightningModule):
    def __init__(self, cfg, verbose=False) -> None:
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
        self.margin = 1.0

        self.verbose = verbose

    def triplet_margin_loss_online(self, x):
        B, I, C = x.shape
        element_range = torch.arange(I)
        batch_range = torch.arange(B)

        # Get all combinations of positive_range for anchor and positive examples
        element_combinations = torch.combinations(element_range, r=2)
        
        # Initialize a list to store the triplets_indices
        triplets_indices = []

        # Iterate over the batch size for positive examples
        for b_positive in batch_range:
            # Iterate over the combinations
            for element1, element2 in element_combinations:  
                # Iterate over the batch size for negative examples
                for b_negative in batch_range:
                    # Ensure the negative example is from a different batch
                    if b_negative != b_positive:
                        # Iterate over the negative indices
                        for negative in element_range:
                            # Add the triplet to the list
                            triplets_indices.append(((b_positive, element1), (b_positive, element2), (b_negative, negative)))

        # Convert the list of triplets_indices to a tensor
        triplets_indices = torch.tensor(triplets_indices) 
        triplets_indices = triplets_indices.view(-1, 2) 
        
        # Index x with triplets_indices
        triplets = x[triplets_indices[:, 0], triplets_indices[:, 1]]

        # Reshape triplets for triplet loss calculation
        triplets = triplets.view(-1, 3, C)
        
        # Mine hard negatives
        positive_anchor_distances = F.pairwise_distance(triplets[:, 1], triplets[:, 0]) # torch.cdist(triplets[:, 1].unsqueeze(1), triplets[:, 0].unsqueeze(1)).squeeze(1) # torch.norm(triplets[:, 0] - triplets[:, 1], dim=1)
        negative_anchor_distances = F.pairwise_distance(triplets[:, 2], triplets[:, 0]) # torch.cdist(triplets[:, 2].unsqueeze(1), triplets[:, 0].unsqueeze(1)).squeeze(1) # torch.norm(triplets[:, 0] - triplets[:, 2], dim=1)
        positive_anchor_margin_distances = positive_anchor_distances + self.margin

        semi_hard_negative_mask = (positive_anchor_distances < negative_anchor_distances) & (negative_anchor_distances < positive_anchor_margin_distances)
        
        masked_positive_anchor_distances = positive_anchor_distances[semi_hard_negative_mask]
        masked_negative_anchor_distances = negative_anchor_distances[semi_hard_negative_mask]

        # Calculate triplet margin loss
        loss = torch.clamp_min(self.margin + masked_positive_anchor_distances - masked_negative_anchor_distances, 0)
        return torch.mean(loss)
    
    def forward(self, batch) -> Any:
        x = batch['data']

        B, I, C, D, H, W = x.shape
        
        x_batched = x.view(B*I, C, D, H, W)
        
        x_batched = self.resnet(x_batched)

        x = x_batched.view(B, I, -1)

        loss = self.triplet_margin_loss_online(x)
    
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