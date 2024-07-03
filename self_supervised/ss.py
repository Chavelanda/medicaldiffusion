from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from vq_gan_3d.model.vqgan import Encoder

#  def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='replicate', num_groups=32):

class SS(pl.LightningModule):

    def __init__(self, n_hiddens, downsample, image_channels=3, norm_type='group', padding_type='replicate', num_groups=32, lr=1e-3):
        super().__init__()

        self.encoder = Encoder(n_hiddens, downsample, 
        image_channel=image_channels, norm_type=norm_type, 
        padding_type=padding_type, num_groups=num_groups)

        self.lr = lr
        
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    # z is a batch of embeddings with shape (B, I, D).
    # (B, 0, D) and (B, 1, D) form positive pairs 
    # Also (B, 1, D) and (B, 0, D) form positive pairs
    def nt_xent_loss(self, z, t=0.1):
        B, I, D = z.shape
        
        # Compute similarity between positive pairs
        numerator = torch.repeat_interleave(torch.exp(F.cosine_similarity(z[:,0], z[:,1])/t), 2)
        
        # Compute similarity between all possible pairs
        similarity_matrix = torch.exp(F.cosine_similarity(z.view(B*I, D).unsqueeze(1), z.view(B*I, D).unsqueeze(0), dim=2)/t)
        
        # Mask similarity of positive pairs
        # mask = self.indicator_function(z.shape)
        mask = torch.eye(B*I, dtype=torch.bool, device=self.device)
        zeros = torch.zeros_like(similarity_matrix)
        similarity_matrix = torch.where(mask, zeros, similarity_matrix)

        # For each example compute sum of negative pairs
        denominator = torch.sum(similarity_matrix, dim=1)
        
        loss = -torch.log(numerator / denominator)

        loss = torch.mean(loss)
        
        return loss

    def indicator_function(self, z_shape):
        B, I, D = z_shape

        mask = torch.repeat_interleave(torch.repeat_interleave(torch.eye(B, dtype=torch.bool, device=self.device), I, dim=0), I, dim=1)

        return mask

    def forward(self, batch):
        x = batch['data']

        B, I, C, D, H, W = x.shape

        # View as normal batched input
        x = x.view(B*I, C, D, H, W)
        
        # Encode data
        x = self.encoder(x)
        
        # Reform positive groups
        x = x.view(B, I, -1)
        
        loss = self.nt_xent_loss(x)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        
        wandb.log({'train/loss': loss, 'trainer/global_step': self.global_step})
        self.log('train/loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)

        wandb.log({'val/loss': loss, 'trainer/global_step': self.global_step})
        self.log('val/loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

if __name__=="__main__":
    ss = SS(16, [4,4,4], image_channels=1, num_groups=16)
    img = torch.rand(3, 1, 1, 64, 64, 64)
    img = torch.cat((img, img + (0.8**0.5)*torch.randn(3, 1, 1, 64, 64, 64)), dim=1)
    # img = torch.rand(3, 2, 1, 64, 64, 64)
    
    print(ss.forward({'data': img}))
    

