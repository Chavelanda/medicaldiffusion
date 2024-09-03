from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from vq_gan_3d.model.vqgan import Encoder

import matplotlib

#  def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='replicate', num_groups=32):

class SS(pl.LightningModule):

    def __init__(self, n_hiddens, downsample, image_channels=3, norm_type='group', padding_type='replicate', num_groups=32, lr=1e-3, temperature=0.1):
        super().__init__()

        self.encoder = Encoder(n_hiddens, downsample, 
        image_channel=image_channels, norm_type=norm_type, 
        padding_type=padding_type, num_groups=num_groups)

        self.lr = lr
        self.t = temperature
        
        # save hyper-parameters to self.hparams (auto-logged by W&B)
        self.save_hyperparameters()

    # z is a batch of embeddings with shape (B, I, D).
    # (B, 0, D) and (B, 1, D) form positive pairs 
    # Also (B, 1, D) and (B, 0, D) form positive pairs
    def nt_xent_loss(self, z):
        B, I, D = z.shape
        
        # Compute similarity between positive pairs
        numerator = torch.repeat_interleave(torch.exp(F.cosine_similarity(z[:,0], z[:,1])/self.t), 2)
        
        # Compute similarity between all possible pairs
        similarity_matrix = torch.exp(F.cosine_similarity(z.view(B*I, D).unsqueeze(1), z.view(B*I, D).unsqueeze(0), dim=2)/self.t)
        
        # Mask similarity of positive pairs
        mask = torch.eye(B*I, dtype=torch.bool, device=self.device)
        zeros = torch.zeros_like(similarity_matrix)
        similarity_matrix = torch.where(mask, zeros, similarity_matrix)

        # For each example compute sum of negative pairs
        denominator = torch.sum(similarity_matrix, dim=1)
        
        loss = -torch.log(numerator / denominator)

        loss = torch.mean(loss)
        
        return loss

    def nt_xent_loss2(self, z):
        B, I, D = z.shape
        z = z.view(B*I, D)

        similarity_matrix = F.cosine_similarity(z[None,:,:], z[:,None,:], dim=-1)

        eye = torch.eye(B*I).bool()

        y = similarity_matrix

        y[eye] = float("-inf")

        # This forms a ground truth like this tensor([1, 0, 3, 2, 5, 4, 7, 6]). 
        # This because we provided the following positive pairs (0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)
        target = torch.arange(B*I, device=self.device)
        target[0::2] += 1
        target[1::2] -= 1

        loss = F.cross_entropy(y / self.t, target, reduction="mean")
        
        return loss

    def forward(self, batch):
        x = batch['data']

        B, I, C, D, H, W = x.shape

        # View as normal batched input
        x = x.view(B*I, C, D, H, W)
        
        # Encode data
        x = self.encoder(x)
        
        # Reform positive groups
        x = x.view(B, I, -1)
        
        # loss = self.nt_xent_loss(x)
        loss = self.nt_xent_loss2(x)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        
        self.log('train/loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)

    def test_step(self, batch):
        x = batch['data']
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


from dataset.mrnet import MRNetDatasetSS
from torch.utils.data import DataLoader

if __name__=="__main__":
    ss = SS(16, [4,4,4], image_channels=1, num_groups=16, temperature=10)

    ds = MRNetDatasetSS(root_dir='data/mrnet', recon_root_dir='data/mrnet-vqgan-01')
    dl = DataLoader(dataset=ds, batch_size=10, shuffle=False)
    img = torch.rand(3, 1, 1, 128, 128, 128)
    img = torch.cat((img, img + (0.5**0.8)*torch.randn(3, 1, 1, 128, 128, 128)), dim=1)
    print(img.shape)
    for batch in dl:
        print(batch['data'].shape)
        print(ss.forward(batch))
        break
    # img = torch.rand(3, 2, 1, 64, 64, 64)

    

