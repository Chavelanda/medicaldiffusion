"""Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import argparse
import numpy as np
import pickle as pkl

import wandb
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from vq_gan_3d.utils import shift_dim, adopt_weight, comp_getattr
from vq_gan_3d.model.lpips import LPIPS
from vq_gan_3d.model.codebook import Codebook


def silu(x):
    return x*torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def pad_to_multiple(x, divisors=(4, 4, 4)):
    """
    Pads a 3D input tensor along its last three spatial dimensions to make them divisible
    by the given divisors. Ensures symmetric padding where possible.
    
    Args:
        x (torch.Tensor): 3D input tensor of shape (..., D, H, W).
        divisors (tuple): A tuple of 3 integers specifying the divisors for the depth, height, and width.
    
    Returns:
        tuple: (padded_tensor, padding_sizes)
            - padded_tensor (torch.Tensor): Padded tensor.
            - padding_sizes (tuple): Tuple of 3 tuples representing the padding applied 
                                     for each dimension as (pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right).
    """
    d, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
    div_d, div_h, div_w = divisors

    # Compute padding for each dimension to make divisible
    pad_d = (div_d - d % div_d) % div_d
    pad_h = (div_h - h % div_h) % div_h
    pad_w = (div_w - w % div_w) % div_w

    # Distribute padding symmetrically, adding extra to the end if necessary
    pad_front, pad_back = pad_d // 2, pad_d - pad_d // 2
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

    # Apply padding using F.pad
    padded_x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
    
    # Return padded tensor and the padding sizes
    return padded_x, ((pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right))


def crop_to_original(x, padding_sizes):
    """
    Crops a padded 3D tensor back to its original size based on the padding_sizes.
    
    Args:
        x (torch.Tensor): The padded 3D tensor of shape (..., D, H, W).
        padding_sizes (tuple): A tuple of 3 tuples representing the padding applied
                               for each dimension as (pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right).
    
    Returns:
        torch.Tensor: Cropped tensor with the original size before padding.
    """
    (pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right) = padding_sizes
    
    # Compute the cropping indices
    d_start = pad_front
    d_end = -pad_back if pad_back > 0 else None
    
    h_start = pad_top
    h_end = -pad_bottom if pad_bottom > 0 else None
    
    w_start = pad_left
    w_end = -pad_right if pad_right > 0 else None
    
    # Crop the tensor using slicing
    cropped_x = x[..., d_start:d_end, h_start:h_end, w_start:w_end]
    
    return cropped_x


class VQGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = cfg.model.embedding_dim
        self.n_codes = cfg.model.n_codes

        self.encoder = Encoder(cfg.model.n_hiddens, cfg.model.downsample,
                               cfg.dataset.image_channels, cfg.model.norm_type, cfg.model.padding_type,
                               cfg.model.num_groups,)
        self.decoder = Decoder(
            cfg.model.n_hiddens, cfg.model.downsample, cfg.dataset.image_channels, cfg.model.norm_type, cfg.model.num_groups)
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(
            self.enc_out_ch, cfg.model.embedding_dim, 1, padding_type=cfg.model.padding_type)
        self.post_vq_conv = SamePadConv3d(
            cfg.model.embedding_dim, self.enc_out_ch, 1)

        self.codebook = Codebook(cfg.model.n_codes, cfg.model.embedding_dim,
                                 no_random_restart=cfg.model.no_random_restart, restart_thres=cfg.model.restart_thres)
        
        # init padding
        img_size = (cfg.dataset.d, cfg.dataset.h, cfg.dataset.w)
        _, self.padding_sizes = pad_to_multiple(torch.rand(img_size), self.cfg.model.downsample)

        self.gan_feat_weight = cfg.model.gan_feat_weight
        # TODO: Changed batchnorm from sync to normal
        self.image_discriminator = NLayerDiscriminator(
            cfg.dataset.image_channels, cfg.model.disc_channels, cfg.model.disc_layers, norm_layer=nn.BatchNorm2d)
        self.video_discriminator = NLayerDiscriminator3D(
            cfg.dataset.image_channels, cfg.model.disc_channels, cfg.model.disc_layers, norm_layer=nn.BatchNorm3d)

        if cfg.model.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif cfg.model.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = cfg.model.image_gan_weight
        self.video_gan_weight = cfg.model.video_gan_weight

        self.perceptual_weight = cfg.model.perceptual_weight

        self.l1_weight = cfg.model.l1_weight

        self.automatic_optimization = False
        self.gradient_clip_val = cfg.model.gradient_clip_val

        # For ReduceLROnPlateau scheduler
        self.val_step_metric = []

        self.save_hyperparameters()
        

    def encode(self, x, include_embeddings=False, quantize=True):
        x, _ = pad_to_multiple(x, self.cfg.model.downsample)

        h = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output['embeddings'], vq_output['encodings']
            else:
                return vq_output['encodings']
        return h

    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        h = self.decoder(h)
        h = crop_to_original(h, self.padding_sizes)
        return h

    def forward(self, x, optimizer_idx=None, name='train'):
        # Pad image so that it is divisible by downsampling scale
        x, padding_sizes = pad_to_multiple(x, self.cfg.model.downsample)

        B, C, T, H, W = x.shape

        losses = {}

        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))

        if name=='test': return crop_to_original(x_recon, padding_sizes)

        # VQ-VAE losses
        losses[f'{name}/perplexity'] = vq_output['perplexity']
        losses[f'{name}/commitment_loss'] = vq_output['commitment_loss']
        losses[f'{name}/recon_loss'] = F.l1_loss(x_recon, x) * self.l1_weight

        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, T, [B]).to(self.device)
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)
        # Still VQ-VAE loss
        if optimizer_idx != 1: losses[f'{name}/perceptual_loss'] = self.perceptual_loss(frames, frames_recon)

        # VQ-GAN losses
        disc_factor = adopt_weight(self.global_step, threshold=self.cfg.model.discriminator_iter_start)
        # Train the "generator" (autoencoder)
        if optimizer_idx == 0 and disc_factor > 0:            
            pred_image_fake, pred_video_fake, losses[f'{name}/g_image_loss'], losses[f'{name}/g_video_loss'], losses[f'{name}/g_loss'] = self.dg_loss(x_recon, frames_recon, disc_factor)
            # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
            losses[f'{name}/image_gan_feat_loss'], losses[f'{name}/video_gan_feat_loss'], losses[f'{name}/gan_feat_loss'] = self.gan_feat_loss(x, frames, pred_image_fake, pred_video_fake, disc_factor)    
        # Train discriminator
        elif optimizer_idx == 1:
            # Discriminator loss
            _, _, _, _, losses[f'{name}/d_image_loss'], losses[f'{name}/d_video_loss'], losses[f'{name}/discloss'] = self.dd_loss(x, x_recon, frames, frames_recon, disc_factor)
        
        return x_recon, losses, (frames[0].detach().cpu(), frames_recon[0].detach().cpu())

    def perceptual_loss(self, frames, frames_recon):
        perceptual_loss = torch.zeros(1).to(self.device)
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight       
        
        return perceptual_loss

    def dg_loss(self, x_recon, frames_recon, disc_factor):
        logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon)
        logits_video_fake, pred_video_fake = self.video_discriminator(x_recon)
        
        g_image_loss = -torch.mean(logits_image_fake).to(self.device)
        g_video_loss = -torch.mean(logits_video_fake).to(self.device)
        g_loss = self.image_gan_weight*g_image_loss + self.video_gan_weight*g_video_loss
        
        g_loss = disc_factor * g_loss

        return pred_image_fake, pred_video_fake, g_image_loss, g_video_loss, g_loss

    def gan_feat_loss(self, x, frames, pred_image_fake, pred_video_fake, disc_factor):
        image_gan_feat_loss = 0
        video_gan_feat_loss = 0
        feat_weights = 4.0 / (3 + 1)
        
        if self.image_gan_weight > 0:
            logits_image_real, pred_image_real = self.image_discriminator(frames)
            for i in range(len(pred_image_fake)-1):
                image_gan_feat_loss += feat_weights * \
                        F.l1_loss(pred_image_fake[i], pred_image_real[i].detach()) * (self.image_gan_weight > 0)
        
        if self.video_gan_weight > 0:
            logits_video_real, pred_video_real = self.video_discriminator(x)
            for i in range(len(pred_video_fake)-1):
                video_gan_feat_loss += feat_weights * \
                        F.l1_loss(pred_video_fake[i], pred_video_real[i].detach()) * (self.video_gan_weight > 0)
        
        gan_feat_loss = disc_factor * self.gan_feat_weight * \
                (image_gan_feat_loss + video_gan_feat_loss)
            
        return image_gan_feat_loss,video_gan_feat_loss,gan_feat_loss

    def dd_loss(self, x, x_recon, frames, frames_recon, disc_factor):
        logits_image_real, _ = self.image_discriminator(frames.detach())
        logits_video_real, _ = self.video_discriminator(x.detach())

        logits_image_fake, _ = self.image_discriminator(frames_recon.detach())
        logits_video_fake, _ = self.video_discriminator(x_recon.detach())

        d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
        d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
        
        discloss = disc_factor * \
                (self.image_gan_weight*d_image_loss +
                 self.video_gan_weight*d_video_loss)
             
        return logits_image_real.mean().detach(), logits_video_real.mean().detach(), logits_image_fake.mean().detach(), logits_video_fake.mean().detach(), d_image_loss, d_video_loss, discloss

    def training_step(self, batch, batch_idx):
        opt_ae, opt_disc = self.optimizers()
        
        x = batch['data']
        
        train_gen = self.global_step % 2 == 0 

        # Train autoencoder
        if train_gen or self.global_step < self.cfg.model.discriminator_iter_start:
            _, losses, _ = self.forward(x, optimizer_idx=0, name='train')
            
            # Losses VQ-VAE
            loss_ae = losses['train/recon_loss'] + losses['train/commitment_loss'] + losses['train/perceptual_loss']
            
            # Losses VQ-GAN
            if self.global_step >= self.cfg.model.discriminator_iter_start:
                loss_ae += losses['train/g_loss'] + losses['train/gan_feat_loss']

            losses['train/loss_ae'] = loss_ae

            opt_ae.zero_grad()
            self.manual_backward(loss_ae)
            self.clip_gradients(opt_ae, self.gradient_clip_val)
            opt_ae.step()
        
        # Train discriminator
        elif not train_gen and self.global_step >= self.cfg.model.discriminator_iter_start:
            _, losses, _ = self.forward(x, optimizer_idx=1, name='train_d')
            
            loss_disc = losses['train_d/discloss']
            
            opt_disc.zero_grad()
            self.manual_backward(loss_disc)
            self.clip_gradients(opt_disc, self.gradient_clip_val)
            opt_disc.step()
            
        self.log_dict(losses, prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)

    def on_train_epoch_end(self):
        warmup_scheduler = self.lr_schedulers()[0]
        warmup_scheduler.step()


    def validation_step(self, batch, batch_idx):
        x = batch['data'] 
        _, losses, frames = self.forward(x, name='val')
        
        loss_ae = losses['val/recon_loss'] + losses['val/commitment_loss'] + losses['val/perceptual_loss']
        losses['val/loss_ae'] = loss_ae
        self.val_step_metric.append(loss_ae)

        # Log image
        if batch_idx == 0:
            self.logger.experiment.log({'samples': [wandb.Image(frames[0], caption='original'), wandb.Image(frames[1].to(torch.float32), caption='recon')], 'trainer/global_step': self.global_step})
        
        self.log_dict(losses, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        metric = torch.mean(torch.tensor(self.val_step_metric))

        # plateau_scheduler = self.lr_schedulers()[1]
        # plateau_scheduler.step(metric)

        self.val_step_metric.clear()

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_recon = self.forward(x, name='test')
        return x_recon

    def configure_optimizers(self):
        print('Setting up optimizers')
        lr = self.cfg.model.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.pre_vq_conv.parameters()) +
                                  list(self.post_vq_conv.parameters()) +
                                  list(self.codebook.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(list(self.image_discriminator.parameters()) +
                                    list(self.video_discriminator.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        
        # compute start factor to begin with base_lr
        start_factor = self.cfg.model.base_lr/lr
        ae_scheduler = {'scheduler': torch.optim.lr_scheduler.LinearLR(opt_ae, start_factor=start_factor, total_iters=5), 'name': 'warmup-ae'}
        plateau_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ae, 'min', patience=20), 'name': 'plateau-ae'}
        
        return [opt_ae, opt_disc], [ae_scheduler, plateau_scheduler] 


def Normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        # TODO Changed num_groups from 32 to 8
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        self.conv_first = SamePadConv3d(
            image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * 2**i
            out_channels = n_hiddens * 2**(i+1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            block.down = SamePadConv3d(
                in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            block.res = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group', num_groups=32):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()

        in_channels = n_hiddens*2**max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type, num_groups=num_groups),
            SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i == 0 else n_hiddens*2**(max_us-i+1)
            out_channels = n_hiddens*2**(max_us-i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(
                in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            block.res2 = ResBlock(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv3d(
            out_channels, image_channel, kernel_size=3)

    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='replicate', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = SamePadConv3d(
            in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv2 = SamePadConv3d(
            out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:  # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), None


class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), None
