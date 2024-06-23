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
        self.save_hyperparameters()
        

    def encode(self, x, include_embeddings=False, quantize=True):
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
        return self.decoder(h)

    def forward(self, x, optimizer_idx=None, log_image=False, name='train', test=False):
        B, C, T, H, W = x.shape

        losses = {}

        # print('Encoder')
        z = self.pre_vq_conv(self.encoder(x))
        # print('Codebook')
        vq_output = self.codebook(z)
        # print('Decoder')
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))

        if test:
            return x_recon

        # print('Losses')
        losses[f'{name}/perplexity'] = vq_output['perplexity']
        losses[f'{name}/commitment_loss'] = vq_output['commitment_loss']
        losses[f'{name}/recon_loss'] = F.l1_loss(x_recon, x) * self.l1_weight

        # print('Selecting random frames')
        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, T, [B]).to(self.device)
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if log_image:
            # print('Logging images')
            self.logger.experiment.log({'samples': [wandb.Image(frames[0].cpu(), caption='original'), wandb.Image(frames_recon[0].cpu().to(torch.float32), caption='recon')], 'trainer/global_step': self.global_step})

        # print('adopt weight')
        disc_factor = adopt_weight(self.global_step, threshold=self.cfg.model.discriminator_iter_start)
        # Autoencoder - train the "generator"
        if optimizer_idx == 0:
            # Perceptual loss
            losses[f'{name}/perceptual_loss'] = self.perceptual_loss(frames, frames_recon)

            # Discriminator loss (turned on after a certain epoch)
            if disc_factor > 0:
                pred_image_fake, pred_video_fake, losses[f'{name}/g_image_loss'], losses[f'{name}/g_video_loss'], losses[f'{name}/aeloss'] = self.dg_loss(x_recon, frames_recon, disc_factor)

                # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
                losses[f'{name}/image_gan_feat_loss'], losses[f'{name}/video_gan_feat_loss'], losses[f'{name}/gan_feat_loss'] = self.gan_feat_loss(x, frames, pred_image_fake, pred_video_fake, disc_factor)
            else:
                losses[f'{name}/g_image_loss'], losses[f'{name}/g_video_loss'], losses[f'{name}/aeloss'], losses[f'{name}/image_gan_feat_loss'], losses[f'{name}/video_gan_feat_loss'], losses[f'{name}/gan_feat_loss'] = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)

        # Train discriminator
        elif optimizer_idx == 1:
            # Discriminator loss
            #losses[f'{name}/logits_image_real'], losses[f'{name}/logits_video_real'], losses[f'{name}/logits_image_fake'], losses[f'{name}/logits_video_fake'], losses[f'{name}/d_image_loss'], losses[f'{name}/d_video_loss'], losses[f'{name}/discloss'] = self.dd_loss(x, x_recon, frames, frames_recon, disc_factor)
            _, _, _, _, losses[f'{name}/d_image_loss'], losses[f'{name}/d_video_loss'], losses[f'{name}/discloss'] = self.dd_loss(x, x_recon, frames, frames_recon, disc_factor)
            
        # Validation
        else:
            # print('Perceptual loss')
            losses[f'{name}/perceptual_loss'] = self.perceptual_loss(frames, frames_recon)
        
        return x_recon, losses

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

    def dg_loss(self, x_recon, frames_recon, disc_factor):
        logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon)
        logits_video_fake, pred_video_fake = self.video_discriminator(x_recon)
        
        g_image_loss = -torch.mean(logits_image_fake).to(self.device)
        g_video_loss = -torch.mean(logits_video_fake).to(self.device)
        g_loss = self.image_gan_weight*g_image_loss + self.video_gan_weight*g_video_loss
        
        aeloss = disc_factor * g_loss

        return pred_image_fake, pred_video_fake, g_image_loss, g_video_loss, aeloss

    def perceptual_loss(self, frames, frames_recon):
        perceptual_loss = torch.zeros(1).to(self.device)
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight       
        
        return perceptual_loss

    def training_step(self, batch, batch_idx):
        opt_ae, opt_disc = self.optimizers()
        
        x = batch['data']
        
        train_gen = self.global_step % 2 == 0 

        # Train generator
        if train_gen or self.global_step < self.cfg.model.discriminator_iter_start:
            _, losses = self.forward(x, optimizer_idx=0, name='train_g')
            
            loss_ae = losses['train_g/recon_loss'] + losses['train_g/commitment_loss'] + losses['train_g/aeloss'] + losses['train_g/perceptual_loss'] + losses['train_g/gan_feat_loss']
            
            opt_ae.zero_grad()
            self.manual_backward(loss_ae)
            self.clip_gradients(opt_ae, self.gradient_clip_val)
            opt_ae.step()
            
            #wandb.log(losses)
            # del losses['trainer/global_step']
            self.log_dict(losses, prog_bar=True, on_step=True, on_epoch=True)
        
        # Train discriminator when needed
        elif not train_gen and self.global_step >= self.cfg.model.discriminator_iter_start:
            _, losses = self.forward(x, optimizer_idx=1, name='train_d')
            
            loss_disc = losses['train_d/discloss']
            
            opt_disc.zero_grad()
            self.manual_backward(loss_disc)
            self.clip_gradients(opt_disc, self.gradient_clip_val)
            opt_disc.step()
            
            #wandb.log(losses)
            # del losses['trainer/global_step']
            self.log_dict(losses, prog_bar=True, on_step=True, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        x = batch['data']  # TODO: batch['stft']
        if batch_idx == 0:
            _, losses = self.forward(x, name='val', log_image=True)
        else:
            _, losses = self.forward(x, name='val')

        self.log_dict(losses, prog_bar=True)
        # wandb.log(losses)

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_recon = self.forward(x, test=True)
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
        return opt_ae, opt_disc
    
    # def log_optim_0(self, vq_output, recon_loss, perceptual_loss, g_image_loss, g_video_loss, aeloss, image_gan_feat_loss, video_gan_feat_loss):
    #     logs = {'train/g_image_loss': g_image_loss, 'train/g_video_loss': g_video_loss, 
    #             'train/image_gan_feat_loss': image_gan_feat_loss, 
    #             'train/video_gan_feat_loss': video_gan_feat_loss, 
    #             'train/perceptual_loss': perceptual_loss, 'train/recon_loss': recon_loss, 
    #             'train/aeloss': aeloss, 'train/commitment_loss': vq_output['commitment_loss'], 
    #             'train/perplexity': vq_output['perplexity'], 'trainer/global_step': self.global_step}
    #     self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)
    #     wandb.log(logs)
        
        
    # def log_optim_1(self, logits_image_fake, logits_video_fake, logits_image_real, logits_video_real, d_image_loss, d_video_loss, discloss):
    #     logs = {'train/logits_image_real': logits_image_real.mean().detach(), 
    #             'train/logits_image_fake': logits_image_fake.mean().detach(), 
    #             'train/logits_video_real': logits_video_real.mean().detach(), 
    #             'train/logits_video_fake': logits_video_fake.mean().detach(), 
    #             'train/d_image_loss': d_image_loss, 'train/d_video_loss': d_video_loss, 
    #             'train/discloss': discloss, 'trainer/global_step': self.global_step}
    #     self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)
    #     wandb.log(logs)

    # def log_images(self, batch, **kwargs):
    #     log = dict()
    #     x = batch['data']
    #     x = x.to(self.device)
    #     frames, frames_rec, _, _ = self(x, log_image=True)
    #     log["inputs"] = frames
    #     log["reconstructions"] = frames_rec
    #     #log['mean_org'] = batch['mean_org']
    #     #log['std_org'] = batch['std_org']
    #     return log

    # def log_videos(self, batch, **kwargs):
    #     log = dict()
    #     x = batch['data']
    #     _, _, x, x_rec = self(x, log_image=True)
    #     log["inputs"] = x
    #     log["reconstructions"] = x_rec
    #     #log['mean_org'] = batch['mean_org']
    #     #log['std_org'] = batch['std_org']
    #     return log


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
            return self.model(input), _


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
            return self.model(input), _
