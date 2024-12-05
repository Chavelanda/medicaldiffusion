import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from vq_gan_3d.model.vqgan import VQGAN, pad_to_multiple, SamePadConvTranspose3d


class VQVAE_Upsampling(VQGAN):
    def __init__(self, cfg, original_d, original_h, original_w):
        super().__init__(cfg)
        self.size = (original_d, original_h, original_w)
        self.upsampling = nn.Upsample(self.size, mode='trilinear')

    def decode(self, latent, quantize=False):
        decoded = super().decode(latent, quantize)
        decoded = self.upsampling(decoded)
        return decoded


    def forward(self, x, x_upsampled, name='train'):
        # Pad image so that it is divisible by downsampling scale
        x, _ = pad_to_multiple(x, self.cfg.model.downsample)

        B, C, T, H, W = x.shape

        losses = {}

        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))
        
        # Key modification to the model
        x_recon_upsampled = self.upsampling(x_recon)

        if name=='test': return x_recon_upsampled

        # VQ-VAE losses
        losses[f'{name}/perplexity'] = vq_output['perplexity']
        losses[f'{name}/commitment_loss'] = vq_output['commitment_loss']

        # Key modification to the model
        losses[f'{name}/recon_loss'] = F.l1_loss(x_recon_upsampled, x_upsampled) * self.l1_weight

        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, self.size[0], [B]).to(self.device)
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, self.size[1], self.size[2])
        frames = torch.gather(x_upsampled, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon_upsampled, 2, frame_idx_selected).squeeze(2)
        # Still VQ-VAE loss
        losses[f'{name}/perceptual_loss'] = self.perceptual_loss(frames, frames_recon)

        return x_recon_upsampled, losses, (frames[0].detach().cpu(), frames_recon[0].detach().cpu())
    

    def training_step(self, batch, batch_idx):
        opt_ae = self.optimizers()
        
        x = batch['data']
        x_upsampled = batch['data_original']

        
        _, losses, _ = self.forward(x, x_upsampled, name='train')
        
        # Losses VQ-VAE
        loss_ae = losses['train/recon_loss'] + losses['train/commitment_loss'] + losses['train/perceptual_loss']
        
        losses['train/loss_ae'] = loss_ae

        opt_ae.zero_grad()
        self.manual_backward(loss_ae)
        self.clip_gradients(opt_ae, self.gradient_clip_val)
        opt_ae.step()
            
        self.log_dict(losses, prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)

    def validation_step(self, batch, batch_idx):
        x = batch['data']
        x_upsampled = batch['data_original']

        _, losses, frames = self.forward(x, x_upsampled, name='val')
        
        loss_ae = losses['val/recon_loss'] + losses['val/commitment_loss'] + losses['val/perceptual_loss']
        losses['val/loss_ae'] = loss_ae
        self.val_step_metric.append(loss_ae)

        # Log image
        if batch_idx == 0:
            self.logger.experiment.log({'samples': [wandb.Image(frames[0], caption='original'), wandb.Image(frames[1].to(torch.float32), caption='recon')], 'trainer/global_step': self.global_step})
        
        self.log_dict(losses, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x = batch['data']
        x_upsampled = batch['data_original']

        x_recon = self.forward(x, x_upsampled, name='test')
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
        
        # compute start factor to begin with base_lr
        start_factor = self.cfg.model.base_lr/lr
        ae_scheduler = {'scheduler': torch.optim.lr_scheduler.LinearLR(opt_ae, start_factor=start_factor, total_iters=5), 'name': 'warmup-ae'}
        plateau_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ae, 'min', patience=20), 'name': 'plateau-ae'}
        
        return [opt_ae], [ae_scheduler, plateau_scheduler] 



class VQVAE_SuperResolution(VQVAE_Upsampling):
    

    def __init__(self, cfg, original_d, original_h, original_w):
        super().__init__(cfg, original_d, original_h, original_w)
        self.up1 = SamePadConvTranspose3d(1, 1, kernel_size=4, stride=2)
        self.up2 = nn.Upsample(self.size, mode='trilinear')

        self.upsampling = nn.Sequential(self.up1, self.up2)
