from omegaconf import OmegaConf
from torchsummary import summary
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vq_gan_3d.model.vqgan import VQGAN, pad_to_multiple, SamePadConvTranspose3d, SamePadConv3d


class VQVAEUpsampling(VQGAN):
    # The idea is that in each setup the decoder is modified in a different way.
    # To modify the decoder I substitute the last convolution eith a sequential layer 
    # In all cases the principle guiding the new decoder is the aim of reaching a resolution of the reconstructed image
    # that is equal to the original size provided as attribute of the init method.

    def __init__(self, *args, original_d, original_h, original_w, architecture='base', architecture_down='base', **kwargs):
        super().__init__(*args, **kwargs)
        self.size = (original_d, original_h, original_w)
        self.architecture_down = architecture_down
        self.architecture = architecture

        print(f'\nSetting up with decoder architecture {architecture} and encoder architecture {architecture_down}\n')

        setup_dict = {
            'up': self.setup_up,
            'up_conv': self.setup_up_conv,
            'up_res_conv': self.setup_up_res_conv,
            'super': self.setup_super,
            'base': lambda *args, **kwargs: None,
            'down_furbo': self.setup_down_furbo
        }
        
        setup_dict[self.architecture_down]()
        setup_dict[self.architecture]()

        self.save_hyperparameters()

    def setup_up(self):
        # As last layer I do a deterministic trilinear upsampling
        conv = SamePadConv3d(self.decoder.conv_last.conv.in_channels, self.image_channels, kernel_size=3)
        upsampling = nn.Upsample(size=self.size, mode='trilinear')

        self.decoder.conv_last = nn.Sequential(conv, upsampling)

    def setup_up_conv(self):
        # As last layer I do a deterministic trilinear upsampling
        # Moreover, I add a final conv layer
        conv1 = SamePadConv3d(self.decoder.conv_last.conv.in_channels, self.image_channels, kernel_size=3)
        upsampling = nn.Upsample(size=self.size, mode='trilinear')
        conv2 = SamePadConv3d(self.image_channels, self.image_channels, kernel_size=3)

        self.decoder.conv_last = nn.Sequential(conv1, upsampling, conv2)

    def setup_up_res_conv(self):
        # As last layer I do a deterministic trilinear upsampling
        # Moreover, I add a final residual conv layer
        conv1 = SamePadConv3d(self.decoder.conv_last.conv.in_channels, self.image_channels, kernel_size=3)
        upsampling = nn.Upsample(size=self.size, mode='trilinear')
        conv2 = ResidualSamePadConv3d(self.image_channels, self.image_channels, kernel_size=3)

        self.decoder.conv_last = nn.Sequential(conv1, upsampling, conv2)

    def setup_super(self):
        # In this case, instead of a trilinear upsampling, I apply a learnable transposed convolution.
        # The transposed convolution is initialized as a trilinear upsampling (it makes sense since we are in the image space)
        # After that I still apply a trilinear upsampling layer to fill the resolution gap to the original size 
        conv = SamePadConv3d(self.decoder.conv_last.conv.in_channels, self.image_channels, kernel_size=3)
        kernel_size=4
        up1 = SamePadConvTranspose3d(1, 1, kernel_size=kernel_size, stride=2)
        self.init_trilinear_kernel(kernel_size, up1.convt)
        up2 = nn.Upsample(size=self.size, mode='trilinear')

        self.decoder.conv_last = nn.Sequential(conv, up1, up2)

    def setup_down_furbo(self):
        conv1 = SamePadConv3d(self.image_channels, 8, kernel_size=3, stride=2, padding_type=self.padding_type)
        conv2 = SamePadConv3d(8, self.n_hiddens, kernel_size=3, padding_type=self.padding_type)

        self.encoder.conv_first = nn.Sequential(conv1, conv2)

    def init_trilinear_kernel(self, kernel_size, conv):
        # Generate 1D kernel
        center_idx = (kernel_size - 1) / 2
        linear_kernel = np.array([1 - abs(i - center_idx) / center_idx for i in range(kernel_size)])
        
        # Create the outer product to form the 3D kernel
        kernel = np.outer(linear_kernel, linear_kernel).reshape(kernel_size, kernel_size, 1) * linear_kernel
        # Normalize to ensure weights sum to 1
        kernel /= kernel.sum()

        # Convert to PyTorch tensor and set weights
        for i in range(conv.out_channels):
            for j in range(conv.in_channels):
                conv.weight.data[i, j] = torch.tensor(kernel, dtype=conv.weight.dtype)

        # Set bias to zero
        if conv.bias is not None:
            conv.bias.data.zero_()


    def forward(self, x, x_original=None, name='train'):
        # Pad image so that it is divisible by downsampling scale
        x, _ = pad_to_multiple(x, self.downsample)

        B, C, T, H, W = x.shape

        losses = {}

        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))

        if name=='test': return x_recon

        # VQ-VAE losses
        losses[f'{name}/perplexity'] = vq_output['perplexity']
        losses[f'{name}/commitment_loss'] = vq_output['commitment_loss']

        # Key modification to the model
        losses[f'{name}/recon_loss'] = F.l1_loss(x_recon, x_original) * self.l1_weight

        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, self.size[0], [B]).to(self.device)
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, self.size[1], self.size[2])
        frames = torch.gather(x_original, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)
        # Still VQ-VAE loss
        losses[f'{name}/perceptual_loss'] = self.perceptual_loss(frames, frames_recon)

        return x_recon, losses, (frames[0].detach().cpu(), frames_recon[0].detach().cpu())
    

    def training_step(self, batch, batch_idx):
        opt_ae = self.optimizers()
        
        x = batch['data']
        x_original = batch['data_original']

        
        _, losses, _ = self.forward(x, x_original, name='train')
        
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
        x_original = batch['data_original']

        _, losses, frames = self.forward(x, x_original, name='val')
        
        loss_ae = losses['val/recon_loss'] + losses['val/commitment_loss'] + losses['val/perceptual_loss']
        losses['val/loss_ae'] = loss_ae
        self.val_step_metric.append(loss_ae)

        # Log image
        if batch_idx == 0:
            self.logger.experiment.log({'samples': [wandb.Image(frames[0], caption='original'), wandb.Image(frames[1].to(torch.float32), caption='recon')], 'trainer/global_step': self.global_step})
        
        self.log_dict(losses, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x = batch['data']

        x_recon = self.forward(x, name='test')
        return x_recon

    def configure_optimizers(self):
        print('Setting up optimizers')
        lr = self.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.pre_vq_conv.parameters()) +
                                  list(self.post_vq_conv.parameters()) +
                                  list(self.codebook.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        
        # compute start factor to begin with base_lr
        start_factor = self.base_lr/lr
        ae_scheduler = {'scheduler': torch.optim.lr_scheduler.LinearLR(opt_ae, start_factor=start_factor, total_iters=5), 'name': 'warmup-ae'}
        plateau_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ae, 'min', patience=20), 'name': 'plateau-ae'}
        
        return [opt_ae], [ae_scheduler, plateau_scheduler] 


class ResidualSamePadConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = SamePadConv3d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return x + self.conv(x)



if __name__ == '__main__':
    print('start')
    # create cfg as I wish
    cfg_dict = {
        'model': {
            'lr': 1e-5,
            'base_lr': 1e-5,
            'downsample': [4,4,4],
            'embedding_dim': 8,
            'n_codes': 16384,
            'n_hiddens': 16,
            'norm_type': 'group',
            'padding_type': 'replicate',
            'num_groups': 16,
            'no_random_restart': False,
            'restart_thres': 1.0,
            'gan_feat_weight': 4.0,
            'disc_channels': 64,
            'disc_layers': 3,
            'disc_loss_type': 'hinge',
            'image_gan_weight': 1.0,
            'video_gan_weight': 1.0,
            'perceptual_weight': 4.0,
            'l1_weight': 4.0,
            'gradient_clip_val': 1.0,
            'discriminator_iter_start': 500000
        },
        'dataset': {
            'image_channels': 1,
            'd': 192,
            'h': 148,
            'w': 216,
        }
    }
    cfg = OmegaConf.create(cfg_dict)

    model = VQVAEUpsampling(cfg, 456, 352, 512, 'super')

    summary(model, [(1, 192, 148, 216), (1, 456, 352, 512)], device='cpu', depth=10)

    print('end')