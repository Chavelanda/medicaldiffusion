import gc
from omegaconf import OmegaConf
from torchsummary import summary
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vq_gan_3d.model.vqgan import VQGAN, crop_to_original, pad_to_multiple, SamePadConvTranspose3d, SamePadConv3d, silu, SiLU, Normalize, ResBlock
from vq_gan_3d.utils import adopt_weight, shift_dim


class VQVAEUpsampling(VQGAN):
    # The idea is that in each setup the decoder is modified in a different way.
    # To modify the decoder I substitute the last convolution eith a sequential layer 
    # In all cases the principle guiding the new decoder is the aim of reaching a resolution of the reconstructed image
    # that is equal to the original size provided as attribute of the init method.

    def __init__(self, *args, original_d, original_h, original_w, 
                 architecture='base', architecture_down='base', up_factor=1, upsampling_mode='trilinear', 
                 model_parallelism=False, simple_architecture=False, 
                 noise_prob=0, **kwargs):
        
        super().__init__(*args, simple_architecture=simple_architecture, **kwargs)

        self.size = (original_d, original_h, original_w)
        self.architecture_down = architecture_down
        self.architecture = architecture
        # The up factor is used to decide how much to upsample the image in the decoder as a factor of the original size. 0 to not upsample 
        self.up_factor = up_factor
        self.upsampling_mode = upsampling_mode

        self.model_parallelism = model_parallelism

        self.simple_architecture = simple_architecture

        self.noise_prob = noise_prob
        self.noisy_decoder = noise_prob > 0
        # The generator is instantiated in configure model and setup model parallelism
        
        print(f'\nSetting up with decoder architecture {architecture} and encoder architecture {architecture_down}\n')

        setup_dict = {
            'up': self.setup_up,
            'up_conv': self.setup_up_conv,
            'up_res_conv': self.setup_up_res_conv,
            'super': self.setup_super,
            'base': lambda *args, **kwargs: None,
            'down_furbo': self.setup_down_furbo
        }

        # Setup architecture        
        setup_dict[self.architecture_down]()
        setup_dict[self.architecture]()

        if self.simple_architecture:
            print('Setting up simple architecture')
            for block in self.decoder.conv_blocks:
                block.res2 = nn.Identity()
            pass

        if self.noisy_decoder:
            self.setup_noisy_decoder()
            self.variance_range = [0.05, 0.4]

        self.initialized = False

        self.save_hyperparameters()

    def setup_noisy_decoder(self):
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
            
        self.pre_vq_conv.eval()
        for p in self.pre_vq_conv.parameters():
            p.requires_grad = False
            
        self.codebook.eval()
        self.codebook.training = False
        for p in self.codebook.parameters():
            p.requires_grad = False


    def configure_model(self):
        if self.initialized:
            return
        else:
            self.initialized = True
            # Setup model parallelism 1
            if self.device.index is None:
                self.idx_0 = 'cpu'
                self.generator = torch.Generator()
            else:
                self.idx_0 = self.device.index
                self.generator = torch.Generator(device=self.device)
            
            self.idx_1 = self.idx_0
            self.idx_2 = self.idx_0
            self.idx_3 = self.idx_0
            self.idx_4 = self.idx_0
        if self.model_parallelism and not self.simple_architecture:
            self.idx_1 = self.idx_0 + 1
            self.codebook.to(self.idx_1)
            self.post_vq_conv.to(self.idx_1)
            self.decoder.final_block.to(self.idx_1)
            

    def set_model_parallelism(self):
        if self.model_parallelism:
            if self.simple_architecture:
                self.idx_1 = self.idx_0 + 1
                self.idx_2 = self.idx_1
                self.idx_3 = self.idx_1
                self.idx_4 = self.idx_0
            else:
                self.idx_1 = self.idx_0 + 1
                self.idx_2 = self.idx_0 + 2
                self.idx_3 = self.idx_0 + 1
                self.idx_4 = self.idx_0 + 3
                
        print(f'Indices set to: {self.idx_0} - {self.idx_1} - {self.idx_2} - {self.idx_3} - {self.idx_4}')

    def setup_up(self):
        # As last layer I do a deterministic trilinear upsampling
        conv = SamePadConv3d(self.decoder.conv_last.conv.in_channels, self.image_channels, kernel_size=3)

        if self.up_factor != 0:
            final_size = [s * self.up_factor for s in self.size]
            upsampling = nn.Upsample(size=final_size, mode=self.upsampling_mode)
        else:
            upsampling = nn.Identity()

        self.decoder.conv_last = nn.Sequential(conv, upsampling)

    def setup_up_conv(self):
        # As last layer I do a deterministic trilinear upsampling
        # Moreover, I add a final conv layer
        conv1 = SamePadConv3d(self.decoder.conv_last.conv.in_channels, self.image_channels, kernel_size=3)
        upsampling = nn.Upsample(size=self.size, mode=self.upsampling_mode)
        conv2 = SamePadConv3d(self.image_channels, self.image_channels, kernel_size=3)

        self.decoder.conv_last = nn.Sequential(conv1, upsampling, conv2)

    def setup_up_res_conv(self):
        # As last layer I do a deterministic trilinear upsampling
        # Moreover, I add a final residual conv layer
        conv1 = SamePadConv3d(self.decoder.conv_last.conv.in_channels, self.image_channels, kernel_size=3)
        upsampling = nn.Upsample(size=self.size, mode=self.upsampling_mode)
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
        up2 = nn.Upsample(size=self.size, mode=self.upsampling_mode)

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

    
    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output['encodings']
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        h = self.decoder(h)
        if self.architecture == 'base':
            h = crop_to_original(h, self.padding_sizes)
        return h


    def on_fit_start(self):
        self.set_model_parallelism()

        self.generator = torch.Generator(device=f'cuda:{self.idx_1}')
        self.codebook.to(self.idx_1)
        self.post_vq_conv.to(self.idx_1)
        
        self.decoder.final_block.to(self.idx_1)
        block0 = self.decoder.conv_blocks[0]
        block0.to(self.idx_2)
        block1 = self.decoder.conv_blocks[1]
        block1.up.to(self.idx_2)
        block1.res1.norm1.to(self.idx_2)
        block1.res1.conv1.to(self.idx_3)
        block1.res1.norm2.to(self.idx_3)

        
        block1.res1.conv2.to(self.idx_3)

        block1.res2.to(self.idx_4)
        for i, block in enumerate(self.decoder.conv_blocks):
            if i not in (0,1):
                block.to(self.idx_0)
        self.decoder.conv_last.to(self.idx_0)


    def forward(self, x, x_original=None, name='train', batch_idx=0, dataloader_idx=0):
        if not self.initialized:
            self.configure_model()
        # Pad image so that it is divisible by downsampling scale
        x, _ = pad_to_multiple(x, self.downsample)

        B, C, T, H, W = x.shape

        losses = {}

        x = self.pre_vq_conv(self.encoder(x)).to(self.idx_1)

        if self.noisy_decoder and name == 'train' and torch.rand(1) < self.noise_prob:
            if batch_idx == 0:
                gc.collect()
                torch.cuda.empty_cache()
            var = torch.rand(B, device=x.device) * (self.variance_range[1]- self.variance_range[0]) + self.variance_range[0]
            noise = self.denormalize_noise(torch.randn_like(x))
            x = x + noise * var.reshape(-1, 1, 1, 1, 1)
        elif self.noisy_decoder and name == 'val' and dataloader_idx > 0:
            if batch_idx == 0:
                gc.collect()
                torch.cuda.empty_cache()
            self.generator.manual_seed(batch_idx)
            var = torch.rand((B,), device=x.device, generator=self.generator) * (self.variance_range[1]- self.variance_range[0]) + self.variance_range[0]
            noise = torch.randn(x.size(), generator=self.generator, dtype=x.dtype, layout=x.layout, device=x.device)
            noise = self.denormalize_noise(noise)
            x = x + noise * var.reshape(-1, 1, 1, 1, 1)

        vq_output = self.codebook(x)
        x = vq_output['embeddings']
        x = self.post_vq_conv(x)

        # Decoder is decomposed for model parallelism
        n_blocks = len(self.decoder.conv_blocks)
        
        x_recon = self.decoder.final_block(x).to(self.idx_2)
        
        x_recon = self.decoder.forward_block(0, x_recon)
        
        block = self.decoder.conv_blocks[1]
        x_recon = block.up(x_recon)
        # I have to decompose also res1 to reach best parallelism
        h = x_recon
        h = block.res1.norm1(h)
        h = silu(h).to(self.idx_3)
        h = block.res1.conv1(h)
        h = block.res1.norm2(h)
        if not self.simple_architecture:
            h = silu(h)
        h = block.res1.conv2(h)

        if block.res1.in_channels != block.res1.out_channels:
            x_recon = block.res1.conv_shortcut(x_recon)

        x_recon = x_recon.to(self.idx_3)
        x_recon =  (x_recon+h).to(self.idx_4)
        # end res 1
        
        x_recon = block.res2(x_recon).to(self.idx_0)
        
        for i in range(n_blocks - 2):
            i = i + 2
            x_recon = self.decoder.forward_block(i, x_recon)
        
        x_recon = self.decoder.conv_last(x_recon)

        # For reconstruction inference there is no need to compute the loss
        if name=='test': 
            if self.architecture == 'base':
                return crop_to_original(x_recon, self.padding_sizes) 
            else:
                return x_recon
        
        # VQ-VAE losses
        losses[f'{name}/perplexity'] = vq_output['perplexity'].to(self.idx_0)
        losses[f'{name}/commitment_loss'] = vq_output['commitment_loss'].to(self.idx_0)

        # Key modification to the model
        losses[f'{name}/recon_loss'] = F.l1_loss(x_recon, x_original) * self.l1_weight

        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, x_recon.shape[2], [B]).to(self.device)
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, x_recon.shape[3], x_recon.shape[4])
        frames = torch.gather(x_original, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)
        # Still VQ-VAE loss
        losses[f'{name}/perceptual_loss'] = self.perceptual_loss(frames, frames_recon)

        return x_recon, losses, (frames, frames_recon)
    
    def on_train_epoch_start(self):
        # Fix optimizer state
        if self.model_parallelism:
            print('Fixing optimizer state to have right device')
            opt, _ = self.optimizers()

            params = opt.optimizer.param_groups[0]['params']

            state = opt.optimizer.state

            for i, p in enumerate(params):
                if p in state:
                    for k, v in state[p].items():
                        if isinstance(v, torch.Tensor) and v.device != p.device:
                            state[p][k] = state[p][k].to(p.device)

        # Fix noisy decoder training
        if self.noisy_decoder:
            self.setup_noisy_decoder()


    def training_step(self, batch, batch_idx):
        opt_ae, opt_disc = self.optimizers()
        
        x = batch['data']
        if self.architecture == 'base':
            x_original = x.detach().clone()
        else:
            x_original = batch['data_original']

        x_recon, losses, (frames, frames_recon) = self.forward(x, x_original, name='train', batch_idx=batch_idx)
        
        # Losses VQ-VAE
        loss_ae = losses['train/recon_loss'] + losses['train/commitment_loss'] + losses['train/perceptual_loss']
        
        # Generator loss
        if self.discriminator_iter_start >= 0:
            disc_factor = adopt_weight(self.global_step, threshold=self.discriminator_iter_start)
            if disc_factor > 0:
                pred_image_fake, pred_video_fake, losses[f'train/g_image_loss'], losses[f'train/g_video_loss'], losses[f'train/g_loss'] = self.dg_loss(x_recon, frames_recon, disc_factor)
                losses[f'train/image_gan_feat_loss'], losses[f'train/video_gan_feat_loss'], losses[f'train/gan_feat_loss'] = self.gan_feat_loss(x_original, frames, pred_image_fake, pred_video_fake, disc_factor)
                loss_ae += losses[f'train/g_loss'] + losses[f'train/gan_feat_loss']

        losses['train/loss_ae'] = loss_ae

        opt_ae.zero_grad()
        self.manual_backward(loss_ae)
        self.clip_gradients(opt_ae, self.gradient_clip_val)
        opt_ae.step()

        # Discriminator loss (are there detatching errors?)
        if self.discriminator_iter_start >= 0:
            disc_factor = adopt_weight(self.global_step, threshold=self.discriminator_iter_start)
            if disc_factor > 0:
                _, _, _, _, losses[f'train_d/d_image_loss'], losses[f'train_d/d_video_loss'], losses[f'train_d/discloss'] = self.dd_loss(x_original, x_recon, frames, frames_recon, disc_factor)

                opt_disc.zero_grad()
                self.manual_backward(losses[f'train_d/discloss'])
                self.clip_gradients(opt_disc, self.gradient_clip_val)
                opt_disc.step()
            
        self.log_dict(losses, prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch['data']
        if self.architecture == 'base':
            x_original = x.detach().clone()
        else:
            x_original = batch['data_original']

        _, losses, frames = self.forward(x, x_original, name='val', batch_idx=batch_idx, dataloader_idx=dataloader_idx)
        
        loss_ae = losses['val/recon_loss'] + losses['val/commitment_loss'] + losses['val/perceptual_loss']
        losses['val/loss_ae'] = loss_ae
        self.val_step_metric.append(loss_ae)

        # Log image
        if batch_idx == 0:
            key = 'samples' if dataloader_idx == 0 else f'dl{dataloader_idx}_samples'
            self.logger.experiment.log({key: [wandb.Image(frames[0][0].detach().cpu(), caption='original'), wandb.Image(frames[1][0].detach().cpu().to(torch.float32), caption='recon')], 'trainer/global_step': self.global_step})
        
        # Used just when validating with more datasets
        if dataloader_idx != 0:
            # Update keys
            losses = {f'dl{dataloader_idx}_{k}': v for k, v in losses.items()}

        self.log_dict(losses, prog_bar=True, sync_dist=True, add_dataloader_idx=False)

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
        
        if self.discriminator_iter_start >= 0:
            opt_disc = torch.optim.Adam(list(self.image_discriminator.parameters()) +
                                        list(self.video_discriminator.parameters()),
                                        lr=lr, betas=(0.5, 0.9))
        else:
            dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
            opt_disc = torch.optim.Adam([dummy_param], lr=lr, betas=(0.5, 0.9))
        
        # compute start factor to begin with base_lr
        start_factor = self.base_lr/lr
        ae_scheduler = {'scheduler': torch.optim.lr_scheduler.LinearLR(opt_ae, start_factor=start_factor, total_iters=5), 'name': 'warmup-ae'}
        plateau_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt_ae, 'min', patience=20), 'name': 'plateau-ae'}
        
        return [opt_ae, opt_disc], [ae_scheduler, plateau_scheduler] 
    
    def denormalize_noise(self, noise):
        noise = (((noise + 1.0) / 2.0) * (self.codebook.embeddings.max() - self.codebook.embeddings.min())) + self.codebook.embeddings.min()

        return noise


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
    model = VQVAEUpsampling(embedding_dim=8,
                            n_codes=16384,
                            n_hiddens=16,
                            downsample=[4,4,4],
                            image_channels=1,
                            norm_type='group',
                            padding_type='replicate',
                            num_groups=16,
                            no_random_restart=False,
                            restart_thres=1.0,
                            d=192,
                            h=148,
                            w=216,
                            gan_feat_weight=1.0,
                            disc_channels=64,
                            disc_layers=3,
                            disc_loss_type='hinge',
                            image_gan_weight=1.0,
                            video_gan_weight=1.0,
                            perceptual_weight=1.0,
                            l1_weight=1.0,
                            gradient_clip_val=1.0,
                            discriminator_iter_start=0,
                            lr=1e-5,
                            base_lr=1e-5, 
                            original_d=456, 
                            original_h=352, 
                            original_w=512, 
                            architecture='up', 
                            architecture_down='base',
                            model_parallelism=False,
                            simple_architecture=False,
                            noise_prob=0,)
    

    summary(model, [(1, 192, 148, 216), (1, 456, 352, 512)], device='cpu', depth=10)

    print('end')