from typing import Optional
import inspect
from einops import rearrange
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel

import pytorch_lightning as pl
from diffusers.training_utils import EMAModel

from vq_gan_3d.model.vqvae_upsampling import VQVAEUpsampling
from ddpm.unet3d import Unet3D
from ddpm.ldm3d_pipeline import LDM3DPipeline
from ddpm.utils import video_tensor_to_gif


class Diffuser(pl.LightningModule):
    def __init__(self, 
                vqvae_ckpt,
                noise_scheduler_class, 
                in_channels=1,
                sample_d=64,
                sample_h=64,
                sample_w=64,
                dim=32,
                dim_mults=(1,2,4,8),
                attn_heads=8,
                attn_dim_head=32,
                use_class_cond=False,
                cond_dim=None,
                init_kernel_size=7,
                use_sparse_linear_attn=True,
                resnet_groups=8,
                null_cond_prob=0.1,
                ema_decay=0.995,
                update_ema_every_n_steps=1,
                loss='l2', 
                lr=1e-4,
                results_folder=None,
                training_timesteps=300):
        super().__init__()
        self.vqvae = VQVAEUpsampling.load_from_checkpoint(vqvae_ckpt)
        self.vqvae.eval()
        self.vqvae.freeze()

        self.unet = self.setup_unet(in_channels=in_channels, sample_d=sample_d, sample_h=sample_h, sample_w=sample_w, dim=dim, dim_mults=dim_mults, attn_heads=attn_heads, attn_dim_head=attn_dim_head, use_class_cond=use_class_cond, cond_dim=cond_dim, init_kernel_size=init_kernel_size, use_sparse_linear_attn=use_sparse_linear_attn, resnet_groups=resnet_groups)
        self.noise_scheduler = noise_scheduler_class(num_train_timesteps=training_timesteps)
        self.training_timesteps = training_timesteps

        self.ema_model = AveragedModel(self.unet, 
                                       multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
                                       use_buffers=True,
                                       )

        self.pipeline = LDM3DPipeline(self.ema_model.module, self.noise_scheduler, self.vqvae)

        
        self.ema_model.to(self.device)

        self.use_class_cond = use_class_cond
        self.null_cond_prob = null_cond_prob
        self.cond_dim = cond_dim
        
        self.ema_decay = ema_decay
        self.update_ema_every_n_steps = update_ema_every_n_steps

        assert loss in ['l1', 'l2'], 'Loss must be either l1 or l2'
        self.loss = F.mse_loss if loss == 'l2' else F.l1_loss

        self.lr = lr

        self.results_folder = results_folder

        self.save_hyperparameters(ignore=['noise_scheduler'])

    def setup_unet(self, **kwargs):
        return Unet3D(**kwargs)

    def forward(self, x, cond=None, use_ema=False, timesteps=None):
        # encode the image
        x = self.vqvae.encode(x, quantize=False, include_embeddings=True)

        # normalize the image
        x = ((x - self.vqvae.codebook.embeddings.min()) /
                     (self.vqvae.codebook.embeddings.max() -
                      self.vqvae.codebook.embeddings.min())) * 2.0 - 1.0

        # sample the noise
        noise = torch.randn_like(x)

        # sample a random timestep for each image
        if timesteps is None:
            batch_size = x.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=x.device).long()

        # add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)

        # predict the noise residual
        if use_ema:
            noise_prediction = self.ema_model(noisy_x, timesteps, cond, self.null_cond_prob)
        else:
            noise_prediction = self.unet(noisy_x, timesteps, cond, self.null_cond_prob)

        return noise_prediction, noise
    
    def training_step(self, batch):
        x = batch['data']
        cond = batch['cond'] if self.use_class_cond else None
        
        noise_prediction, noise = self(x, cond)
        loss = self.loss(noise_prediction, noise)

        self.log_dict({'train/loss': loss}, prog_bar=True, on_step=True, on_epoch=False, rank_zero_only=True)

        return loss
    
    def on_before_zero_grad(self, optimizer):
        super().on_before_zero_grad(optimizer)
        # update the EMA model
        if self.global_step % self.update_ema_every_n_steps == 0:
            self.ema_model.update_parameters(self.unet)
    
    def validation_step(self, batch):
        x = batch['data']
        cond = batch['cond'] if self.use_class_cond else None
        batch_size = x.shape[0]

        timesteps = torch.arange(self.noise_scheduler.config.num_train_timesteps, device=x.device).long().expand(batch_size, -1).T
        outputs = [self(x, cond, use_ema=True, timesteps=t) for t in timesteps]
        loss_l1 = torch.stack([F.l1_loss(noise_prediction, noise) for noise_prediction, noise in outputs]).mean()
        loss_l2 = torch.stack([F.mse_loss(noise_prediction, noise) for noise_prediction, noise in outputs]).mean()

        self.log_dict({'val/loss_l1': loss_l1, 'val/loss_l2': loss_l2}, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return loss_l1, loss_l2
    
    def sample_with_random_cond(self):
        cond = torch.randint(0, self.cond_dim, (1,), device=self.device) if self.use_class_cond else None
        print('\nSampling with condition:', cond)
        return self.sample(batch_size=1, num_inference_steps=self.training_timesteps, cond=cond)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr)
        return optimizer

    @torch.no_grad()
    def sample_latent(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        cond: torch.Tensor = None,
        cond_scale = None,
    ) -> torch.Tensor:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                Number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            cond (`torch.Tensor`, *optional*):
                The class condition to use for the generation.

        Returns:
            `torch.Tensor`: The generated images.
        """
        latents = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_d, self.unet.sample_h, self.unet.sample_w),
            generator=generator,
        )
        latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma

        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in tqdm(self.noise_scheduler.timesteps):
            latent_model_input = self.noise_scheduler.scale_model_input(latents, t)
            # predict the noise residual
            if cond_scale is not None:
                noise_prediction = self.ema_model.module.forward_with_cond_scale(latent_model_input, t.expand(batch_size), cond=cond, cond_scale=cond_scale)
            else:    
                noise_prediction = self.ema_model(latent_model_input, t.expand(batch_size), cond=cond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_prediction, t, latents, **extra_kwargs).prev_sample

        return latents

    @torch.no_grad()
    def sample(self,
               batch_size: int = 1,
               generator: Optional[torch.Generator] = None,
               eta: float = 0.0,
               num_inference_steps: int = 50,
               cond: torch.Tensor = None,
               cond_scale = None,):
        
        latents = self.sample_latent(batch_size=batch_size, generator=generator, eta=eta, num_inference_steps=num_inference_steps, cond=cond, cond_scale=cond_scale)

        # denormalize the latents
        latents = (((latents + 1.0) / 2.0) * (self.vqvae.codebook.embeddings.max() - self.vqvae.codebook.embeddings.min())) + self.vqvae.codebook.embeddings.min()

        # decode the image latents with the VAE
        samples = self.vqvae.decode(latents, quantize=True)

        return samples