from collections import OrderedDict
import inspect
from typing import Optional, Tuple, Union
from diffusers import DDPMScheduler, DiffusionPipeline, ImagePipelineOutput
import torch

from vq_gan_3d.model.vqvae_upsampling import VQVAE_Upsampling 
from ddpm.unet3d import Unet3D

import matplotlib.pyplot as plt

class LDM3DPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler, vqvae):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vqvae=vqvae)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
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
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        latents = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_d, self.unet.sample_h, self.unet.sample_w),
            generator=generator,
        )
        latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(self.scheduler.timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            # predict the noise residual
            noise_prediction = self.unet(latent_model_input, t.expand(batch_size), **kwargs).sample
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_prediction, t, latents, **extra_kwargs).prev_sample

        # decode the image latents with the VAE
        image = self.vqvae.decode(latents, quantize=True).sample

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
    


if __name__ == '__main__':
    # Testing diffuser pipeline
    # Setting unet, scheduler, and vqvae
    unet = Unet3D(
        dim=32,
        out_dim=8,
        dim_mults=(1, 2, 4, 8),
        channels=8,
        sample_d=48,
        sample_h=37,
        sample_w=54,
        use_class_cond=True,
        cond_dim=5,
    ).to('cuda')

    # Load Unet3d weights
    checkpoint = torch.load('checkpoints/ddpm/AllCTs/allcts-051-512-classifier-free-class-embedding-ddpm-resume/model-best.pt')
    
    denoise_fn_state_dict = OrderedDict()
    for key in checkpoint['ema']:
        if key.startswith('module.denoise_fn'):
            new_key = key.replace("module.denoise_fn.", "") 
            denoise_fn_state_dict[new_key] = checkpoint['ema'][key]
    
    unet.load_state_dict(denoise_fn_state_dict)

    vqvae = VQVAE_Upsampling.load_from_checkpoint('checkpoints/vq_gan_3d/AllCTs-Upsampling/allcts-051-512-up-only-recon-ae/best_val-epoch=378-step=137577.ckpt').to('cuda')

    scheduler = DDPMScheduler(num_train_timesteps=300)

    pipeline = LDM3DPipeline(unet, scheduler, vqvae)

    output = pipeline(batch_size=1, num_inference_steps=300, cond=torch.tensor(0, dtype=torch.int32, device='cuda'), null_cond_prob=1.)

    print(output.images.shape)

    # plot middle slice of the generated image
    plt.imshow(output.images[0, 0, :, 160, :].cpu().numpy())
    plt.savefig('foo.png')
    plt.imshow(output.images[0, 0, 150, :, :].cpu().numpy())
    plt.savefig('foo2.png')
    plt.imshow(output.images[0, 0, :, :, 256].cpu().numpy())
    plt.savefig('foo3.png')
    print('test finished')

