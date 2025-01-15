"Largely taken and adapted from https://github.com/lucidrains/video-diffusion-pytorch"


import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape

from ddpm.utils import is_list_str, exists, default, unnormalize_img, normalize_img
from ddpm.text import tokenize, bert_embed
from vq_gan_3d.model.vqvae_upsampling import VQVAE_Upsampling


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        d,
        h,
        w,
        text_use_bert_cls=False,
        channels=3,
        timesteps=1000,
        loss_type='l1',
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.9,
        vqgan_ckpt=None,
    ):
        super().__init__()
        self.channels = channels
        
        self.d = d
        self.h = h
        self.w = w

        self.denoise_fn = denoise_fn

        if vqgan_ckpt:
            self.vqgan = VQVAE_Upsampling.load_from_checkpoint(vqgan_ckpt).cuda()
            self.vqgan.eval()
        else:
            self.vqgan = None

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps, disable=True):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale)

        return img

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1., batch_size=16):
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        _sample = self.p_sample_loop((batch_size, self.channels, self.d, self.h, self.w), cond=cond, cond_scale=cond_scale)

        if isinstance(self.vqgan, VQVAE_Upsampling):
            # denormalize TODO: Remove eventually
            _sample = (((_sample + 1.0) / 2.0) * (self.vqgan.codebook.embeddings.max() -
                                                  self.vqgan.codebook.embeddings.min())) + self.vqgan.codebook.embeddings.min()

            _sample = self.vqgan.decode(_sample, quantize=True)
        else:
            unnormalize_img(_sample)

        return _sample
    
    @torch.inference_mode()
    def ddim_sample(self, cond=None, cond_scale=1., batch_size=16, seq=[], eta=0):
        # Initialize noisy image
        batch_size = cond.shape[0] if exists(cond) else batch_size
        device = self.betas.device
        x = torch.randn((batch_size, self.channels, self.d, self.h, self.w), device=device)

        # shifted sequence of t, list of pred x0 and list of noisy images
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        # sample loop
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            next_t = torch.full((batch_size,), j, device=device, dtype=torch.long)

            xt = xs[-1].to(device)

            noise = self.denoise_fn.forward_with_cond_scale(xt, t, cond=cond, cond_scale=cond_scale)

            x0_t = self.predict_start_from_noise(xt, t, noise)
            x0_preds.append(x0_t.to('cpu'))

            at = extract(self.alphas_cumprod, t, xt.shape)

            # TODO: Handle better
            # Handle next_t = -1 case. 
            if j == -1:
                # Directly set the final xt_next to at.sqrt() * x0_t without noise
                xt_next = at.sqrt() * x0_t
            else:
                at_next = extract(self.alphas_cumprod, next_t, xt.shape)
                c1 = (eta * torch.sqrt((1 - at / at_next) * (1 - at_next) / (1 - at)))
                c2 = torch.sqrt((1 - at_next) - c1 ** 2)
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * noise
                xs.append(xt_next.to('cpu'))
        
        return xs[-1]


    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond=None, noise=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(cond):
            cond = bert_embed(
                tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)

        x_recon = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        if isinstance(self.vqgan, VQVAE_Upsampling):
            with torch.no_grad():
                x = self.vqgan.encode(
                    x, quantize=False, include_embeddings=True)
                # normalize to -1 and 1
                x = ((x - self.vqgan.codebook.embeddings.min()) /
                     (self.vqgan.codebook.embeddings.max() -
                      self.vqgan.codebook.embeddings.min())) * 2.0 - 1.0
        else:
            print("Hi")
            x = normalize_img(x)

        b, device, = x.shape[0], x.device
        
        check_shape(x, 'b c f h w', c=self.channels, f=self.d, h=self.h, w=self.w)

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x, t, *args, **kwargs)
    

# EMA model
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

