import torch
import torch.nn.functional as F

from vq_gan_3d.model.vqgan import pad_to_multiple
from vq_gan_3d.model.vqvae_upsampling import VQVAEUpsampling

class VQVAEUpsamplingNoisyDecoder(VQVAEUpsampling):

    def __init__(self, *args, variance=0.1, **kwargs):
        super().__init__(*args, **kwargs)

        self.variance = variance

        # Freeze the encoder and the codebook (KEY MODEL MODIFICATION)
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

        self.save_hyperparameters()

    def forward(self, x, x_original=None, name='train'):
        # Pad image so that it is divisible by downsampling scale
        x, _ = pad_to_multiple(x, self.cfg.model.downsample)

        B, C, T, H, W = x.shape

        losses = {}

        z = self.pre_vq_conv(self.encoder(x))

        # Add noise to the latents (KEY MODEL MODIFICATION)
        z = z + torch.randn_like(z) * self.variance

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