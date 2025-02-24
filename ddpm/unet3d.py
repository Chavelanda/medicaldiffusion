"Largely taken and adapted from https://github.com/lucidrains/video-diffusion-pytorch"

import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from einops import rearrange
from einops_exts import rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from ddpm.utils import exists, default, is_odd, prob_mask_like, pad_to_multiple, crop_to_original
from ddpm.text import BERT_MODEL_DIM


# relative positional bias
class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads=8,
        num_buckets=32,
        max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# small helper modules
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# lappala
def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        # lappala
        self.proj = nn.Conv3d(dim, dim_out, (3, 3, 3), padding=(1, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
        self,
        x,
        pos_bias=None,
        focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones(
                (n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

axis_to_one_dim_attn_einops = (
    ('b c d h w', 'b (h w) d c'),
    ('b c d h w', 'b (d w) h c'),
    ('b c d h w', 'b (d h) w c') 
)

axis_to_spatial_attn_einops = (
    ('b c d h w', 'b c d h w'),
    ('b c d h w', 'b c h d w'),
    ('b c d h w', 'b c w d h'),
)

class FactorizedAttention(nn.Module):
    def __init__(
        self, 
        dim,
        heads,
        dim_head=None,
        rotary_emb=None,
        pos_biases=[None, None, None],
        spatial=False,
        ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.rotary_emb = rotary_emb

        self.pos_biases = pos_biases

        self.spatial = spatial

        attn_func = self.spatial_attn if self.spatial else self.one_dim_attn

        # When attenion is spatial, it is done on the two remaining axes 
        # E.g. depth attention means attention on height-width axes
        self.depth_attention = Residual(PreNorm(dim, attn_func(axis=0)))
        self.height_attention = Residual(PreNorm(dim, attn_func(axis=1)))
        self.width_attention = Residual(PreNorm(dim, attn_func(axis=2)))
        
    def one_dim_attn(self, axis=0):
        attn = Attention(self.dim, heads=self.heads, dim_head=self.dim_head, rotary_emb=self.rotary_emb)
        return EinopsToAndFrom(axis_to_one_dim_attn_einops[axis][0], axis_to_one_dim_attn_einops[axis][1], attn)

    def spatial_attn(self, axis=0):
        attn = SpatialLinearAttention(self.dim, heads=self.heads)
        return EinopsToAndFrom(axis_to_spatial_attn_einops[axis][0], axis_to_spatial_attn_einops[axis][1], attn)

    def set_pos_biases(self, pos_biases):
        self.pos_biases = pos_biases

    def forward(self, xs, **kwargs):
        x0, x1, x2 = xs
        # Positional bias is only used for one dim attention
        if self.spatial:
            x0 = self.depth_attention(x0)
            # x1 = self.height_attention(x1)
            # x2 = self.width_attention(x2)
        else:
            x0 = self.depth_attention(x0, pos_bias=self.pos_biases[0], **kwargs)
            # x1 = self.height_attention(x1, pos_bias=self.pos_biases[1], **kwargs)
            # x2 = self.width_attention(x2, pos_bias=self.pos_biases[2], **kwargs)

        return x0, torch.zeros_like(x0), torch.zeros_like(x0) # x1, x2
        

# model


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        use_class_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type='resnet',
        resnet_groups=8
    ):
        super().__init__()
        self.channels = channels

        # relative positional encoding for one dim attention
        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        # realistically will not be able to generate that many frames of video... yet
        self.dim_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        # lappala
        self.init_conv = nn.Conv3d(channels, init_dim, (init_kernel_size, init_kernel_size, init_kernel_size), padding=(init_padding, init_padding, init_padding))

        self.init_one_dim_attention = FactorizedAttention(init_dim, attn_heads, attn_dim_head, rotary_emb=rotary_emb)

        # dimensions
        dim_mults = default(dim_mults, (1, 2, 4, 8))
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.pad_divisors = (len(dim_mults), len(dim_mults), len(dim_mults))

        # time conditioning
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # conditioning
        assert not (use_bert_text_cond and use_class_cond), 'cannot use both bert text cond and class cond'
        self.has_cond = exists(cond_dim) or use_bert_text_cond or use_class_cond

        # class conditioning
        self.use_class_cond = use_class_cond
        n_classes = cond_dim
        self.class_cond_mlp = nn.Embedding(n_classes, time_dim) if use_class_cond else None
        
        # text conditioning        
        cond_dim = BERT_MODEL_DIM if use_bert_text_cond else time_dim if use_class_cond else cond_dim

        # null conditioning embedding
        self.null_cond_emb = nn.Parameter(
            torch.randn(1, cond_dim)) if self.has_cond else None

        # final cond_dim
        cond_dim = time_dim + int(cond_dim or 0)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=cond_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),
                FactorizedAttention(dim_out, attn_heads, spatial=True) if use_sparse_linear_attn else nn.Identity(),
                FactorizedAttention(dim_out, attn_heads, attn_dim_head, rotary_emb=rotary_emb),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        self.mid_spatial_attn = FactorizedAttention(mid_dim, attn_heads, spatial=True)
        self.mid_one_dim_attention = FactorizedAttention(mid_dim, attn_heads, attn_dim_head, rotary_emb=rotary_emb)

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                FactorizedAttention(dim_in, attn_heads, spatial=True) if use_sparse_linear_attn else nn.Identity(),
                FactorizedAttention(dim_in, attn_heads, attn_dim_head, rotary_emb=rotary_emb),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def get_pos_biases(self, x):
        d_rel_pos_bias = self.dim_rel_pos_bias(x.shape[-3], device=x.device)
        h_rel_pos_bias = self.dim_rel_pos_bias(x.shape[-2], device=x.device)
        w_rel_pos_bias = self.dim_rel_pos_bias(x.shape[-1], device=x.device)
        return d_rel_pos_bias,h_rel_pos_bias,w_rel_pos_bias

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,
        null_cond_prob=0.,
        focus_present_mask=None,
        # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        prob_focus_present=0.
    ):
        assert not (self.has_cond and not exists(cond)
                    ), 'cond must be passed in if cond_dim specified'
        batch, device = x.shape[0], x.device

        # pad to make downsizeable
        x, padding_sizes = pad_to_multiple(x, self.pad_divisors)

        focus_present_mask = default(focus_present_mask, lambda: prob_mask_like(
            (batch,), prob_focus_present, device=device))

        pos_biases = self.get_pos_biases(x)

        x = self.init_conv(x)
        r = x.clone()

        self.init_one_dim_attention.set_pos_biases(pos_biases)
        x = sum(self.init_one_dim_attention((x, x, x)))

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance
        if self.has_cond:
            # embed cond
            cond = torch.squeeze(self.class_cond_mlp(cond)) if self.use_class_cond else cond

            # mask cond with null_cond_prob
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, cond)
            
            # concatenate time and cond embeddings
            t = torch.cat((t, cond), dim=-1)

        h = []

        for block1, block2, spatial_attn, one_dim_attention, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            
            xd, xh, xw = spatial_attn((x, x, x))
            one_dim_attention.set_pos_biases(pos_biases)
            x = sum(one_dim_attention((xd, xh, xw)))
            
            h.append(x)
            x = downsample(x)
            
            pos_biases = self.get_pos_biases(x)

        x = self.mid_block1(x, t)

        xd, xh, xw = self.mid_spatial_attn((x, x, x))
        self.mid_one_dim_attention.set_pos_biases(pos_biases)
        x = sum(self.mid_one_dim_attention((xd, xh, xw)))

        x = self.mid_block2(x, t)

        for block1, block2, spatial_attn, one_dim_attention, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)

            xd, xh, xw = spatial_attn((x, x, x))
            one_dim_attention.set_pos_biases(pos_biases)
            x = sum(one_dim_attention((xd, xh, xw)))

            x = upsample(x)

            pos_biases = self.get_pos_biases(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_conv(x)
        
        # crop to initial shape
        x = crop_to_original(x, padding_sizes)

        return x