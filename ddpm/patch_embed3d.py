""" 
A 3D adaptation to Image Patch Embedding
Image to Patch Embedding using Conv3d
A convolution based approach to patchifying a 3D image w/ embedding projection.

Adapted from:
https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py
"""
import logging
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn as nn
import torch.nn.functional as F

from timm.layers.helpers import to_3tuple
from timm.layers.trace_utils import _assert

_logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[Union[int, Tuple[int, int, int]]] = 224,
            patch_size: int = 16,
            in_chans: int = 1,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
            strict_img_size: bool = False,
            dynamic_img_pad: bool = True,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.original_img_size = to_3tuple(img_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # flatten spatial dim and transpose to channels last
        self.flatten = flatten
        
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        img_size = img_size if not dynamic_img_pad else (s * p for s,p in zip(self.dynamic_feat_size(self.original_img_size), self.patch_size))
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_3tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        return img_size, grid_size, num_patches

    def set_input_size(
            self,
            img_size: Optional[Union[int, Tuple[int, int, int]]] = None,
            patch_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_3tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            with torch.no_grad():
                new_proj = nn.Conv3d(
                    self.proj.in_channels,
                    self.proj.out_channels,
                    kernel_size=new_patch_size,
                    stride=new_patch_size,
                    bias=self.proj.bias is not None,
                )
                new_proj.weight.copy_(resample_patch_embed(self.proj.weight, new_patch_size, verbose=True))
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """ Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(img_size[1] / self.patch_size[1]), math.ceil(img_size[2] / self.patch_size[2])
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1], img_size[2] // self.patch_size[2]

    def forward(self, x):
        B, C, D, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(D == self.img_size[0], f"Input height ({D}) doesn't match model ({self.img_size[0]}).")
                _assert(H == self.img_size[1], f"Input height ({H}) doesn't match model ({self.img_size[1]}).")
                _assert(W == self.img_size[2], f"Input width ({W}) doesn't match model ({self.img_size[2]}).")
            elif not self.dynamic_img_pad:
                _assert(
                    D % self.patch_size[0] == 0,
                    f"Input height ({D}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    H % self.patch_size[1] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[1]})."
                )
                _assert(
                    W % self.patch_size[2] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[2]})."
                )
        if self.dynamic_img_pad:
            pad_d = (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0]
            pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
            pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x
    
    def unpatchify(self, x):
        """Unpatchify the input tensor"""
        p = self.patch_size[0]

        d, h, w = (s // p for s in self.img_size)
        c = self.in_chans
        
        assert d*h*w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], d, h, w, p, p, p, c))
        x = torch.einsum('ndhwpqrc->ncdphqwr', x)
        imgs = x.reshape(shape=(x.shape[0], c, d * p, h * p, w * p))

        if self.dynamic_img_pad:
            imgs = imgs[:, :, :self.original_img_size[0], :self.original_img_size[1], :self.original_img_size[2]]

        return imgs

def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np
    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_size) == 2, "New shape should only be hw"
    old_size = patch_embed.shape[-2:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    if verbose:
        _logger.info(f"Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation.")

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.tensor(np.linalg.pinv(resize_mat.T), device=patch_embed.device)

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    orig_dtype = patch_embed.dtype
    patch_embed = patch_embed.float()
    patch_embed = v_resample_kernel(patch_embed)
    patch_embed = patch_embed.to(orig_dtype)
    return patch_embed

