""" Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from vq_gan_3d.utils import shift_dim


class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim, no_random_restart=False, restart_thres=1.0):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        # z: [b, c, t, h, w]
        self._need_init = False
        if torch.any(self.N):
            pass
        else:
            flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
            y = self._tile(flat_inputs)

            d = y.shape[0]
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)
            self.embeddings.data.copy_(_k_rand)
            self.z_avg.data.copy_(_k_rand)
            self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        # we want each vector with length 8. let's call it z_n
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)  # [bthw, c]
        # for each z_n, calculating the distance with all the vectors in the codebook 
        distances = torch.cdist(flat_inputs, self.embeddings, p=2)  # [bthw, n_codes]
        # getting the index of the closest one
        encoding_indices = torch.argmin(distances, dim=1) # [bthw]
        # for each code, compute how many times it was "called"
        n_total = torch.bincount(encoding_indices, minlength=self.n_codes) # [n_codes]

        # for each code index, sum all z_n mapped to it
        encode_sum = torch.zeros((self.n_codes, self.embedding_dim), device=flat_inputs.device, dtype=torch.float)
        encode_sum.index_add_(0, encoding_indices, flat_inputs.type_as(encode_sum))

        # Reshape indices as input data. For each z_n we have an index
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])  # [b, t, h, w]
        
        # Extract corresponding code from codebook
        embeddings = F.embedding(
            encoding_indices, self.embeddings)  # [b, t, h, w, c]
        embeddings = shift_dim(embeddings, -1, 1)  # [b, c, t, h, w]

        # Calculate how much the input differs from the embedding
        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum, alpha=0.01)
          
            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1)
                         >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = n_total / flat_inputs.shape[0]
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity)

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings
