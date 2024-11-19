import torch
import torch.nn as nn

from vq_gan_3d.model import VQGAN

class FIDEncoder(nn.Module):
    
    def __init__(self, ckpt: str=None):
        super(FIDEncoder, self).__init__()
        self.ae = VQGAN.load_from_checkpoint(ckpt)
        self.pooling = nn.AdaptiveAvgPool3d(8)
        self.num_features = 4096

    def forward(self, input):
        
        features = self.ae.encode(input, quantize=False)
        features = self.pooling(features)
        features = torch.flatten(features, start_dim=1)

        return features
    
