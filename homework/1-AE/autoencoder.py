import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import RegularBlock, DenoisingBlock       

class AutoEncoder(nn.Module):
    def __init__(self, block=RegularBlock):
        super().__init__()
        self.encoder = nn.Sequential(
            block(1, 16, 3, stride=2),
            block(16, 32, 3, stride=2),
            block(32, 32, 3, stride=2),
            block(32, 32, 3, stride=2),
            block(32, 32, 3, stride=1).conv,
        )
        
        self.decoder = nn.Sequential(
            block(32, 32, 3, upsample=True),
            block(32, 32, 3, upsample=True),
            block(32, 32, 3, upsample=True),
            block(32, 16, 3, upsample=True),
            block(16, 1, 3).conv,
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x
    
    def get_latent_features(self, x):
        return self.encoder(x)


class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self):
        super().__init__(block=DenoisingBlock)
    