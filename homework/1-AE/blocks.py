import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class RegularBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, \
                bias=False, upsample=False):
                
        super().__init__()
        self.upsample = upsample

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, \
                        stride=stride, padding=(kernel-1)//2, bias=bias)
        self.activ = nn.ELU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return self.activ(self.batchnorm(self.conv(x)))


class DenoisingBlock(RegularBlock):
    def __init__(self, in_channels, out_channels, kernel, stride=1, \
              bias=False, upsample=False):

        super().__init__(in_channels, out_channels, kernel, stride, \
              bias=bias, upsample=upsample)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        x = x + torch.randn_like(x) * 0.05
        x = self.dropout(x)
        return self.activ(self.batchnorm(self.conv(x)))
