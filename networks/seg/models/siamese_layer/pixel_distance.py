import torch.nn as nn
import torch
import torch.nn.functional as F
from networks.base.cnn import ConvModule
from ..builder import SIAMESE_LAYER


@SIAMESE_LAYER.register_module()
class PixelDistance(nn.Module):

    def __init__(self, att_cfg=None):
        super(PixelDistance, self).__init__()

    def forward(self, n, g):
        return F.pairwise_distance(n, g, keepdim=True)


@SIAMESE_LAYER.register_module()
class PixelCat(nn.Module):
    def __init__(self, in_c, ou_c, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super(PixelCat, self).__init__()
        self.conv_modules = nn.Sequential(ConvModule(in_c * 2, in_c, 3, 1, 1,
                                                     norm_cfg=norm_cfg,
                                                     act_cfg=act_cfg
                                                     ),
                                          ConvModule(in_c, in_c // 2, 3, 1, 1,
                                                     norm_cfg=norm_cfg,
                                                     act_cfg=act_cfg
                                                     ))
        self.final_conv = nn.Conv2d(in_c // 2, ou_c, 1)

    def forward(self, n, g):
        f = torch.cat([n, g], dim=1)
        f = self.conv_modules(f)
        f = self.final_conv(f)
        return f


@SIAMESE_LAYER.register_module()
class PixelSub(nn.Module):
    def __init__(self, in_c, ou_c, add_conv=True):
        super(PixelSub, self).__init__()
        self.add_conv = add_conv
        self.conv_modules = nn.Sequential(ConvModule(in_c, in_c//2, 3, 1, 1),
                                          ConvModule(in_c // 2, in_c // 4, 3, 1, 1))
        self.final_conv = nn.Conv2d(in_c // 4, ou_c, 1)

    def forward(self, n, g):
        f = torch.tanh(n - g)
        if self.add_conv:
            f = self.conv_modules(f)
            f = self.final_conv(f)
            return f
        return f
