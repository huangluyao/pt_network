import torch
import torch.nn as nn
from ..components.conv_module import ConvModule
from .utils import make_divisible

class EfficientNet(nn.Module):

    def __init__(self,in_channels,
                 stage_channels=[24, 128, 256, 512, 1024],
                 out_levels=[3, 4, 5],
                 block_numbers=[3, 9, 9, 3],
                 strides=[2, 2, 2, 2],
                 depth_multiple=1,
                 width_multiple=1,
                 use_spp=True,
                 use_transformer=False,
                 num_classes=None,
                 act_cfg=dict(type="SiLu"),
                 norm_cfg = dict(type= "BN2d"),
                 **kwargs):

        self.stage_channels = [make_divisible(x * width_multiple, 8) for x in stage_channels]

        for i in range(3):
            if i == 0:
                conv_stem = [ConvModule(in_channels, self.stage_channels[0], kernel_size=3,
                                    padding=1, stride=2,act_cfg=act_cfg, norm_cfg=norm_cfg
                                    )]
            else:
                conv_stem.append(ConvModule(self.stage_channels[0], self.stage_channels[0], kernel_size=3,
                                    padding=1, stride=2,act_cfg=act_cfg, norm_cfg=norm_cfg, shortcut=True
                                    ))

        self.conv_stem = nn.Sequential(*conv_stem)


