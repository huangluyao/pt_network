import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base.cnn.components import ConvModule
from .gated_conv_module import SimpleGatedConvModule
from ....builder import MODULES


@MODULES.register_module()
class GLDilationNeck(nn.Module):
    """
    Dilation Backbone used in Global&Local model.
    """
    _conv_type = dict(conv=ConvModule, gated_conv=SimpleGatedConvModule)

    def __init__(self,
                 in_channels=256,
                 conv_type='conv',
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__()
        conv_module = self._conv_type[conv_type]
        dilation_convs_ = []

        for i in range(4):
            dilation_ = int(2 ** (i + 1))
            dilation_convs_.append(
                conv_module(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding= dilation_,
                    dilation= dilation_,
                    stride=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **kwargs
                )
            )

        self.dilation_convs = nn.Sequential(*dilation_convs_)

    def forward(self, x):
        return self.dilation_convs(x)
