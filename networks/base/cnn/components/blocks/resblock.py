import torch.nn as nn
from ...components.conv_module import DepthwiseSeparableConvModule, ConvModule
from ...components.plugins import SELayer
from ...backbones.utils import make_divisible

class InvertedResidual(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 exp_ratio=4,
                 noskip=False,
                 use_se=False,
                 se_radio=16,
                 dw_norm_cfg=dict(type="BN2d"),
                 dw_act_cfg=dict(type="ReLU"),
                 pw_norm_cfg=dict(type="BN2d"),
                 pw_act_cfg=dict(type="ReLU"),
                 se_act_cfg=[dict(type='ReLU'), dict(type='Sigmoid')],
                 norm_cfg=dict(type="BN2d"),
                 **kwargs
                 ):
        super(InvertedResidual, self).__init__()

        mid_channels = int(in_channels * exp_ratio)
        self.has_residual = (in_channels == out_channels and stride == 1) and not noskip


        self.conv1 = ConvModule(in_channels, mid_channels, 1, 1, 0, bias=False, act_cfg=pw_act_cfg, norm_cfg=pw_norm_cfg)
        self.conv2 = ConvModule(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2,bias=False,
                                groups=mid_channels,act_cfg=dw_act_cfg, norm_cfg=dw_norm_cfg)

        self.se = SELayer(mid_channels, se_radio, act_cfg=se_act_cfg, rd_round_fn=make_divisible) if use_se else nn.Identity()

        self.conv3 = ConvModule(mid_channels, out_channels, 1, 1, 0,norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.se(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.has_residual:
            x += shortcut

        return x
