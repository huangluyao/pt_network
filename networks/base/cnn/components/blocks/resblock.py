import torch.nn as nn
from ...components.conv_module import DepthwiseSeparableConvModule, ConvModule
from ...components.plugins import SELayer

class InvertedResidual(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 conv_cfg = None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True)):

        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.identity = stride ==1 and in_channels==out_channels

        if in_channels == mid_channels:
            self.conv = nn.Sequential(
                # dw
                DepthwiseSeparableConvModule(mid_channels, mid_channels,kernel_size,stride,(kernel_size-1)//2,
                                                     dw_act_cfg=act_cfg,dw_norm_cfg=norm_cfg,
                                                     pw_act_cfg=act_cfg, pw_norm_cfg=norm_cfg),
                SELayer(mid_channels, ratio=4) if use_se else nn.Identity())
        else:
            self.conv = nn.Sequential(ConvModule(in_channels, mid_channels,1, 1, 0,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg),
                      DepthwiseSeparableConvModule(mid_channels, mid_channels, kernel_size, stride, (kernel_size - 1) // 2,
                                                   dw_act_cfg=act_cfg, dw_norm_cfg=norm_cfg,
                                                   pw_act_cfg=act_cfg, pw_norm_cfg=norm_cfg),
                      SELayer(mid_channels, ratio=4) if use_se else nn.Identity(),
                      ConvModule(mid_channels, out_channels, 1, 1, 0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def forward(self, x):

        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
