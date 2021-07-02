from collections import OrderedDict
import torch
import torch.nn as nn

from ..activation import build_activation_layer
from ..conv import build_conv_layer
from ..norm import build_norm_layer
from ..plugin import build_plugin_layer
from ..registry import BLOCK_LAYERS


@BLOCK_LAYERS.register_module()
class ShuffleV2Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 *,
                 kernel_size=3,
                 stride=1,
                 se_cfg=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(ShuffleV2Block, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5, 7]
        self.stride = stride

        if stride == 1:
            in_channels = in_channels // 2
        mid_channels = in_channels

        branch_main_channels = out_channels - mid_channels
        branch_main = [
            ('0', build_conv_layer(
                conv_cfg,
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )),
            ('1', build_norm_layer(norm_cfg, mid_channels, anonymous=True)),
            ('2', build_activation_layer(act_cfg)),
            ('3', build_conv_layer(
                conv_cfg,
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=mid_channels,
                bias=False
            )),
            ('4', build_norm_layer(norm_cfg, mid_channels, anonymous=True)),
            ('5', build_activation_layer(act_cfg)),
            ('6', build_conv_layer(
                conv_cfg,
                in_channels=mid_channels,
                out_channels=branch_main_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )),
            ('7', build_norm_layer(norm_cfg, branch_main_channels, anonymous=True)),
            ('8', build_activation_layer(act_cfg)),
        ]

        if se_cfg is not None:
            branch_main.append((str(len(branch_main)), build_plugin_layer(
                se_cfg, anonymous=True, channels=branch_main_channels)))

        self.branch_main = nn.Sequential(OrderedDict(branch_main))

        if stride == 2:
            self.branch_proj = nn.Sequential(OrderedDict([
                ('0', build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=False
                )),
                ('1', build_norm_layer(norm_cfg, in_channels, anonymous=True)),
                ('2', build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                )),
                ('3', build_norm_layer(norm_cfg, in_channels, anonymous=True)),
                ('4', build_activation_layer(act_cfg))
            ]))
        else:
            self.branch_proj = None

    def forward(self, x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj, x = x, x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


@BLOCK_LAYERS.register_module()
class ShuffleXception(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 *,
                 kernel_size=3,
                 stride=1,
                 plugins=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(ShuffleXception, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5, 7]
        self.stride = stride

        if stride == 1:
            mid_channels = in_channels // 2
        elif stride == 2:
            mid_channels = in_channels

        branch_main_channels = out_channels - mid_channels
        self.branch_main = nn.Sequential(OrderedDict([
            ('0', build_conv_layer(
                conv_cfg,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False
            )),
            ('1', build_norm_layer(norm_cfg, in_channels, anonymous=True)),
            ('2', build_conv_layer(
                conv_cfg,
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )),
            ('3', build_norm_layer(norm_cfg, mid_channels, anonymous=True)),
            ('4', build_activation_layer(act_cfg)),
            ('5', build_conv_layer(
                conv_cfg,
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=mid_channels,
                bias=False
            )),
            ('6', build_norm_layer(norm_cfg, mid_channels, anonymous=True)),
            ('7', build_conv_layer(
                conv_cfg,
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )),
            ('8', build_norm_layer(norm_cfg, mid_channels, anonymous=True)),
            ('9', build_activation_layer(act_cfg)),
            ('10', build_conv_layer(
                conv_cfg,
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=mid_channels,
                bias=False
            )),
            ('11', build_norm_layer(norm_cfg, mid_channels, anonymous=True)),
            ('12', build_conv_layer(
                conv_cfg,
                in_channels=mid_channels,
                out_channels=branch_main_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )),
            ('13', build_norm_layer(norm_cfg, branch_main_channels, anonymous=True)),
            ('14', build_activation_layer(act_cfg)),
        ]))

        if stride == 2:
            self.branch_proj = nn.Sequential(OrderedDict([
                ('0', build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=False
                )),
                ('1', build_norm_layer(norm_cfg, in_channels, anonymous=True)),
                ('2', build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                )),
                ('3', build_norm_layer(norm_cfg, in_channels, anonymous=True)),
                ('4', build_activation_layer(act_cfg))
            ]))
        else:
            self.branch_proj = None

    def forward(self, x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj, x = x, x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


def channel_shuffle(x):
    B, C, H, W = x.data.size()
    assert (C % 4 == 0)
    x = x.reshape(B, C // 2, 2, H * W)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(B, 2, C // 2, H, W)
    return x[:, 0], x[:, 1]
