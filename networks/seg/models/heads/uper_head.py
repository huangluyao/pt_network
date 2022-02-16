import torch
import torch.nn as nn
import torch.nn.functional as F
from base.cnn import (ConvModule, resize, )
from .decode_head import BaseDecodeHead

from ..builder import HEADS


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """
    def __init__(self, pool_scales, in_channels, channels,
                                  conv_cfg, norm_cfg, act_cfg, align_corners,
                                  **kwargs):
        super(PPM, self).__init__()

        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                ConvModule(self.in_channels,
                           self.channels,
                           1,
                           conv_cfg=conv_cfg,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg,
                           **kwargs
                           )
            ))

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(ppm_out,
                                       size=x.size()[2:],
                                       mode='bilinear',
                                       align_corners=self.align_corners
                                       )
            ppm_outs.append(upsampled_ppm_out)

        return ppm_outs


@HEADS.register_module()
class UPerHead(BaseDecodeHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(**kwargs)

        self.psp_modules = PPM(pool_scales,
                              self.in_channels[-1],
                              self.head_width,
                              conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg,
                              act_cfg=self.act_cfg,
                              align_corners=self.align_corners
                              )

        self.bottleneck = ConvModule(
           self.in_channels[-1] + len(pool_scales) * self.head_width,
           self.head_width,
           3,
           padding=1,
           conv_cfg = self.conv_cfg,
           norm_cfg = self.norm_cfg,
           act_cfg = self.act_cfg)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in self.in_channels[:-1]:

            l_conv = ConvModule(
                in_channels,
                self.head_width,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False
            )

            fpn_conv = ConvModule(
                self.head_width,
                self.head_width,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.head_width,
            self.head_width,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def forward(self, inputs):

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # psp
        high, psp_outs = inputs[-1], [inputs[-1]]
        psp_outs.extend(self.psp_modules(high))
        psp_outs = torch.cat(psp_outs, dim=1)
        laterals.append(self.bottleneck(psp_outs))

        # build top-down path
        levels_use = len(laterals)

        for i in range(levels_use - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners
            )

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(levels_use - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        # resize
        for i in range(levels_use - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)
        return output
