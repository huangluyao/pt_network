import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base.cnn import ConvModule

from ..builder import NECKS
from ...specific.seg_block import SelfAttentionBlock


'''image-level context module'''
class ImageLevelContext(nn.Module):
    def __init__(self, feats_channels, transform_channels, concat_input=False,
                 norm_cfg=dict(type='BN2d'),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 ):
        super(ImageLevelContext, self).__init__()

        self.align_corners = align_corners
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.correlate_net = SelfAttentionBlock(
            key_in_channels=feats_channels * 2,
            query_in_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        if concat_input:
            self.bottleneck = ConvModule(feats_channels *2, feats_channels,  kernel_size=3, stride=1, padding=1,
                                         norm_cfg=norm_cfg, act_cfg=act_cfg
                                         )

    '''forward'''
    def forward(self, x):
        x_global = self.global_avgpool(x)
        x_global = F.interpolate(x_global, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners)
        feats_il = self.correlate_net(x, torch.cat([x_global, x], dim=1))
        if hasattr(self, 'bottleneck'):
            feats_il = self.bottleneck(torch.cat([x, feats_il], dim=1))
        return feats_il


@NECKS.register_module()
class ImageLevelContextNeck(nn.Module):
    def __init__(self,
                 in_channels=[16, 32, 128, 256, 512, 1024],
                 use_level=[2, 3, 4],
                 transform_channels=256,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(ImageLevelContextNeck, self).__init__()

        self.use_level= use_level
        for i in range(len(in_channels)):
            if i in use_level:
                setattr(self, f"ilcm_{i}", ImageLevelContext(in_channels[i], transform_channels,
                                                concat_input=True, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def forward(self, inputs):
        outputs = []
        for i, x in enumerate(inputs):
            if i in self.use_level:
                outputs.append(getattr(self, f"ilcm_{i}")(x))
            else:
                outputs.append(x)
        return outputs