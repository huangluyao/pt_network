import torch
import torch.nn as nn

from base.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, ConvModule, resize, _BatchNorm)
from .decode_head import BaseDecodeHead
from ..builder import HEADS
from ..criterions import accuracy


class SpatialPath(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 num_downscales=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(SpatialPath, self).__init__()
        kwargs = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        conv_layers = []
        for i in range(num_downscales):
            if i == 0:
                ksize = 7
            else:
                ksize = 3
                in_channels = mid_channels
            conv_layers.append(ConvModule(in_channels, mid_channels, kernel_size=ksize,
                                          stride=2, padding=ksize // 2, **kwargs))
        conv_layers.append(ConvModule(mid_channels, out_channels,
                                      kernel_size=1, stride=1, padding=0, **kwargs))
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        return x


class AttentionRefinement(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(AttentionRefinement, self).__init__()

        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='Sigmoid'))
        )

    def forward(self, x):
        feature = self.conv1(x)
        attention = self.channel_attention(feature)
        return feature * attention


class FeatureFusion(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(FeatureFusion, self).__init__()

        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                out_channels,
                out_channels // reduction,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=act_cfg),
            ConvModule(
                out_channels // reduction,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=None,
                act_cfg=dict(type='Sigmoid'))
        )

    def forward(self, x1, x2):
        feature = torch.cat([x1, x2], dim=1)
        feature = self.conv1(feature)
        attention = self.channel_attention(feature)
        return feature + feature * attention


class SegHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 head_width=64,
                 final_drop=0.1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(SegHead, self).__init__()

        self.conv1 = ConvModule(
            in_channels, head_width, kernel_size=3, stride=1, padding=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if final_drop > 0:
            self.dropout = nn.Dropout2d(final_drop)
        else:
            self.dropout = None
        self.conv_seg = nn.Conv2d(head_width, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv_seg(x)


@HEADS.register_module()
class BiSeNetHead(BaseDecodeHead):
    def __init__(self,
                 use_spatial_path=True,
                 num_arms=2,
                 refine_channels=128,
                 deep_supervision=False,
                 aux_weight=0.4,
                 se_ratio=1,
                 **kwargs):
        super(BiSeNetHead, self).__init__(**kwargs)
        self.deepsv = deep_supervision
        self.aux_weight = aux_weight
        layer_cfg = dict(conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.layer_cfg = layer_cfg

        if use_spatial_path:
            self.spatial_path = SpatialPath(self.in_channels[0], refine_channels,
                                            num_downscales=5-num_arms, **layer_cfg)
        else:
            self.spatial_path = None

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(self.in_channels[-1], refine_channels,
                       kernel_size=1, stride=1, padding=0, **layer_cfg))

        arms = []
        for i in range(num_arms):
            idx = -1 - i
            arms.append(AttentionRefinement(self.in_channels[idx], refine_channels, **layer_cfg))
        self.arms = nn.ModuleList(arms)

        refines = []
        for i in range(num_arms):
            refines.append(ConvModule(refine_channels, refine_channels,
                                      kernel_size=3, stride=1, padding=1, **layer_cfg))
        self.refines = nn.ModuleList(refines)

        if use_spatial_path:
            ffm_channels = refine_channels + refine_channels
        else:
            ffm_channels = refine_channels + self.in_channels[0]
        self.ffm = FeatureFusion(ffm_channels, refine_channels * 2, reduction=se_ratio, **layer_cfg)

        heads = []
        if self.deepsv:
            for i in range(num_arms):
                heads.append(SegHead(refine_channels, self.num_classes, head_width=256,
                                     final_drop=self.final_drop, **layer_cfg))

        heads.append(SegHead(refine_channels * 2, self.num_classes, head_width=self.head_width,
                             final_drop=self.final_drop, **layer_cfg))
        self.heads = nn.ModuleList(heads)

    def forward(self, inputs):
        """Forward function.

        Parameters
        ----------
        inputs : list[Tensor]
            List of multi-level img features.

        Returns
        -------
        Tensor
            The output of BiSeNetHead.
        """
        if self.spatial_path is not None:
            spatial_out = self.spatial_path(inputs[0])
        else:
            spatial_out = inputs[0]
        global_context = self.global_context(inputs[-1])
        global_context = resize(global_context,
                                size=inputs[-1].size()[2:],
                                mode='bilinear',
                                align_corners=self.align_corners)
        last_feature = global_context

        features = inputs[1:][::-1]
        feat_size = [x.size()[2:] for x in features]
        feat_size.append(spatial_out.size()[2:])
        pred_out = []
        for i, (feature, arm, refine) in enumerate(zip(features,
                                                       self.arms,
                                                       self.refines)):
            feature = arm(feature)
            feature += last_feature
            last_feature = resize(feature,
                                  size=feat_size[i+1],
                                  mode='bilinear',
                                  align_corners=self.align_corners)
            last_feature = refine(last_feature)
            pred_out.append(last_feature)
        context_out = last_feature

        concate_feature = self.ffm(spatial_out, context_out)
        pred_out.append(concate_feature)

        if self.training:
            if not self.deepsv:
                pred_out = pred_out[-1:]
            seg_logit = [head(pred) for (head, pred) in zip(self.heads, pred_out)]
            if len(seg_logit) == 1:
                seg_logit = seg_logit[0]
            return seg_logit
        else:
            return self.heads[-1](pred_out[-1])

    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_label = seg_label.squeeze(1)
        input_size = seg_label.shape[1:]

        if isinstance(seg_logit, (tuple, list)):
            loss_aux = 0.0
            for aux_logit in seg_logit[:-1]:
                aux_logit = resize(input=aux_logit,
                                   size=input_size,
                                   mode='bilinear',
                                   align_corners=self.align_corners)
                loss_aux += self.loss(aux_logit,
                                      seg_label,
                                      weight=None,
                                      ignore_label=self.ignore_label)
            seg_logit = resize(input=seg_logit[-1],
                               size=input_size,
                               mode='bilinear',
                               align_corners=self.align_corners)
            loss_seg = self.loss(seg_logit,
                                 seg_label,
                                 weight=None,
                                 ignore_label=self.ignore_label)
            loss_seg = self.aux_weight * loss_aux + loss_seg
        else:
            seg_logit = resize(input=seg_logit,
                               size=input_size,
                               mode='bilinear',
                               align_corners=self.align_corners)
            loss_seg = self.loss(seg_logit,
                                 seg_label,
                                 weight=None,
                                 ignore_label=self.ignore_label)
        loss['loss_seg'] = loss_seg
        return loss
