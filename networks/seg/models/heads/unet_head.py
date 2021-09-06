import torch
import torch.nn as nn

from base.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, ConvModule, resize, _BatchNorm)
from .decode_head import BaseDecodeHead
from ..builder import HEADS
from ..criterions import accuracy


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

        self.conv1 = ConvModule(in_channels, head_width, kernel_size=3, stride=1, padding=1,
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
class UNetHead(BaseDecodeHead):
    """U-Net: Convolutional Networks for Biomedical Image Segmentation.

    This head is implemented of `U-Net <https://arxiv.org/abs/1505.04597>`_.
    """

    def __init__(self,
                 num_upscales=4,
                 refine_channels=128,
                 deep_supervision=False,
                 aux_weight=0.4,
                 **kwargs):
        super(UNetHead, self).__init__(**kwargs)
        self.deepsv = deep_supervision
        self.aux_weight = aux_weight
        layer_cfg = dict(conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.layer_cfg = layer_cfg

        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(self.in_channels[-1], refine_channels,
                       kernel_size=1, stride=1, padding=0, **layer_cfg))

        arms = []
        for i in range(num_upscales):
            idx = -1 - i
            arms.append(AttentionRefinement(self.in_channels[idx], refine_channels, **layer_cfg))
        self.arms = nn.ModuleList(arms)

        refines = []
        for i in range(num_upscales):
            refines.append(ConvModule(refine_channels, refine_channels,
                                      kernel_size=3, stride=1, padding=1, **layer_cfg))
        self.refines = nn.ModuleList(refines)

        heads = []
        if self.deepsv:
            for i in range(num_upscales-1):
                heads.append(SegHead(refine_channels, self.num_classes, head_width=256,
                                     final_drop=self.final_drop, **layer_cfg))

        heads.append(SegHead(refine_channels, self.num_classes, head_width=self.head_width,
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
            The output of UNetHead.
        """
        global_context = self.global_context(inputs[-1])
        global_context = resize(global_context,
                                size=inputs[-1].size()[2:],
                                mode='bilinear',
                                align_corners=self.align_corners)
        last_feature = global_context

        features = inputs[::-1]
        feat_size = [x.size()[2:] for x in features]
        features = features[:-1]
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
