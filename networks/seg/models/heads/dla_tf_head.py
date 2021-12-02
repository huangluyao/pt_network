import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from base.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, ConvModule, resize, _BatchNorm)
from ..builder import HEADS, build_loss
from ..criterions.detail_aggregate_loss import DetailAggregateLoss
from .memory_head import ASPP, FeaturesMemory
from ...specific.samplers import build_pixel_sampler


class FusionLayer(nn.Module):

    def __init__(self,in_channels, out_channel, kernel=3, up_factors=2,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU6', inplace=True),
                 align_corners=True,
                 ):
        super(FusionLayer, self).__init__()

        assert isinstance(in_channels, list) and len(in_channels)==2 ,\
            f"FusionLayer in_channels must be a list and len(in_channels) == 2"


        self.conv = ConvModule(in_channels[1], in_channels[0], kernel_size=1, stride=1,
                          padding=0, norm_cfg=None, act_cfg=None)

        self.node = ConvModule(in_channels[0] * 2, out_channel, kernel_size=kernel, stride=1,
                          padding=kernel // 2,
                          norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up = nn.Upsample(scale_factor=up_factors, mode="bilinear", align_corners=align_corners)
        self.init_weights()

    def forward(self, high_feature, low_feature):

        out = self.up(self.conv(high_feature))
        out = torch.cat([out, low_feature], dim=1)
        out = self.node(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class FusionUP(nn.Module):
    def __init__(self, in_channels,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU6'),
                 align_corners=False,
                 scale_factor = 2
                 ):
        super(FusionUP, self).__init__()

        n_channel = len(in_channels)
        assert n_channel > 1, "in_channels should be lager than 1"

        for stage in range(n_channel - 1):
            for index in range(n_channel -1 -stage):
                setattr(self, f"fusion_layer_{stage}_{index}",
                        FusionLayer(in_channels[index:index+2], in_channels[index],norm_cfg=norm_cfg, act_cfg=act_cfg,
                                    align_corners=align_corners)
                        )

        self.final_conv = ConvModule(in_channels[0], in_channels[0], kernel_size=1, use_bias=True,
                                     norm_cfg=None, act_cfg=None)
        self.up = nn.Upsample(scale_factor=scale_factor , mode="bilinear", align_corners=align_corners)

    def forward(self, layers, out_list=False):
        stage_in_layers = list(layers)
        assert len(layers) > 1

        nLayer = len(layers)
        if out_list:
            outs = []

        for stage in (range(nLayer -1)):
            temp_out_list = []
            for index in range(nLayer - 1 - stage):
                x = getattr(self,  f"fusion_layer_{stage}_{index}")(stage_in_layers[index+1],
                                                                    stage_in_layers[index])
                temp_out_list.append(x)
            if out_list:
                outs.append(x)
            stage_in_layers = temp_out_list

        outs.append(self.up(self.final_conv(outs[-1])))

        return outs if out_list else outs[-1]


@HEADS.register_module()
class DLATFHead(nn.Module):

    def __init__(self, num_classes,
                 in_channels=[16, 32, 128, 256, 512, 1024],
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_label=None,
                 align_corners=True,
                 dropout=0.,
                 sampler=None,
                 fuse_finals=True,
                 **kwargs):
        super(DLATFHead, self).__init__()

        self.num_classes = num_classes

        if isinstance(loss, dict):
            self.loss_decode = build_loss(loss)
        elif isinstance(loss, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss)}')

        self.ignore_label = ignore_label
        self.align_corners = align_corners
        self.num_classes = num_classes

        self.dla_up = FusionUP(in_channels,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            align_corners=align_corners
                            )
        self.fuse_finals = fuse_finals
        if not fuse_finals:
            self.fc = nn.Sequential(
                nn.Dropout2d(dropout),
                nn.Conv2d(in_channels[0], num_classes, kernel_size=1,
                          stride=1, padding=0, bias=True))
        else:
            self.fc = nn.ModuleList()
            for i in range(len(in_channels) - 2, -1, -1):
                self.fc.append(
                    nn.Sequential(
                        nn.Dropout2d(dropout),
                        nn.Conv2d(in_channels[i], num_classes, kernel_size=1,
                                  stride=1, padding=0, bias=True)))
            self.fc.append(nn.Sequential(
                        nn.Dropout2d(dropout),
                        nn.Conv2d(in_channels[0], num_classes, kernel_size=1,
                                  stride=1, padding=0, bias=True)))

            self.final_conv = nn.Conv2d(in_channels=len(self.fc) * num_classes, out_channels=num_classes,
                                        kernel_size=1, bias=True)

        up_factor = 2
        self.up = nn.Upsample(scale_factor=up_factor, mode="bilinear", align_corners=self.align_corners)

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.init_weights()

    def forward(self, x):

        out_puts = self.dla_up(x, self.fuse_finals)
        if self.fuse_finals:
            out_puts = [f(out) for f, out in zip(self.fc, out_puts)]
            input_size = out_puts[-1].shape[-2:]
            seg_logits = [resize(input=output,
                                 size=input_size,
                                 mode='bilinear',
                                 align_corners=self.align_corners) for output in out_puts]
            final_concat = torch.cat(seg_logits, dim=1)
            final_logits = self.final_conv(final_concat)
        else:
            final_logits = self.fc(out_puts)

        return final_logits


    def init_weights(self):

        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(-math.log((1-0.01) / 0.01))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None

        loss = dict()
        seg_label = seg_label.squeeze(1)
        input_size = seg_label.shape[1:]
        seg_logit = resize(input=seg_logit,
                           size=input_size,
                           mode='bilinear',
                           align_corners=self.align_corners)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            loss[loss_decode.loss_name] = loss_decode(
                 seg_logit,
                 seg_label,
                 weight=seg_weight,
                 ignore_label=self.ignore_label)
        return loss

    def forward_train(self, inputs, gt_semantic_seg, **kwargs):
        """Forward function for training.

        Parameters
        ----------
        inputs : list[Tensor]
            List of multi-level img features.
        gt_semantic_seg : Tensor
            Semantic segmentation masks
            used if the architecture supports semantic segmentation task.

        Returns
        -------
        dict[str, Tensor]
            a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses, seg_logits

    def forward_infer(self, inputs, **kwargs):
        """Forward function for testing.

        Parameters
        ----------
        inputs : list[Tensor]
            List of multi-level img features.

        Returns
        -------
        Tensor
            Output segmentation map.
        """
        return self.forward(inputs)

