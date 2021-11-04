import torch
import math
import torch.nn as nn
import numpy as np
from copy import deepcopy
from base.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, ConvModule, resize, _BatchNorm)
from ..builder import HEADS, build_loss
from ..criterions.detail_aggregate_loss import DetailAggregateLoss



def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[:, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))


class IDAUP(nn.Module):

    def __init__(self, kernel, out_dim, channels, up_factors,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 use_deconv=True,
                 ):
        super(IDAUP, self).__init__()
        self.channels = channels
        self.out_dim = out_dim

        for i, c in enumerate(channels):
            if c == out_dim:
                proj = nn.Identity()
            else:
                proj = ConvModule(c, out_dim, kernel_size=1, stride=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            f = int(up_factors[i])
            if f == 1:
                up = nn.Identity()
            else:
                if use_deconv:
                    up = nn.ConvTranspose2d(out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                                            output_padding=0, groups=out_dim, bias=False)
                    fill_up_weights(up)
                else:
                    up = nn.Upsample(scale_factor=f, mode="bilinear")

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = ConvModule(out_dim * 2, out_dim, kernel_size=kernel, stride=1,
                              padding=kernel // 2,
                              norm_cfg=norm_cfg, act_cfg=act_cfg)

            setattr(self, 'node_' + str(i), node)

        self.init_weights()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))

        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))

        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], dim=1))
            y.append(x)
        return x, y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DLAUP(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16),
                 in_channels=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 ):
        super(DLAUP, self).__init__()
        if in_channels is None:
            in_channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUP(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j], norm_cfg=norm_cfg, act_cfg=act_cfg))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers, out_list=False):
        layers = list(layers)
        assert len(layers) > 1
        if out_list:
            outs = []
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
            if out_list:
                outs.append(x)

        return outs if out_list else x


@HEADS.register_module()
class DLAHead(nn.Module):

    def __init__(self, num_classes,
                 in_channels=[16, 32, 128, 256, 512, 1024],
                 down_ratio=2,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_label=None,
                 align_corners=False,
                 use_deconv=True,
                 **kwargs):
        super(DLAHead, self).__init__()

        assert down_ratio in [2, 4, 8, 16]

        self.loss = build_loss(loss)
        self.ignore_label = ignore_label
        self.align_corners = align_corners
        self.num_classes = num_classes

        scales = [2 ** i for i in range(len(in_channels))]
        self.dla_up = DLAUP(in_channels, scales=scales,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg
                            )
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels[0], num_classes, kernel_size=1,
                      stride=1, padding=0, bias=True))
        up_factor = 2
        if use_deconv:
            up = nn.ConvTranspose2d(num_classes, num_classes, up_factor * 2,
                                    stride=up_factor, padding=up_factor // 2,
                                    output_padding=0, groups=num_classes,
                                    bias=False)

            fill_up_weights(up)
            up.weight.requires_grad = False
        else:
            up = nn.Upsample(scale_factor=up_factor, mode="bilinear")
        self.up = up

        self.init_weights()

    def forward(self, x):
        x = self.dla_up(x)
        x = self.fc(x)
        return self.up(x)

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
        loss = dict()
        seg_label = seg_label.squeeze(1)
        input_size = seg_label.shape[1:]
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
        return losses

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


@HEADS.register_module()
class DLAStrongHead(DLAHead):

    def __init__(self,
                 num_classes,
                 in_channels=[16, 32, 128, 256, 512, 1024],
                 head_width=64,
                 final_drop=0.1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 support_loss=dict(type="StructureLoss"),
                 **kwargs):
        super(DLAStrongHead, self).__init__(num_classes=num_classes,
                                            in_channels=deepcopy(in_channels),
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            **kwargs)

        self.fc = nn.ModuleList()
        for i in range(len(in_channels)-2, -1, -1):
            self.fc.append(
                nn.Conv2d(in_channels[i], num_classes, kernel_size=1,
                          stride=1, padding=0, bias=True)
            )

        self.conv_out_sp8 = nn.Sequential(
            ConvModule(in_channels[-3], head_width, kernel_size=3, stride=1, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Dropout2d(final_drop),
            nn.Conv2d(head_width, 1, kernel_size=1)
        )

        self.detail_aggregate_loss = DetailAggregateLoss()

        self.support_loss = build_loss(support_loss)
        self.init_weights()

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
        outputs = self.dla_up(inputs, True)


        outputs = [f(out) for f, out in zip(self.fc, outputs)]
        outputs[-1] =  self.up(outputs[-1])

        outputs.append(self.conv_out_sp8(inputs[-3]))
        losses = self.losses(outputs, gt_semantic_seg)
        return losses

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
        x = self.dla_up(inputs)
        x = self.fc[-1](x)
        return self.up(x)


    def losses(self, outputs, seg_label):
        """Compute segmentation loss."""
        seg_label = seg_label.squeeze(1)
        input_size = seg_label.shape[1:]
        seg_logits = [resize(input=output,
                           size=input_size,
                           mode='bilinear',
                           align_corners=self.align_corners) for output in outputs[:-1]]

        loss = {f'support_loss{i}': self.support_loss(seg_logit,
                                              seg_label,
                                              weight=None,
                                              ignore_label=self.ignore_label) for i, seg_logit in enumerate(seg_logits[:-1])}

        loss['seg_loss'] = self.loss(seg_logits[-1], seg_label)
        loss['detail_aggregate_loss'] = self.detail_aggregate_loss(outputs[-1], seg_label, outputs[-1].type())


        return loss

