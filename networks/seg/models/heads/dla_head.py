import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from base.cnn import (build_activation_layer, build_conv_layer, build_upsample_layer,
                      build_norm_layer, ConvModule, resize, _BatchNorm)
from ..builder import HEADS, build_loss
from ..criterions.detail_aggregate_loss import DetailAggregateLoss
from .memory_head import ASPP, FeaturesMemory
from ...specific.samplers import build_pixel_sampler

from scipy.ndimage.morphology import distance_transform_edt

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
                 act_cfg=dict(type='ReLU6', inplace=True),
                 upsample_cfg=dict(type="bilinear", scale_factor=2),
                 order=('conv', 'norm', 'act'),
                 **kwargs
                 ):
        super(IDAUP, self).__init__()
        self.channels = channels
        self.out_dim = out_dim

        for i, c in enumerate(channels):
            if c == out_dim:
                proj = nn.Identity()
            else:
                proj = ConvModule(c, out_dim, kernel_size=1, stride=1, norm_cfg=norm_cfg, act_cfg=act_cfg, order=order)
            f = int(up_factors[i])
            if f == 1:
                up = nn.Identity()
            else:
                # if use_deconv:
                #     up = nn.ConvTranspose2d(out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                #                             output_padding=0, groups=out_dim, bias=False)
                #     fill_up_weights(up)
                # else:
                #     # up = nn.Upsample(scale_factor=f, mode=up_mode, align_corners=align_corners)
                #     up = build_upsample_layer(upsample_cfg)
                if upsample_cfg.get("type") == "deconv":
                    upsample_cfg["in_channels"] = out_dim
                    upsample_cfg["out_channels"] = out_dim
                    upsample_cfg["kernel_size"] = f * 2
                    upsample_cfg["stride"] = f
                    upsample_cfg["padding"] = f // 2
                    upsample_cfg["groups"] = out_dim
                    upsample_cfg["output_padding"] = 0
                    upsample_cfg["bias"] = False
                elif  upsample_cfg.get("type") == "pixel_shuffle":
                    upsample_cfg["in_channels"] = out_dim
                    upsample_cfg["out_channels"] = out_dim
                    upsample_cfg["scale_factor"] = f
                    upsample_cfg["upsample_kernel"] = upsample_cfg.get("upsample_kernel", 3)

                up = build_upsample_layer(upsample_cfg)
                if upsample_cfg.get("type") == "deconv":
                    fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = ConvModule(out_dim * 2, out_dim, kernel_size=kernel, stride=1,
                              padding=kernel // 2,
                              norm_cfg=norm_cfg, act_cfg=act_cfg,order=order)

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
                 act_cfg=dict(type='ReLU6'),
                 upsample_cfg=dict(type="bilinear", scale_factor=2),
                 order=('conv', 'norm', 'act'),
                 **kwargs
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
                          scales[j:] // scales[j], norm_cfg=norm_cfg, act_cfg=act_cfg,
                          upsample_cfg=upsample_cfg, order=order
                          ))
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
                 dropout=0.,
                 sampler=None,
                 order=('conv', 'norm', 'act'),
                 upsample_cfg=dict(type="bilinear", scale_factor=2),
                 **kwargs):
        super(DLAHead, self).__init__()

        assert down_ratio in [2, 4, 8, 16]
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

        scales = [2 ** i for i in range(len(in_channels))]
        self.dla_up = DLAUP(in_channels, scales=scales,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            upsample_cfg=upsample_cfg,
                            order=order
                            )
        self.fc = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels[0], num_classes, kernel_size=1,
                      stride=1, padding=0, bias=True))
        up_factor = 2
        # if use_deconv:
        #     up = nn.ConvTranspose2d(num_classes, num_classes, up_factor * 2,
        #                             stride=up_factor, padding=up_factor // 2,
        #                             output_padding=0, groups=num_classes,
        #                             bias=False)
        #
        #     fill_up_weights(up)
        #     # up.weight.requires_grad = False
        # else:
        #     # up = nn.Upsample(scale_factor=f, mode=up_mode, align_corners=align_corners)
        #     up = build_upsample_layer(upsample_cfg)
        if upsample_cfg.get("type") == "deconv":
            upsample_cfg["in_channels"] = num_classes
            upsample_cfg["out_channels"] = num_classes
            upsample_cfg["kernel_size"] = up_factor * 2
            upsample_cfg["stride"] = up_factor
            upsample_cfg["padding"] = up_factor // 2
            upsample_cfg["groups"] = num_classes
            upsample_cfg["output_padding"] = 0
            upsample_cfg["bias"] = False
        elif upsample_cfg.get("type") == "pixel_shuffle":
            upsample_cfg["in_channels"] = num_classes
            upsample_cfg["out_channels"] = num_classes
            upsample_cfg["scale_factor"] = up_factor
            upsample_cfg["upsample_kernel"] = upsample_cfg.get("upsample_kernel", 3)

        self.up = build_upsample_layer(upsample_cfg)
        if upsample_cfg.get("type") == "deconv":
            fill_up_weights(self.up)

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.init_weights()

    def forward(self, x):

        x = self.dla_up(x)
        x = self.fc(x)
        return x

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
                 dropout=0.0,
                 aspp_cfg=None,
                 memory_update_cfg=None,
                 up_mode="bilinear",
                 **kwargs):
        super(DLAStrongHead, self).__init__(num_classes=num_classes,
                                            in_channels=deepcopy(in_channels),
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            dropout=dropout,
                                            **kwargs)

        if aspp_cfg is not None:

            self.memory_update_cfg = memory_update_cfg
            self.lr = 1e-3
            self.context_within_image_module = ASPP(**aspp_cfg)

            self.aspp_conv = ConvModule(in_channels[-1] *2,
                                        in_channels[-1],
                                        kernel_size=1,
                                        stride=1,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg
                                        )

            self.bottleneck = ConvModule(in_channels[-1],
                                        in_channels[-1],
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg
                                        )
            self.memory_module = FeaturesMemory(
                num_classes=num_classes,
                feats_channels=in_channels[-1],
                transform_channels=in_channels[-1],
                out_channels=in_channels[-1],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )

            self.decoder = nn.Sequential(
                ConvModule(in_channels[-1],
                           in_channels[-1],
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg
                           ),
                nn.Dropout2d(dropout),
                nn.Conv2d(in_channels[-1], self.num_classes, kernel_size=1, stride=1, padding=0))



        self.fc = nn.ModuleList()
        for i in range(len(in_channels)-2, -1, -1):
            self.fc.append(
                nn.Sequential(
                    nn.Dropout2d(dropout),
                    nn.Conv2d(in_channels[i], num_classes, kernel_size=1,
                              stride=1, padding=0, bias=True)))

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
        if hasattr(self, "context_within_image_module"):
            features = getattr(self, "context_within_image_module")(inputs[-1])
            neck_features = self.bottleneck(inputs[-1])
            preds1 = self.decoder(neck_features)
            stored_memory, memory_output = self.memory_module(neck_features, preds1, features)
            inputs[-1] = memory_output

            with torch.no_grad():
                input_size = gt_semantic_seg.shape[1:]
                self.memory_module.update(
                    features=F.interpolate(neck_features, size=input_size, mode='bilinear', align_corners=self.align_corners),
                    segmentation=gt_semantic_seg,
                    learning_rate=self.lr,
                    **self.memory_update_cfg
                )

        outputs = self.dla_up(inputs, True)
        outputs = [f(out) for f, out in zip(self.fc, outputs)]
        outputs[-1] =  self.up(outputs[-1])
        seg_logits = outputs[-1]
        outputs.append(self.conv_out_sp8(inputs[-3]))
        losses = self.losses(outputs, gt_semantic_seg)
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
        if hasattr(self, "context_within_image_module"):
            features = getattr(self, "context_within_image_module")(inputs[-1])
            neck_features = self.bottleneck(inputs[-1])
            preds1 = self.decoder(neck_features)
            stored_memory, memory_output = self.memory_module(neck_features, preds1, features)
            inputs[-1] = memory_output

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

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            loss[loss_decode.loss_name] = loss_decode(
                 seg_logits[-1],
                 seg_label,
                 weight=None,
                 ignore_label=self.ignore_label)

        # loss['seg_loss'] = self.loss(seg_logits[-1], seg_label)
        loss['detail_aggregate_loss'] = self.detail_aggregate_loss(outputs[-1], seg_label, outputs[-1].type())


        return loss


@HEADS.register_module()
class DLAConStructFinals(DLAHead):
    def __init__(self,
                 num_classes,
                 in_channels=[16, 32, 128, 256, 512, 1024],
                 dropout=0.0,
                 use_deconv=False,
                 up_mode="bilinear",
                 **kwargs):
        super(DLAConStructFinals, self).__init__(num_classes=num_classes,
                                                 in_channels=deepcopy(in_channels),
                                                 use_deconv=use_deconv,
                                                 **kwargs)

        self.fc = nn.ModuleList()
        for i in range(len(in_channels)-2, -1, -1):
            self.fc.append(
                nn.Sequential(
                    nn.Dropout2d(dropout),
                    nn.Conv2d(in_channels[i], num_classes, kernel_size=1,
                              stride=1, padding=0, bias=True)))

        self.final_conv = nn.Conv2d(in_channels=len(self.fc)* num_classes, out_channels=num_classes,
                                    kernel_size=1, bias=True)

        self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=self.align_corners)


        self.final_conv.weight.data.normal_(0, math.sqrt(2. / num_classes))
        self.final_conv.bias.data.fill_(-math.log((1 - 0.01) / 0.01))


    def forward_train(self, inputs, gt_semantic_seg, **kwargs):

        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses, seg_logits

    def forward_infer(self, inputs, **kwargs):

        return self.forward(inputs)


    def forward(self, x):
        out_puts = self.dla_up(x, True)
        conv = self.fc[-1]

        original_logits = conv(self.up(out_puts[-1]))
        input_size = original_logits.shape[-2:]
        out_puts = [f(out) for f, out in zip(self.fc[:-1], out_puts[:-1])]
        seg_logits = [resize(input=output,
                           size=input_size,
                           mode='bilinear',
                           align_corners=self.align_corners) for output in out_puts]
        seg_logits.append(original_logits)
        final_concat = torch.cat(seg_logits, dim=1)
        final_logits = self.final_conv(final_concat)

        return final_logits


