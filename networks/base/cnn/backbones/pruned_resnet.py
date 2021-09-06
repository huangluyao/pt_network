from collections import OrderedDict
import torch.nn as nn
import torch.utils.checkpoint as cp

from .builder import BACKBONES
from .utils.pruned_resnet_settings import model_settings
from ..components import (build_activation_layer, build_conv_layer,
                          build_norm_layer, ConvModule, _BatchNorm)
from ...utils import load_checkpoint
from ..utils import constant_init, kaiming_init
from ...utils import get_logger


class _Bottleneck(nn.Module):
    """Bottleneck block for ResNet.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(_Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if isinstance(stride, (tuple, list)):
            self.conv1_stride = stride[0]
            self.conv2_stride = stride[1]
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, out_channels[0], postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels[1], postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels[2], postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels[0],
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            out_channels[0],
            out_channels[1],
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            out_channels[1],
            out_channels[2],
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = build_activation_layer(cfg=act_cfg)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class _ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.
    """

    def __init__(self,
                 block,
                 block_channels,
                 stride=1,
                 dilation=1,
                 downsample3x3=False,
                 avg_down=False,
                 shortcut_projection=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 multi_grid=None,
                 contract_dilation=False,
                 **kwargs):
        num_blocks = len(block_channels)
        block0_in_channels = block_channels[0]['in_channels']
        block0_out_channels = block_channels[0]['out_channels'][-1]

        self.block = block
        downsample = None
        if shortcut_projection or stride != 1 or block0_in_channels != block0_out_channels:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False))
            downsample.append(build_conv_layer(
                conv_cfg,
                block0_in_channels,
                block0_out_channels,
                kernel_size=1,
                stride=conv_stride,
                bias=False))
            norm_name, norm_module = build_norm_layer(norm_cfg, block0_out_channels, postfix='1')
            downsample.append(norm_module)
            downsample = nn.Sequential(*downsample)

        layers = []
        if multi_grid is None:
            if dilation > 1 and contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]

        if stride > 1 and downsample3x3:
            stride = (1, stride, 1)

        layers.append(
            block(
                **block_channels[0],
                stride=stride,
                dilation=first_dilation,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                **kwargs))
        for i in range(1, num_blocks):
            layers.append(
                block(
                    **block_channels[i],
                    stride=1,
                    dilation=dilation if multi_grid is None else multi_grid[i],
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(_ResLayer, self).__init__(*layers)


@BACKBONES.register_module()
class PrunedResNet(nn.Module):
    """PrunedResNet backbone.
    """

    def __init__(self,
                 model_name,
                 in_channels=3,
                 num_classes=None,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_levels=(1, 2, 3, 4, 5),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 num_stages=4,
                 frozen_stages=-1,
                 norm_eval=False,
                 multi_grid=None,
                 contract_dilation=False,
                 with_cp=False,
                 zero_init_residual=True):
        super(PrunedResNet, self).__init__()

        kwargs = model_settings[model_name]
        for key in kwargs:
            setattr(self, key, kwargs[key])

        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_levels = out_levels
        assert max(out_levels) < 6
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.multi_grid = multi_grid
        self.contract_dilation = contract_dilation
        self.zero_init_residual = zero_init_residual
        self.num_classes = num_classes
        self.block = _Bottleneck

        self._make_stem_layer(in_channels, self.stem_channels)

        self.res_layers = []
        self.feature_channels = []
        for i, num_blocks in enumerate(self.stage_blocks):
            layer_name = f'layer{i+1}'

            stride = strides[i]
            dilation = dilations[i]
            stage_multi_grid = multi_grid if i == len(
                self.stage_blocks) - 1 else None

            res_layer = self.make_res_layer(
                block=self.block,
                block_channels=self.stage_channels[layer_name],
                stride=stride,
                dilation=dilation,
                downsample3x3=self.downsample3x3,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation)

            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            self.feature_channels.append(self.stage_channels[layer_name][-1]['out_channels'][-1])

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.feature_channels[-1], num_classes)

        self._freeze_stages()

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return _ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, out_channels):
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.conv1 = nn.Sequential(OrderedDict([
                ('0', build_conv_layer(
                    self.conv_cfg,
                    in_channels=in_channels,
                    out_channels=out_channels[0],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False
                )),
                ('1', build_norm_layer(self.norm_cfg, out_channels[0], postfix='1')[1]),
                ('2', build_activation_layer(self.act_cfg)),
                ('3', build_conv_layer(
                    self.conv_cfg,
                    in_channels=out_channels[0],
                    out_channels=out_channels[1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )),
                ('4', build_norm_layer(self.norm_cfg, out_channels[1], postfix='2')[1]),
                ('5', build_activation_layer(self.act_cfg)),
                ('6', build_conv_layer(
                    self.conv_cfg,
                    in_channels=out_channels[1],
                    out_channels=out_channels[2],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ))
            ]))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)

        if isinstance(out_channels, (tuple, list)):
            out_channels = out_channels[-1]
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, out_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = build_activation_layer(cfg=self.act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Parameters
        ----------
        pretrained : str, optional
            Path to pre-trained weights.
            Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_logger('deepcv_base')
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, _Bottleneck):
                        constant_init(m.norm3, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        outs = []
        if 0 in self.out_levels:
            outs.append(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        if 1 in self.out_levels:
            outs.append(x)
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i+2 in self.out_levels:
                outs.append(x)

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = x.contiguous().view(-1, self.feature_channels[-1])
            x = self.fc(x)
            return x
        else:
            if len(outs) == 1:
                return outs[0]
            else:
                return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(PrunedResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
