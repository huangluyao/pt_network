from collections import OrderedDict
import torch.nn as nn
import torch.utils.checkpoint as cp

from .builder import BACKBONES
from .utils.res_layer import ResLayer, BasicBlockV2, BottleneckV2
from ..components import (build_activation_layer, build_conv_layer,
                          build_norm_layer, ConvModule, _BatchNorm)
from ...utils import load_checkpoint
from ..utils import constant_init, kaiming_init
from ...utils import get_logger


@BACKBONES.register_module()
class ResNetV2(nn.Module):
    """ResNetV2 backbone.
    """

    stage_channels = {
        18:  (64, 128, 256, 512),
        34:  (64, 128, 256, 512),
        50:  (256, 512, 1024, 2048),
        101: (256, 512, 1024, 2048),
        152: (256, 512, 1024, 2048)
    }

    arch_settings = {
        18: (BasicBlockV2, (2, 2, 2, 2)),
        34: (BasicBlockV2, (3, 4, 6, 3)),
        50: (BottleneckV2, (3, 4, 6, 3)),
        101: (BottleneckV2, (3, 4, 23, 3)),
        152: (BottleneckV2, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_classes=None,
                 stage_channels=None,
                 base_channels=64,
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
        super(ResNetV2, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
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
        if stage_channels is not None:
            self.stage_channels = stage_channels
        else:
            self.stage_channels = self.stage_channels[depth]
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]

        self._make_conv1(in_channels, base_channels)

        self.res_layers = []
        self.feature_channels = []
        for i, num_blocks in enumerate(self.stage_blocks):
            layer_name = f'layer{i+1}'

            if i == 0:
                _in_channels = base_channels
            else:
                _in_channels = self.stage_channels[i-1]
            _out_channels = self.stage_channels[i]
            stride = strides[i]
            dilation = dilations[i]
            stage_multi_grid = multi_grid if i == len(
                self.stage_blocks) - 1 else None

            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                stride=stride,
                dilation=dilation,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation)

            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            self.feature_channels.append(_out_channels)

        self._make_last(self.feature_channels[-1])

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.feature_channels[-1], num_classes)

        self._freeze_stages()

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_conv1(self, in_channels, out_channels):
        """Make conv1 layer for ResNetV2."""
        self.conv1 = nn.Sequential(OrderedDict([
            ('0', build_norm_layer(self.norm_cfg, in_channels, postfix='1')[1]),
            ('1', build_conv_layer(
                self.conv_cfg,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ))
        ]))

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, out_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = build_activation_layer(cfg=self.act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_last(self, in_channels, out_channels=None):
        self.last = nn.Sequential(OrderedDict([
            ('0', build_norm_layer(self.norm_cfg, in_channels, postfix='1')[1]),
            ('1', build_activation_layer(cfg=self.act_cfg))
        ]))

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
                    if isinstance(m, BottleneckV2):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlockV2):
                        constant_init(m.norm2, 0)
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
            if (i + 1) == self.num_stages:
                x = self.last(x)

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
        super(ResNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
