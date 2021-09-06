from collections import OrderedDict
import logging

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from .builder import BACKBONES
from ..components import ConvModule
from ..utils import constant_init, kaiming_init
from ...utils import load_checkpoint


class BasicBlock(nn.Module):
    """The basic residual block used in Darknet.

    Each BasicBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.

    In YoloV3 paper, the first convLayer has half of the number of the filters
    as much as the second convLayer. The first convLayer has filter size of 1x1
    and the second one has the filter size of 3x3.

    Parameters
    ----------
    in_channels : int
        The input channels. Must be even.
    conv_cfg : dict
        Config dict for convolution layer. Default: None.
    norm_cfg : dict
        Dictionary to construct and config norm layer.
        Default: dict(type='BN', requires_grad=True)
    act_cfg : dict
        Config dict for activation layer.
        Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(BasicBlock, self).__init__()
        assert in_channels % 2 == 0  # ensure the in_channels is even
        mid_channels = in_channels // 2
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(in_channels, mid_channels, 1, **cfg)
        self.conv2 = ConvModule(mid_channels, in_channels, 3, padding=1, **cfg)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual

        return out


@BACKBONES.register_module()
class Darknet(nn.Module):
    """Darknet backbone.

    Parameters
    ----------
    depth : int
        Depth of Darknet. Currently only support 53.
    out_levels : Sequence[int]
        Output from which levels.
        If only one level is specified, a single tensor (feature map) is returned,
        otherwise multiple levels are specified, a tuple of tensors will be returned.
        Default: ``(5, )``, means return the 1/32x feature map.
    frozen_stages : int
        Stages to be frozen (stop grad and set eval mode).
        -1 means not freezing any parameters. Default: -1.
    in_channels : int
        Input channels of the network.
    num_classes : int, default None
        Number of classes. If it is None, the network is as a backbone of a model.
    conv_cfg : dict
        Config dict for convolution layer. Default: None.
    norm_cfg : dict
        Dictionary to construct and config norm layer.
        Default: dict(type='BN', requires_grad=True)
    act_cfg : dict
        Config dict for activation layer.
        Default: dict(type='LeakyReLU', negative_slope=0.1).

    Examples
    --------
    >>> #model = Darknet19(out_levels=(3, 4, 5))
    >>> model = Darknet53(out_levels=(3, 4, 5))
    >>> model.eval()
    >>> inputs = torch.rand(1, 3, 416, 416)
    >>> outputs = self.forward(inputs)
    >>> for output in outputs:
    ...     print(output.shape)
    ...
    (1, 256, 52, 52)
    (1, 512, 26, 26)
    (1, 1024, 13, 13)
    """

    arch_settings = {
        53: ((1, 2, 8, 8, 4), ((32, 64), (64, 128), (128, 256), (256, 512),
                               (512, 1024)))
    }

    def __init__(self,
                 depth=53,
                 out_levels=(1, 2, 3, 4, 5),
                 frozen_stages=-1,
                 in_channels=3,
                 num_classes=None,
                 norm_eval=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(Darknet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for darknet')
        self.depth = depth
        self.out_levels = out_levels
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.blocks, self.channels = self.arch_settings[depth]
        self.num_classes = num_classes

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(in_channels, 32, 3, padding=1, **cfg)

        self.layer_names = ['conv1']
        for i, n_blocks in enumerate(self.blocks):
            layer_name = f'stage{i + 1}'
            in_c, out_c = self.channels[i]
            self.add_module(
                layer_name,
                self.make_stage(in_c, out_c, n_blocks, **cfg))
            self.layer_names.append(layer_name)

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.logits = nn.Linear(1024, num_classes)

    def forward(self, x):
        outs = []
        if 0 in self.out_levels:
            outs.append(x)
        for i, layer_name in enumerate(self.layer_names):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_levels:
                outs.append(x)

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = x.contiguous().view(-1, 1024)
            x = self.logits(x)
            return x
        else:
            if len(outs) == 1:
                return outs[0]
            else:
                return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self, self.layer_names[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(Darknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    @staticmethod
    def make_stage(in_channels,
                   out_channels,
                   block_repeat,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN', requires_grad=True),
                   act_cfg=dict(type='LeakyReLU',
                                negative_slope=0.1)):
        """In Darknet backbone, ConvLayer is usually followed by BasicBlock. This
        function will make that. The Conv layers always have 3x3 filters with
        stride=2. The number of the filters in Conv layer is the same as the
        out channels of the BasicBlock.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        block_repeat : int
            The number of BasicBlocks.
        conv_cfg : dict
            Config dict for convolution layer. Default: None.
        norm_cfg : dict
            Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg : dict
            Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        """

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        model = nn.Sequential()
        model.add_module(
            'conv1', ConvModule(
                in_channels, out_channels, 3, stride=2, padding=1, **cfg)
        )
        for idx in range(block_repeat):
            model.add_module('block{}'.format(idx+1),
                             BasicBlock(out_channels, **cfg))
        return model


@BACKBONES.register_module()
class Darknet53(Darknet):

    def __init__(self, **kwargs):
        super(Darknet53, self).__init__(depth=53, **kwargs)


@BACKBONES.register_module()
class Darknet19(nn.Module):

    def __init__(self,
                 out_levels=(1, 2, 3, 4, 5),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 norm_eval=False,
                 in_channels=3,
                 num_classes=None):
        super(Darknet19, self).__init__()
        self.out_levels = out_levels
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.num_classes = num_classes

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(in_channels, 32, 3, padding=1, **cfg)
        self.layer_names = ['conv1']

        self.stage1 = nn.Sequential(OrderedDict([
            ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1', ConvModule(32, 64, 3, padding=1, **cfg))
        ]))
        self.layer_names.append('stage1')

        self.stage2 = nn.Sequential(OrderedDict([
            ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1', ConvModule(64, 128, 3, padding=1, **cfg)),
            ('conv2', ConvModule(128, 64, 1, padding=0, **cfg)),
            ('conv3', ConvModule(64, 128, 3, padding=1, **cfg))
        ]))
        self.layer_names.append('stage2')

        self.stage3 = nn.Sequential(OrderedDict([
            ('maxpool3', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1', ConvModule(128, 256, 3, padding=1, **cfg)),
            ('conv2', ConvModule(256, 128, 1, padding=0, **cfg)),
            ('conv3', ConvModule(128, 256, 3, padding=1, **cfg))
        ]))
        self.layer_names.append('stage3')

        self.stage4 = nn.Sequential(OrderedDict([
            ('maxpool4', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1', ConvModule(256, 512, 3, padding=1, **cfg)),
            ('conv2', ConvModule(512, 256, 1, padding=0, **cfg)),
            ('conv3', ConvModule(256, 512, 3, padding=1, **cfg)),
            ('conv4', ConvModule(512, 256, 1, padding=0, **cfg)),
            ('conv5', ConvModule(256, 512, 3, padding=1, **cfg))
        ]))
        self.layer_names.append('stage4')

        self.stage5 = nn.Sequential(OrderedDict([
            ('maxpool5', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv1', ConvModule(512, 1024, 3, padding=1, **cfg)),
            ('conv2', ConvModule(1024, 512, 1, padding=0, **cfg)),
            ('conv3', ConvModule(512, 1024, 3, padding=1, **cfg)),
            ('conv4', ConvModule(1024, 512, 1, padding=0, **cfg)),
            ('conv5', ConvModule(512, 1024, 3, padding=1, **cfg))
        ]))
        self.layer_names.append('stage5')

        if num_classes is not None:
            self.classifier = nn.Sequential(OrderedDict([
                ('logits', nn.Conv2d(1024, num_classes, 1)),
                ('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
            ]))

    def forward(self, x):
        outs = []
        if 0 in self.out_levels:
            outs.append(x)
        for i, layer_name in enumerate(self.layer_names):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_levels:
                outs.append(x)

        if self.num_classes is not None:
            x = self.classifier(x)
            x = x.contiguous().view(-1, self.num_classes)
            return x
        else:
            if len(outs) == 1:
                return outs[0]
            else:
                return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self, self.layer_names[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(Darknet19, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
