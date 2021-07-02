import math
import torch
import torch.nn as nn

from .builder import BACKBONES
from ..components import build_activation_layer, build_norm_layer, _BatchNorm
from ...utils import load_checkpoint
from ..utils import constant_init, kaiming_init
from ...utils import get_logger
from .utils.split_attention import SplAtConv2d

__all__ = ['ResNeSt']


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 dropblock_prob=0.0, last_gamma=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, group_width)[1]
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_cfg=norm_cfg,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = build_norm_layer(norm_cfg, group_width)[1]
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = build_norm_layer(norm_cfg, group_width)[1]

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = build_norm_layer(norm_cfg, planes * 4)[1]

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet Variants

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    num_classes : int
        Number of classification classes.
    out_levels : Sequence[int]
        Output from which levels.
        If only one level is specified, a single tensor (feature map) is returned,
        otherwise multiple levels are specified, a tuple of tensors will be returned.
        Default: ``(5, )``, means return the 1/32x feature map.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_cfg : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).

    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    def __init__(self, block, layers, radix=1, groups=1, bottleneck_width=64,
                 dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0, last_gamma=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 in_channels=3,
                 num_classes=None,
                 base_channels=(64, 128, 256, 512),
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_levels=(1, 2, 3, 4, 5),
                 multi_grid=None,
                 contract_dilation=False,
                 frozen_stages=-1,
                 norm_eval=False):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        self.num_classes = num_classes
        self.out_levels = out_levels

        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            from rfconv import RFConv2d
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(in_channels, stem_width, kernel_size=3, stride=2, padding=1, bias=False, **conv_kwargs),
                build_norm_layer(self.norm_cfg, stem_width)[1],
                build_activation_layer(self.act_cfg),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
                build_norm_layer(self.norm_cfg, stem_width)[1],
                build_activation_layer(self.act_cfg),
                conv_layer(stem_width, stem_width*2, kernel_size=3, stride=1, padding=1, bias=False, **conv_kwargs),
            )
        else:
            self.conv1 = conv_layer(in_channels, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, **conv_kwargs)
        self.bn1 = build_norm_layer(self.norm_cfg, self.inplanes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_names = []
        for i in range(4):
            if i == 0:
                is_first = False
            else:
                is_first = True
            res_layer = self._make_layer(block=block,
                                         planes=base_channels[i],
                                         blocks=layers[i],
                                         stride=strides[i],
                                         dilation=dilations[i],
                                         norm_cfg=norm_cfg,
                                         is_first=is_first,
                                         multi_grid=multi_grid if i == 3 else None,
                                         contract_dilation=contract_dilation)
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, res_layer)
            self.layer_names.append(layer_name)

        if num_classes is not None:
            self.avgpool = GlobalAvgPool2d()
            self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._freeze_stages()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    norm_cfg=None, dropblock_prob=0.0, is_first=True,
                    multi_grid=None, contract_dilation=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(build_norm_layer(self.norm_cfg, planes * block.expansion)[1])
            downsample = nn.Sequential(*down_layers)

        if multi_grid is None:
            if dilation > 1 and contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            radix=self.radix, cardinality=self.cardinality,
                            bottleneck_width=self.bottleneck_width,
                            avd=self.avd, avd_first=self.avd_first,
                            dilation=first_dilation, is_first=is_first,
                            rectified_conv=self.rectified_conv,
                            rectify_avg=self.rectify_avg,
                            norm_cfg=norm_cfg, dropblock_prob=dropblock_prob,
                            last_gamma=self.last_gamma))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation if multi_grid is None else multi_grid[i],
                                rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_cfg=norm_cfg, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
            self._initialize_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        if 0 in self.out_levels:
            outs.append(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if 1 in self.out_levels:
            outs.append(x)
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.layer_names):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i+2 in self.out_levels:
                outs.append(x)

        if self.num_classes is not None:
            x = self.avgpool(x)
            if self.drop is not None:
                x = self.drop(x)
            x = self.fc(x)
            return x
        else:
            if len(outs) == 1:
                return outs[0]
            else:
                return tuple(outs)


@BACKBONES.register_module()
class ResNeSt(ResNet):
    """ResNeSt

    Examples
    --------
    >>> from base.cnn import ResNeSt

    >>> resnest14 = ResNeSt(
    >>>     layers=[1, 1, 1, 1],
    >>>     radix=2,
    >>>     groups=1,
    >>>     bottleneck_width=64,
    >>>     deep_stem=True,
    >>>     stem_width=32,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=False)

    >>> resnest26 = ResNeSt(
    >>>     layers=[2, 2, 2, 2],
    >>>     radix=2,
    >>>     groups=1,
    >>>     bottleneck_width=64,
    >>>     deep_stem=True,
    >>>     stem_width=32,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=False)

    >>> resnest50 = ResNeSt(
    >>>     layers=[3, 4, 6, 3],
    >>>     radix=2,
    >>>     groups=1,
    >>>     bottleneck_width=64,
    >>>     deep_stem=True,
    >>>     stem_width=32,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=False)

    >>> resnest50_fast_4s2x40d = ResNeSt(
    >>>     layers=[3, 4, 6, 3],
    >>>     radix=4,
    >>>     groups=2,
    >>>     bottleneck_width=40,
    >>>     deep_stem=True,
    >>>     stem_width=32,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=True)

    >>> resnest101 = ResNeSt(
    >>>     layers=[3, 4, 23, 3],
    >>>     radix=2,
    >>>     groups=1,
    >>>     bottleneck_width=64,
    >>>     deep_stem=True,
    >>>     stem_width=64,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=False)

    >>> resnest200 = ResNeSt(
    >>>     layers=[3, 24, 36, 3],
    >>>     radix=2,
    >>>     groups=1,
    >>>     bottleneck_width=64,
    >>>     deep_stem=True,
    >>>     stem_width=64,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=False)

    >>> resnest269 = ResNeSt(
    >>>     layers=[3, 30, 48, 8],
    >>>     radix=2,
    >>>     groups=1,
    >>>     bottleneck_width=64,
    >>>     deep_stem=True,
    >>>     stem_width=64,
    >>>     avg_down=True,
    >>>     avd=True,
    >>>     avd_first=False)
    """

    def __init__(self, **kwargs):
        super(ResNeSt, self).__init__(
            block=Bottleneck, **kwargs)
