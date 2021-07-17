# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/28 下午5:38

import torch
import math
import torch.nn as nn
from ..components.blocks import Focus, C3, SPP, C3TR
from ..components.conv_module import ConvModule
from .builder import BACKBONES


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


@BACKBONES.register_module()
class DSNet(nn.Module):

    def __init__(self, in_channels,
                 stage_channels=[64, 128, 256, 512, 1024],
                 out_levels=[3, 4, 5],
                 block_numbers=[3, 9, 9, 3],
                 strides=[2, 2, 2, 2],
                 depth_multiple=1,
                 width_multiple=1,
                 use_spp=True,
                 use_transformer=False,
                 num_classes=None,
                 **kwargs
                 ):
        super(DSNet, self).__init__()

        assert len(block_numbers) > 3, "len of block numbers should be list and greater than 3"

        if 'input_size' in kwargs:
            kwargs.pop('input_size')

        self.stage_channels = [make_divisible(x * width_multiple, 8) for x in stage_channels]
        self.out_levels = out_levels
        self.use_spp = use_spp
        # p1  2 samples
        self.focus = Focus(in_channels, self.stage_channels[0],
                           kernel_size=3, stride=1, padding=1,
                           **kwargs)
        # p2 4 samples
        input_channels = self.stage_channels[0]
        # p2 - p5
        self.stage_names = []
        for i in range(len(block_numbers)):

            n = max(round(block_numbers[i] * depth_multiple), 1) if block_numbers[i] > 1 else block_numbers[
                i]  # depth gain

            output_channels = self.stage_channels[i + 1]

            if use_spp and i + 1 == 4:
                self.stage_names.append('spp')
                self.add_module('spp',
                                nn.Sequential(
                                    ConvModule(input_channels, output_channels,
                                               kernel_size=3, stride=strides[i], padding=1, **kwargs),
                                    SPP(output_channels, output_channels, **kwargs),
                                    C3(output_channels, output_channels, number=n, shortcut=False, **kwargs)
                                    if not use_transformer else C3TR(output_channels, output_channels, number=n, shortcut=False, **kwargs)
                                       ))

            else:
                name = 'stage%d' % (i + 1)
                self.stage_names.append(name)
                self.add_module(name,
                                nn.Sequential(
                                    ConvModule(input_channels, output_channels,
                                               kernel_size=3, stride=strides[i], padding=1, **kwargs),
                                    C3(output_channels, output_channels, number=n, **kwargs), ))

            input_channels = output_channels

    def init_weights(self, **kwargs):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

    def forward(self, x):
        outputs = []
        if 0 in self.out_levels:
            outputs.append(x)
        x = self.focus(x)
        if 1 in self.out_levels:
            outputs.append(x)
        for i, name in enumerate(self.stage_names):
            x = getattr(self, name)(x)
            if i + 2 in self.out_levels:
                outputs.append(x)

        return outputs if len(outputs) > 1 else outputs[0]
