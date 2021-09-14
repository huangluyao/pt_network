import math
import torch.nn as nn
from ..components.blocks import Focus
from ..components.pruned import C3Pruned, SPPPruned
from ..components.conv_module import ConvModule
from .builder import BACKBONES


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


@BACKBONES.register_module()
class PrunedSmartBackbone(nn.Module):

    def __init__(self, in_channels,
                 mask_bn_dict,
                 out_levels=[3, 4, 5],
                 block_numbers=[3, 9, 9, 3],
                 strides=[2, 2, 2, 2],
                 use_spp=True,
                 depth_multiple=1,
                 num_classes=None,
                 **kwargs
                 ):
        super(PrunedSmartBackbone, self).__init__()

        assert len(block_numbers) > 3, "len of block numbers should be list and greater than 3"


        if 'input_size' in kwargs:
            kwargs.pop('input_size')
        self.num_classes = num_classes
        self.out_levels = out_levels
        self.use_spp = use_spp
        self.out_level_channel_map = dict()
        self.former_to_map = {}

        # p1  2 samples

        self.focus = Focus(in_channels, int(mask_bn_dict["backbone.focus.conv.bn"].sum()),
                           kernel_size=3, stride=1, padding=1,
                           **kwargs)
        input_channel = int(mask_bn_dict["backbone.focus.conv.bn"].sum())
        self.former_channel_name = "backbone.focus.conv.bn"
        # p2 - p5
        self.stage_names = []
        for i in range(len(block_numbers)):

            n = max(round(block_numbers[i] * depth_multiple), 1) if block_numbers[i] > 1 else block_numbers[
                i]  # depth gain

            if use_spp and i + 1 == 4:
                self.stage_names.append('spp')
                named_m_base = "backbone.spp"
                args_conv, args_spp, args_C3 = self._get_SPP_args(mask_bn_dict, named_m_base, input_channel,
                                                                  self.former_channel_name,
                                                                  n, strides[i])
                self.add_module('spp',
                                nn.Sequential(
                                    ConvModule(*args_conv, **kwargs),
                                    SPPPruned(*args_spp, **kwargs),
                                    C3Pruned(*args_C3, shortcut=False, **kwargs)))

                self.out_level_channel_map[self.former_channel_name]=args_C3[3]

            else:
                name = 'stage%d' % (i + 1)
                named_m_base = "backbone.stage%d" % (i + 1)
                args_C3, args_conv = self._get_C3_args(mask_bn_dict, named_m_base,
                                                       input_channel,
                                                       self.former_channel_name,
                                                       n, strides[i])
                self.stage_names.append(name)
                self.add_module(name,
                                nn.Sequential(
                                    ConvModule(*args_conv, **kwargs),
                                    C3Pruned(*args_C3, **kwargs), ))

                input_channel = args_C3[3]
                self.out_level_channel_map[self.former_channel_name]=args_C3[3]

            if num_classes is not None:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(self.stage_channels[-1], num_classes)

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

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = x.contiguous().view(-1, self.stage_channels[-1])
            x = self.fc(x)
            return x

        return outputs if len(outputs) > 1 else outputs[0]

    def _get_C3_args(self, mask_bn_dict, named_m_base,
                     input_channel, former_channel_name,
                     n, stride):

        args_conv = list()
        named_m_conv_bn = named_m_base + ".0.bn"
        conv_out = int(mask_bn_dict[named_m_conv_bn].sum())
        args_conv.append(input_channel)
        args_conv.append(conv_out)
        args_conv.append(3)     #kenerl
        args_conv.append(stride)
        args_conv.append(1)     # padding
        self.former_to_map[named_m_conv_bn] = former_channel_name

        named_m_cv1_bn = named_m_base + ".1.cv1.bn"
        named_m_cv2_bn = named_m_base + ".1.cv2.bn"
        named_m_cv3_bn = named_m_base + ".1.cv3.bn"

        c3_cv1out = int(mask_bn_dict[named_m_cv1_bn].sum())
        c3_cv2out = int(mask_bn_dict[named_m_cv2_bn].sum())
        c3_cv3out = int(mask_bn_dict[named_m_cv3_bn].sum())
        args_C3 = [conv_out, c3_cv1out, c3_cv2out, c3_cv3out]
        self.former_to_map[named_m_cv1_bn] = named_m_conv_bn
        self.former_to_map[named_m_cv2_bn] = named_m_conv_bn

        bottle_args = []
        chin = [c3_cv1out]

        for p in range(n):
            named_m_bottle_cv1_bn = named_m_base + ".1.m.{}.cv1.bn".format(p)
            named_m_bottle_cv2_bn = named_m_base + ".1.m.{}.cv2.bn".format(p)
            bottle_cv1in = chin[-1]
            bottle_cv1out = int(mask_bn_dict[named_m_bottle_cv1_bn].sum())
            bottle_cv2out = int(mask_bn_dict[named_m_bottle_cv2_bn].sum())
            chin.append(bottle_cv2out)
            bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])

            self.former_to_map[named_m_bottle_cv1_bn] = named_m_cv1_bn
            self.former_to_map[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn


        args_C3.append(bottle_args)
        args_C3.append(n)

        self.former_to_map[named_m_cv3_bn] = [named_m_bottle_cv2_bn, named_m_cv2_bn]
        self.former_channel_name = named_m_cv3_bn
        return args_C3, args_conv

    def _get_SPP_args(self, mask_bn_dict, named_m_base,
                      input_channel, former_channel_name,
                      n, stride):

        args_conv = list()
        named_m_conv_bn = named_m_base + ".0.bn"
        conv_out = int(mask_bn_dict[named_m_conv_bn].sum())
        args_conv.append(input_channel)
        args_conv.append(conv_out)
        args_conv.append(3)     #kenerl
        args_conv.append(stride)
        args_conv.append(1)     # padding
        self.former_to_map[named_m_conv_bn] = former_channel_name

        spp_m_cv1_bn = named_m_base + ".1.cv1.bn"
        spp_m_cv2_bn = named_m_base + ".1.cv2.bn"
        cv1in = conv_out
        ssp_cv1out = int(mask_bn_dict[spp_m_cv1_bn].sum())
        ssp_cv2out = int(mask_bn_dict[spp_m_cv2_bn].sum())
        args_spp  = [cv1in, ssp_cv1out, ssp_cv2out]
        self.former_to_map[spp_m_cv1_bn] = named_m_conv_bn
        self.former_to_map[spp_m_cv2_bn] = [spp_m_cv1_bn] * 4

        named_m_cv1_bn = named_m_base + ".2.cv1.bn"
        named_m_cv2_bn = named_m_base + ".2.cv2.bn"
        named_m_cv3_bn = named_m_base + ".2.cv3.bn"
        c3_cv1out = int(mask_bn_dict[named_m_cv1_bn].sum())
        c3_cv2out = int(mask_bn_dict[named_m_cv2_bn].sum())
        c3_cv3out = int(mask_bn_dict[named_m_cv3_bn].sum())
        args_C3 = [ssp_cv2out, c3_cv1out, c3_cv2out, c3_cv3out]

        self.former_to_map[named_m_cv1_bn] = spp_m_cv2_bn
        self.former_to_map[named_m_cv2_bn] = spp_m_cv2_bn

        bottle_args = []
        chin = [c3_cv1out]

        for p in range(n):
            named_m_bottle_cv1_bn = named_m_base + ".2.m.0.cv1.bn"
            named_m_bottle_cv2_bn = named_m_base + ".2.m.0.cv2.bn"
            bottle_cv1in = chin[-1]
            bottle_cv1out = int(mask_bn_dict[named_m_bottle_cv1_bn].sum())
            bottle_cv2out = int(mask_bn_dict[named_m_bottle_cv2_bn].sum())
            chin.append(bottle_cv2out)
            bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
            self.former_to_map[named_m_bottle_cv1_bn] = named_m_cv1_bn
            self.former_to_map[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn

        args_C3.append(bottle_args)
        args_C3.append(n)

        self.former_to_map[named_m_cv3_bn] = [named_m_bottle_cv2_bn, named_m_cv2_bn]
        self.former_channel_name = named_m_cv3_bn

        return args_conv, args_spp, args_C3
