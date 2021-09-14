import torch
import torch.nn as nn
import torch.nn.functional as F
from base.cnn import ConvModule
from base.cnn.components.blocks.comm_blocks import C3
from networks.det.models.builder import NECKS


@NECKS.register_module()
class CSP_PAN(nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 use_trans_conv=False,
                 start_level=1,
                 end_level=-1,
                 **kwargs):

        super(CSP_PAN, self).__init__()

        assert start_level > 0, "start level must larger than 0"
        if end_level == -1:
            self.backbone_end_level = len(input_channels)
            assert start_level<=self.backbone_end_level, "start_level must smaller or equal than end_level"
        else:
            self.backbone_end_level = end_level
            assert end_level<=len(input_channels), "start_level must smaller or equal output_channels"

        self.start_level = start_level

        self.n = len(input_channels) - len(output_channels)
        assert self.n >= 0, ' number of input channels from backbone should larger than number of levles'

        self.output_channels = output_channels

        self.use_trans_conv = use_trans_conv

        for i in range(1, len(output_channels)):
            in_channel, out_channel = input_channels[-i], input_channels[-i-1]

            # up conv
            self.add_module(f'conv_up{i}',ConvModule(in_channel,out_channel,
                                                     kernel_size=1, stride=1, padding=0, **kwargs))
            self.add_module(f'C3_up{i}',C3(out_channel+out_channel, out_channel, shortcut=False, **kwargs))
            if self.use_trans_conv:
                self.add_module(f'trans_conv{i}' , nn.ConvTranspose2d(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    kernel_size=4,
                    stride=2, padding=2,
                    output_padding=0,
                    bias=False))

            # down conv
            self.add_module(f'conv_down{i}',ConvModule(out_channel, out_channel,
                                                       kernel_size=3, stride=2, padding=1, **kwargs))
            self.add_module(f'C3_down{i}',C3(out_channel+out_channel, in_channel, shortcut=False, **kwargs))

        self.out_convs = nn.ModuleList(ConvModule(ic, oc, kernel_size=1, stride=1, padding=0, **kwargs)
                                       for ic, oc in zip(input_channels[self.start_level-1:self.backbone_end_level],
                                                         output_channels[self.start_level-1:self.backbone_end_level]))

    def forward(self, features):

        temps = []
        outs= []
        num_level = len(features)
        # up sample
        for i in range(1,num_level):
            input = features[-i] if i==1 else tmp
            tmp = getattr(self, f'conv_up{i}')(input)
            temps.append(tmp)
            if self.use_trans_conv:
                tmp = getattr(self,f'trans_conv{i}')(tmp)
            else:
                tmp = F.interpolate(tmp, scale_factor=2, mode='nearest')

            tmp = torch.cat([tmp, features[-i-1]], dim=1)
            tmp = getattr(self, f'C3_up{i}')(tmp)

        if self.n == 0 and self.start_level==1:
            outs.append(tmp)

        # down sample
        for i, x in enumerate(reversed(temps)):
            index = len(temps) - i
            tmp = getattr(self, f'conv_down{index}')(tmp)
            tmp = torch.cat([tmp, x], dim=1)
            tmp = getattr(self, f'C3_down{index}')(tmp)
            if i < self.backbone_end_level-1 and i >= self.start_level-2:
                outs.append(tmp)

        outs = [out_conv(out) for out, out_conv in zip(outs, self.out_convs)]

        return outs

    def init_weights(self, pretrained=None):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True