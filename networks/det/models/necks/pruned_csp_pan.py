import torch
import torch.nn as nn
import torch.nn.functional as F
from base.cnn import ConvModule
from base.cnn.components.pruned.block import C3Pruned
from networks.det.models.builder import NECKS


@NECKS.register_module()
class PrunedCSP_PAN(nn.Module):

    def __init__(self,
                 input_channel_maps,
                 output_channels,
                 mask_bn_dict,
                 use_trans_conv=False,
                 start_level=1,
                 end_level=-1,
                 **kwargs):

        super(PrunedCSP_PAN, self).__init__()

        assert start_level > 0, "start level must larger than 0"
        if end_level == -1:
            self.backbone_end_level = len(input_channel_maps)
            assert start_level<=self.backbone_end_level, "start_level must smaller or equal than end_level"
        else:
            self.backbone_end_level = end_level
            assert end_level<=len(input_channel_maps), "start_level must smaller or equal output_channels"

        self.start_level = start_level

        self.n = len(input_channel_maps) - len(output_channels)
        assert self.n >= 0, ' number of input channels from backbone should larger than number of levles'

        self.output_channels = [int(mask_bn_dict["neck.out_convs.%d.bn" % i].sum()) for i in range(len(output_channels))]
        self.input_channels = [value for value in input_channel_maps.values()][-len(output_channels):]
        input_channel_names = [key for key in input_channel_maps.keys()][-len(output_channels):]
        self.use_trans_conv = use_trans_conv
        self.former_to_map = {}

        out_channel = 0
        c3_inputs = []
        c3_inputs_name = []
        for i in range(1, len(output_channels)):
            if i ==1:
                in_channel, out_channel = self.input_channels[-i], self.input_channels[-i-1]
                self.input_channel_name = input_channel_names[-i]
            else:
                in_channel = out_channel
                out_channel = self.input_channels[-i-1]

            # up convi
            conv_up_args = self._get_Conv_args(mask_bn_dict, i , in_channel, self.input_channel_name, is_up=True)
            self.add_module(f'conv_up{i}',ConvModule(*conv_up_args, **kwargs))
            c3_inputs.append(conv_up_args[1])
            c3_inputs_name.append(self.input_channel_name)
            c3_up_args = self._get_C3_args(mask_bn_dict, i, conv_up_args[1]+out_channel,
                                           [self.input_channel_name, input_channel_names[-i-1]],
                                           is_up=True)
            self.add_module(f'C3_up{i}',C3Pruned(*c3_up_args, shortcut=False, **kwargs))

            if self.use_trans_conv:
                self.add_module(f'trans_conv{i}' , nn.ConvTranspose2d(
                    in_channels=c3_up_args[3],
                    out_channels=c3_up_args[3],
                    kernel_size=4,
                    stride=2, padding=2,
                    output_padding=0,
                    bias=False))

            out_channel = c3_up_args[3]

        c3_outputs = [out_channel]
        c3_outputs_name = [self.input_channel_name]

        for i in reversed(range(1, len(output_channels))):
            # down conv
            conv_down_args = self._get_Conv_args(mask_bn_dict, i, c3_outputs[-1],
                                                 self.input_channel_name,
                                                 is_up=False,
                                                 kernel_size=3, stride=2, padding=1)
            self.add_module(f'conv_down{i}',ConvModule(*conv_down_args, **kwargs))

            # down c3
            c3_down_args = self._get_C3_args(mask_bn_dict, i, conv_down_args[1]+c3_inputs[i-1],
                                             [self.input_channel_name, c3_inputs_name[i-1]],
                                             False)
            self.add_module(f'C3_down{i}',C3Pruned(*c3_down_args, shortcut=False, **kwargs))
            c3_outputs.append(c3_down_args[3])
            c3_outputs_name.append(self.input_channel_name)



        self.out_convs = nn.ModuleList(ConvModule(ic, oc, kernel_size=1, stride=1, padding=0, **kwargs)
                                       for ic, oc in zip(c3_outputs[self.start_level-1:self.backbone_end_level],
                                                         self.output_channels[self.start_level-1:self.backbone_end_level]))

        for ic in range(len(self.out_convs)):
            model_name = "neck.out_convs.%d.bn" % ic
            self.former_to_map[model_name] = c3_outputs_name[ic]

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

        if self.start_level==1:
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

    def _get_Conv_args(self, mask_bn_dict, i,
                          input_channel,
                          input_channel_name,
                          is_up,
                          kernel_size=1, stride=1, padding=0):

        args_conv_up = list()
        named_m_conv_bn = "neck.conv_up%d.bn" % (i) if is_up else "neck.conv_down%d.bn" % (i)
        conv_out = int(mask_bn_dict[named_m_conv_bn].sum())
        args_conv_up.append(input_channel)
        args_conv_up.append(conv_out)
        args_conv_up.append(kernel_size)     #kenerl
        args_conv_up.append(stride)
        args_conv_up.append(padding)     # padding
        self.former_to_map[named_m_conv_bn] = input_channel_name
        self.input_channel_name = named_m_conv_bn

        return args_conv_up


    def _get_C3_args(self, mask_bn_dict, i , input_channel,
                     input_channel_name,
                     is_up,  n=1):
        named_m_base = "neck.C3_up%d" % (i) if is_up else "neck.C3_down%d" % (i)
        named_m_cv1_bn = named_m_base + ".cv1.bn"
        named_m_cv2_bn = named_m_base + ".cv2.bn"
        named_m_cv3_bn = named_m_base + ".cv3.bn"

        cv1out = int(mask_bn_dict[named_m_cv1_bn].sum())
        cv2out = int(mask_bn_dict[named_m_cv2_bn].sum())
        cv3out = int(mask_bn_dict[named_m_cv3_bn].sum())
        args_C3 = [input_channel, cv1out, cv2out, cv3out]
        self.former_to_map[named_m_cv1_bn] = input_channel_name
        self.former_to_map[named_m_cv2_bn] = input_channel_name

        bottle_args = []
        chin = [cv1out]
        for p in range(n):
            named_m_bottle_cv1_bn = named_m_base + ".m.{}.cv1.bn".format(p)
            named_m_bottle_cv2_bn = named_m_base + ".m.{}.cv2.bn".format(p)
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
        self.input_channel_name = named_m_cv3_bn

        return args_C3