import torch
import math
import torch.nn as nn
from functools import partial
from ..components.conv_module import ConvModule
from .utils import make_divisible, load_checkpoint
from ..components.plugins import SELayer
from . import BACKBONES
import re



def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None, round_limit=0.9):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    return make_divisible(channels * multiplier, divisor, channel_min, round_limit=round_limit)


class EdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride

    Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML`
        - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

    This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and EfficientNet-V2 papers
      * MobileDet - https://arxiv.org/abs/2004.14525
      * EfficientNet-X - https://arxiv.org/abs/2102.05610
      * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
    """
    def __init__(
            self, in_channels, out_channels, exp_kernel_size=3, stride=1, dilation=1,
            noskip=False, exp_ratio=1.0, pw_kernel_size=1,
            act_cfg=dict(type="SiLU"),
            norm_cfg=dict(type="BN2d"),
            se_layer=None, drop_path_rate=0.):
        super(EdgeResidual, self).__init__()
        mid_channels = make_divisible(in_channels * exp_ratio)

        self.has_residual = (in_channels == out_channels and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate

        self.conv_exp = ConvModule(in_channels, mid_channels, exp_kernel_size, stride,
                                   exp_kernel_size // 2,dilation,norm_cfg=norm_cfg, act_cfg=act_cfg
                                   )
        self.se = se_layer(mid_channels) if se_layer else nn.Identity()

        self.conv_pwl = ConvModule(mid_channels, out_channels, pw_kernel_size, padding=pw_kernel_size//2, norm_cfg=norm_cfg,
                                   act_cfg=None
                                   )

    def forward(self, x):
        shortcut = x
        # Expansion Convolution
        x = self.conv_exp(x)
        # Squeeze-and-excitation
        x = self.se(x)
        # Point-wise linear projection
        x = self.conv_pwl(x)
        if self.has_residual:
            x += shortcut
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel_size=3, stride=1,
                 dilation=1, noskip=False, exp_ratio=1.0, exp_kernel_size=1,
                 pw_kernel_size=1, act_cfg=dict(type="SiLU"), norm_cfg=dict(type="BN2d"),
                 se_layer=None):

        super(InvertedResidual, self).__init__()
        mid_channels = make_divisible(in_channels * exp_ratio)
        self.has_residual = (in_channels==out_channels and stride ==1) and not noskip
        # Point-wise expansion
        self.conv_pw = ConvModule(in_channels, mid_channels, exp_kernel_size, padding= exp_kernel_size//2,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg
                                  )
        # Depth-wise convolution
        self.conv_dw = ConvModule(mid_channels, mid_channels, dw_kernel_size,stride, padding=dw_kernel_size//2,
                                  dilation=dilation, groups=mid_channels, norm_cfg=norm_cfg, act_cfg=act_cfg
                                  )

        # Squeeze-and-excitation
        self.se = se_layer(mid_channels) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = ConvModule(mid_channels, out_channels, pw_kernel_size, padding=pw_kernel_size//2, norm_cfg=norm_cfg,
                                   act_cfg=None
                                   )

    def forward(self, x):
        shortcut = x
        # Point-wise expansion
        x = self.conv_pw(x)
        # Depth-wise convolution
        x = self.conv_dw(x)
        # Squeeze-and-excitation
        x = self.se(x)
        # Point-wise linear projection
        x = self.conv_pwl(x)
        if self.has_residual:
            x += shortcut
        return x


class EfficientNet(nn.Module):

    def __init__(self,
                 block_args,
                 in_channels=3,
                 num_features=1280,
                 stem_size=32,
                 drop_rate=0.0,
                 fix_stem=False,
                 num_classes=None,
                 act_cfg=dict(type="SiLU"),
                 norm_cfg = dict(type= "BN2d"),
                 round_chs_fn = round_channels,
                 se_from_exp=False,
                 out_levels=[3, 4, 5],
                 pretrained=None,
                 **kwargs):

        super(EfficientNet, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self.round_chs_fn = round_chs_fn
        self.se_from_exp = se_from_exp
        self.out_levels = out_levels
        if not fix_stem:
            stem_size = make_divisible(stem_size)

        self.conv_stem = ConvModule(in_channels, stem_size, kernel_size=3, padding=1, stride=2,
                                    act_cfg=act_cfg, norm_cfg=norm_cfg)

        self.build_block(stem_size, block_args, act_cfg, norm_cfg)

        self.init_weights(pretrained)


    def build_block(self, in_chs, model_block_args, act_cfg, norm_cfg):
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        self.stage_names = []
        stages = []
        add_stages = False
        self.features = []
        self.in_chs = in_chs
        if model_block_args[0][0]['stride'] > 1:
            # if the first block starts with a stride, we need to extract first level feat from stem
            feature_info = dict(
                module='act1', num_chs=in_chs, stage=0, reduction=current_stride,
                hook_type='forward' if self.feature_location != 'bottleneck' else '')
            self.features.append(feature_info)

        # outer list of block_args defines the stacks
        for stack_idx, stack_args in enumerate(model_block_args):
            assert isinstance(stack_args, list)
            blocks = []
            for block_idx, block_args in enumerate(stack_args):
                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:   # only the first block in any stack can have a stride > 1
                    block_args['stride'] = 1

                next_dilation = current_dilation

                # add stages
                if block_args['stride'] > 1:
                    add_stages = True

                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation

                # create the block
                block = self._make_block(block_args, total_block_idx, total_block_count,
                                         act_cfg=act_cfg, norm_cfg=norm_cfg)
                # print(block)
                blocks.append(block)

                total_block_idx += 1  # incr global block idx (across all stacks)

            stages += blocks
            if add_stages:
                name = 'stage%d' % (len(self.stage_names) + 1)
                self.stage_names.append(name)
                self.add_module(name, nn.Sequential(*stages))
                stages = []
                add_stages = False

        if self.num_classes is not None:
            in_channel = block_args["out_channels"]
            self.conv_head = ConvModule(in_channel, 1024, kernel_size=1, padding=0, norm_cfg=norm_cfg,
                                        act_cfg=act_cfg)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten(1)
            self.fc = nn.fc = nn.Linear(1024, self.num_classes, bias=True)


    def forward(self, x):
        outputs = []
        if 0 in self.out_levels:
            outputs.append(x)
        x = self.conv_stem(x)
        if 1 in self.out_levels:
            outputs.append(x)

        for i, name in enumerate(self.stage_names):
            x = getattr(self, name)(x)
            if i + 2 in self.out_levels:
                outputs.append(x)

        if self.num_classes is not None:
            x = self.conv_head(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x

        return outputs if len(outputs) > 1 else outputs[0]


    def _make_block(self, ba, block_idx, block_count, act_cfg, norm_cfg):
        bt = ba.pop('block_type')
        ba['in_channels'] = self.in_chs
        ba['out_channels'] = self.round_chs_fn(ba['out_channels'])
        ba['act_cfg'] = act_cfg
        ba['norm_cfg'] = norm_cfg

        if bt != 'cn':
            se_ratio = ba.pop('se_ratio')
            if se_ratio:
                if not self.se_from_exp:
                    # adjust se_ratio by expansion ratio if calculating se channels from block input
                    se_ratio /= ba.get('exp_ratio', 1.0)
                ba['se_layer'] = partial(SELayer, ratio=1/se_ratio)
        if bt == 'ir':
            block = InvertedResidual(**ba)
        elif bt == 'er':
            block = EdgeResidual(**ba)
        elif bt == 'cn':
            ba["padding"] = ((ba["stride"] - 1) + ba["dilation"] * (ba["kernel_size"] - 1)) // 2
            block = ConvModule(**ba)
        self.in_chs = ba['out_channels']
        return block

    def init_weights(self, pretrained=None):
        """ Weight initialization as per Tensorflow official implementations.

        Args:
            m (nn.Module): module to init
            n (str): module name
            fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/ group convs

        Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
        * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
        * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)  # fan-out
                fan_in = 0
                if 'routing_fn' in n:
                    fan_in = m.weight.size(1)
                init_range = 1.0 / math.sqrt(fan_in + fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

        if pretrained is not None:
            load_checkpoint(self, pretrained, strict=False)




def decode_arch_def(arch_def, depth_multiplier=1.0):

    def _parse_ksize(ss):
        if ss.isdigit():
            return int(ss)
        else:
            return [int(k) for k in ss.split('.')]

    arch_args = []
    for stack_idx, block_strings in enumerate(arch_def):
        assert isinstance(block_strings, list)
        stack_args = []
        repeats = []
        options = {}
        skip = None
        for block_str in block_strings:
            ops = block_str.split('_')
            block_type = ops[0]
            ops = ops[1:]
            for op in ops:
                if op == 'noskip':
                    skip = False  # force no skip connection
                elif op == 'skip':
                    skip = True  # force a skip connection
                else:
                    splits = re.split(r'(\d.*)', op)
                    if len(splits) >= 2:
                        key, value = splits[:2]
                        options[key] = value

        num_repeat = int(options['r'])
        num_repeat = int(math.ceil(num_repeat * depth_multiplier))
        exp_kernel_size = _parse_ksize(options['a']) if 'a' in options else 1
        pw_kernel_size = _parse_ksize(options['p']) if 'p' in options else 1

        if block_type == 'ir':
            block_args = dict(
                block_type=block_type,
                dw_kernel_size=_parse_ksize(options['k']),
                exp_kernel_size=exp_kernel_size,
                pw_kernel_size=pw_kernel_size,
                out_channels=int(options['c']),
                exp_ratio=float(options['e']),
                se_ratio=float(options['se']) if 'se' in options else 0.,
                stride=int(options['s']),
                noskip=skip is False,
            )
        elif block_type == 'er':
            block_args = dict(
                block_type=block_type,
                exp_kernel_size=_parse_ksize(options['k']),
                pw_kernel_size=pw_kernel_size,
                out_channels=int(options['c']),
                exp_ratio=float(options['e']),
                se_ratio=float(options['se']) if 'se' in options else 0.,
                stride=int(options['s']),
                noskip=skip is False,
            )
        elif block_type == 'cn':
            block_args = dict(
                block_type=block_type,
                kernel_size=int(options['k']),
                out_channels=int(options['c']),
                stride=int(options['s']),
                skip=skip is True,
            )
        else:
            assert False, 'Unknown block type (%s)' % block_type
        arch_args.append([block_args.copy() for i in range(num_repeat)])

    return arch_args


@BACKBONES.register_module()
class EfficientNetv2_tiny(EfficientNet):

    def __init__(self,channel_multiplier=0.8, depth_multiplier=0.9, rw=False, pretrained=False, **kwargs):
        arch_def = [
            ['cn_r2_k3_s1_e1_c24_skip'],
            ['er_r4_k3_s2_e4_c48'],
            ['er_r4_k3_s2_e4_c64'],
            ['ir_r6_k3_s2_e4_c128_se0.25'],
            ['ir_r9_k3_s1_e6_c160_se0.25'],
            ['ir_r15_k3_s2_e6_c256_se0.25'],
        ]
        num_features = 1280
        if rw:
            # my original variant, based on paper figure differs from the official release
            arch_def[0] = ['er_r2_k3_s1_e1_c24']
            arch_def[-1] = ['ir_r15_k3_s2_e6_c272_se0.25']
            num_features = 1792

        round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def, depth_multiplier),
            num_features=round_chs_fn(num_features),
            stem_size=24,
            round_chs_fn=round_chs_fn,
            **kwargs,
        )
        super(EfficientNetv2_tiny, self).__init__(**model_kwargs)
    # return EfficientNet()


if __name__=="__main__":
    from test_utils.evaluator.metrics import model_info

    model = EfficientNetv2_tiny()
    o = model(torch.randn(2, 3, 224, 224))
    print("efficientnetv2_rw_t :", model_info(model, 224))
    pass