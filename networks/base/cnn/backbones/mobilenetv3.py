import torch
import torch.nn as nn


from ..components import ConvModule,_BatchNorm
from ...utils import get_logger
from ...utils import load_checkpoint
from ..utils import constant_init, kaiming_init
from .builder import BACKBONES
from ..components.blocks import InvertedResidual

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@BACKBONES.register_module()
class MobileNetV3(nn.Module):

    def __init__(self, mode, input_channel, width_mult=1.,
                 out_levels=[2,3,4,5],
                 norm_cfg=dict(type='BN2d'), num_classes=None):
        super(MobileNetV3, self).__init__()
        assert mode in ['large', 'small'], "model is large or small"

        if mode == "large":
            self.cfgs = [
                        # k,  t,  c, SE, HS, s
                        [3,   1,  16, 0, 0, 1],
                        [3,   4,  24, 0, 0, 2],
                        [3,   3,  24, 0, 0, 1],
                        [5,   3,  40, 1, 0, 2],
                        [5,   3,  40, 1, 0, 1],
                        [5,   3,  40, 1, 0, 1],
                        [3,   6,  80, 0, 1, 2],
                        [3, 2.5,  80, 0, 1, 1],
                        [3, 2.3,  80, 0, 1, 1],
                        [3, 2.3,  80, 0, 1, 1],
                        [3,   6, 112, 1, 1, 1],
                        [3,   6, 112, 1, 1, 1],
                        [5,   6, 160, 1, 1, 2],
                        [5,   6, 160, 1, 1, 1],
                        [5,   6, 160, 1, 1, 1]]
        else:
            self.cfgs = [
                        # k, t, c, SE, HS, s
                        [3,    1,  16, 1, 0, 2],
                        [3,  4.5,  24, 0, 0, 2],
                        [3, 3.67,  24, 0, 0, 1],
                        [5,    4,  40, 1, 1, 2],
                        [5,    6,  40, 1, 1, 1],
                        [5,    6,  40, 1, 1, 1],
                        [5,    3,  48, 1, 1, 1],
                        [5,    3,  48, 1, 1, 1],
                        [5,    6,  96, 1, 1, 2],
                        [5,    6,  96, 1, 1, 1],
                        [5,    6,  96, 1, 1, 1]]
        self.num_classes = num_classes
        mid_channel = _make_divisible(16 * width_mult, 8)

        h_swish_cfg = dict(type='HardSwish')
        self.conv = ConvModule(input_channel, mid_channel,3,2,1,act_cfg=h_swish_cfg, norm_cfg=norm_cfg)
        self.fist_conv_out = 1 in out_levels

        block = InvertedResidual
        self.out_levels = []
        levels_index = 1

        self.layer_names = []
        for i, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(mid_channel * t, 8)

            layer_name = f"layer_{i}"
            if use_hs:
                self.add_module(layer_name, block(mid_channel, exp_size, output_channel, k, s, use_se, act_cfg=h_swish_cfg, norm_cfg=norm_cfg))
            else:
                self.add_module(layer_name, block(mid_channel, exp_size, output_channel, k, s, use_se, norm_cfg=norm_cfg))
                # self.layers.append(block(mid_channel, exp_size, output_channel, k, s, use_se, norm_cfg=norm_cfg))
            self.layer_names.append(layer_name)
            mid_channel = output_channel

            if s == 2:
                levels_index += 1
                if levels_index in out_levels:
                    self.out_levels.append(i)

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(output_channel,  num_classes)


    def forward(self, x):
        x = self.conv(x)
        outputs = [x] if self.fist_conv_out else []
        for i, layers in enumerate(self.layer_names):
            x = getattr(self, layers)(x)
            if i in self.out_levels:
               outputs.append(x)

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
        return outputs if len(outputs) > 1 else outputs[0]


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
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
        else:
            raise TypeError('pretrained must be a str or None')
