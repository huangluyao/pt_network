import torch
import torch.nn as nn
from ..components import ConvModule,_BatchNorm, DepthwiseSeparableConvModule
from ...utils import get_logger
from ...utils import load_checkpoint
from ..utils import constant_init, kaiming_init
from .builder import BACKBONES
from .utils import make_divisible
from ..components.plugins import SELayer
from ..components.blocks import InvertedResidual

@BACKBONES.register_module()
class MobileNetV3(nn.Module):

    def __init__(self,
                 mode,
                 in_channels=3,
                 stem_size=16,
                 width_mult=1.,
                 out_levels=[2,3,4,5],
                 norm_cfg=dict(type='BN2d'),
                 act_cfg=dict(type='ReLU'),
                 num_classes=None,
                 se_act_cfg=[dict(type="ReLU"), dict(type='HardSigmoid')],
                 se_radio=4,
                 final_drop=0.0,
                 **kwargs):
        super(MobileNetV3, self).__init__()
        assert mode in ['large', 'small'], "model is large or small"
        self.num_classes=num_classes
        self.fist_conv_out = 1 in out_levels
        if mode == "large":
            cfgs = [
                #k, e,   c,  SE,HS,s
                [3, 4,   24, 0, 0, 2],
                [3, 3,   24, 0, 0, 1],

                [5, 3,   40, 1, 0, 2],
                [5, 3,   40, 1, 0, 1],
                [5, 3,   40, 1, 0, 1],

                [3, 6,   80, 0, 1, 2],
                [3, 2.5, 80, 0, 1, 1],
                [3, 2.3, 80, 0, 1, 1],
                [3, 2.3, 80, 0, 1, 1],

                [3, 6,  112, 1, 1, 1],
                [3, 6,  112, 1, 1, 1],

                [5, 6,  160, 1, 1, 2],
                [5, 6,  160, 1, 1, 1],
                [5, 6,  160, 1, 1, 1]
            ]
        else:
            cfgs = []

        stem_ch = make_divisible(stem_size)
        self.conv1 = ConvModule(in_channels, stem_ch, kernel_size=3, stride=2, padding=1, bias=False,
                                norm_cfg=norm_cfg, act_cfg=dict(type='HardSwish')
                                )

        self.dsconv = DepthwiseSeparableConvModule(stem_ch, stem_ch, kernel_size=3, stride=1, padding=1,
                                                   dw_act_cfg=dict(type="ReLU"), dw_norm_cfg=norm_cfg,
                                                   pw_norm_cfg=norm_cfg, pw_act_cfg=None
                                                   )

        self.out_levels = []
        levels_index = 0

        self.layer_names = []
        in_channel = stem_ch
        hs_act_cfg = dict(type="HardSwish")
        for i, (k, e, c, use_se, use_hs, s) in enumerate(cfgs):
            output_channel = make_divisible(c * width_mult, 8)
            layer_name = f"layer_{i}"
            if not use_hs:
                self.add_module(layer_name, InvertedResidual(in_channel, output_channel, k, s, exp_ratio=e, use_se=use_se, se_radio=se_radio,
                                                             se_act_cfg=se_act_cfg,dw_act_cfg=act_cfg, pw_act_cfg=act_cfg, pw_norm_cfg=norm_cfg,
                                                             dw_norm_cfg=norm_cfg
                                                             ))
            else:
                self.add_module(layer_name, InvertedResidual(in_channel, output_channel, k, s, exp_ratio=e, use_se=use_se, se_radio=se_radio,
                                                             se_act_cfg=se_act_cfg, dw_act_cfg=hs_act_cfg, pw_act_cfg=hs_act_cfg,
                                                             pw_norm_cfg=norm_cfg, dw_norm_cfg=norm_cfg
                                                             ))

                # self.layers.append(block(mid_channel, exp_size, output_channel, k, s, use_se, norm_cfg=norm_cfg))
            self.layer_names.append(layer_name)

            if s == 2:
                levels_index += 1
                if levels_index in out_levels:
                    self.out_levels.append(i-1)

            in_channel = output_channel

        levels_index +=1
        if levels_index in out_levels:
            self.out_levels.append(len(cfgs) -1)

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
            self.fc = nn.Linear(160, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dsconv(x)
        outputs = [x] if self.fist_conv_out else []
        for i, layers in enumerate(self.layer_names):
            x = getattr(self, layers)(x)
            if i in self.out_levels:
                outputs.append(x)

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
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
