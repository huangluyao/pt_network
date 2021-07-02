from itertools import accumulate
import torch
import torch.nn as nn

from .builder import BACKBONES
from ..components import build_norm_layer, _BatchNorm
from ...utils import load_checkpoint
from ...utils import get_logger


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride, norm_cfg):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=mid_channels),
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                build_norm_layer(norm_cfg, anonymous=True, num_features=inp),
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                build_norm_layer(norm_cfg, anonymous=True, num_features=inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


def channel_shuffle(x):
    B, C, H, W = x.data.size()
    assert (C % 4 == 0)
    x = x.reshape(B, C // 2, 2, H * W)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(B, 2, C // 2, H, W)
    return x[:, 0], x[:, 1]



@BACKBONES.register_module()
class ShuffleNetV2(nn.Module):
    def __init__(self,
                 size='1.5x',
                 in_channels=3,
                 out_levels=(1, 2, 3, 4, 5),
                 num_classes=None,
                 final_drop=0.0,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(ShuffleNetV2, self).__init__()

        self.num_classes = num_classes
        self.out_levels = out_levels
        self.stage_repeats = [4, 8, 4]
        self.model_size = size
        self.stage_out_channels = {
            '0.5x': [-1, 24, 48, 96, 192, 1024],
            '1.0x': [-1, 24, 116, 232, 464, 1024],
            '1.5x': [-1, 24, 176, 352, 704, 1024],
            '2.0x': [-1, 24, 244, 488, 976, 2048]
        }[size]
        self.final_drop = final_drop
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
            build_norm_layer(self.norm_cfg, input_channel)[1],
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel,
                        output_channel // 2, ksize=3, stride=2, norm_cfg=norm_cfg))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel,
                        output_channel // 2, ksize=3, stride=1, norm_cfg=norm_cfg))

                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        output_indices = list(accumulate(self.stage_repeats))
        self.block_in_stages = []
        in_idx = 0
        for i in range(len(output_indices)):
            out_idx = output_indices[i]
            self.block_in_stages.append((in_idx, out_idx))
            in_idx = out_idx

        if num_classes is not None:
            self.conv_last = nn.Sequential(
                nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
                build_norm_layer(self.norm_cfg, self.stage_out_channels[-1])[1],
                nn.ReLU(inplace=True)
            )
            self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
            if self.final_drop > 0:
                self.dropout = nn.Dropout(self.final_drop)
            else:
                self.dropout = None
            self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], num_classes, bias=False))

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, _BatchNorm):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        x = self.first_conv(x)
        if 1 in self.out_levels:
            outs.append(x)
        x = self.maxpool(x)
        if 2 in self.out_levels:
            outs.append(x)

        for i, indices in enumerate(self.block_in_stages):
            x = self.features[indices[0]:indices[1]](x)
            if i+3 in self.out_levels:
                outs.append(x)

        if self.num_classes is not None:
            x = self.conv_last(x)
            x = self.globalpool(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = x.contiguous().view(-1, self.stage_out_channels[-1])
            x = self.classifier(x)
            return x
        else:
            if len(outs) == 1:
                return outs[0]
            else:
                return tuple(outs)
