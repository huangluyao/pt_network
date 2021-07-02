import torch
from torch import nn
from torch.nn import functional as F

from .builder import BACKBONES
from ..components import (build_activation_layer, build_conv_layer,
                          build_norm_layer, _BatchNorm)
from ...utils import load_checkpoint


class SELayer(nn.Module):

	def __init__(self,
                 inplanes,
                 isTensor=True,
                 norm_cfg=dict(type='BN', requires_grad=True, anonymous=True)):
		super(SELayer, self).__init__()
		if isTensor:
			self.SE_opr = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, stride=1, bias=False),
                build_norm_layer(norm_cfg, anonymous=True, num_features=inplanes // 4),
				nn.ReLU(inplace=True),
				nn.Conv2d(inplanes // 4, inplanes, kernel_size=1, stride=1, bias=False),
			)
		else:
			self.SE_opr = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Linear(inplanes, inplanes // 4, bias=False),
				nn.BatchNorm1d(inplanes // 4),
				nn.ReLU(inplace=True),
				nn.Linear(inplanes // 4, inplanes, bias=False),
			)

	def forward(self, x):
		atten = self.SE_opr(x)
		atten = torch.clamp(atten + 3, 0, 6) / 6
		return x * atten


class HS(nn.Module):

	def __init__(self):
		super(HS, self).__init__()

	def forward(self, inputs):
		clip = torch.clamp(inputs + 3, 0, 6) / 6
		return inputs * clip


class Shufflenet(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 base_mid_channels,
                 *,
                 ksize,
                 stride,
                 dilation,
                 activation,
                 useSE,
                 projection_shortcut=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert base_mid_channels == oup//2

        self.base_mid_channel = base_mid_channels
        self.ksize = ksize
        if dilation > 1:
            ksize_effective = ksize + (ksize - 1) * (dilation - 1)
            pad = (ksize_effective - 1) // 2
        else:
            pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.projection_shortcut = projection_shortcut
        outputs = oup - inp

        branch_main = [
            nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=base_mid_channels),
            None,
            nn.Conv2d(base_mid_channels,
                      base_mid_channels,
                      ksize,
                      stride=stride,
                      dilation=dilation,
                      padding=pad,
                      groups=base_mid_channels,
                      bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=base_mid_channels),
            nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=outputs),
            None,
        ]
        if activation == 'ReLU':
            assert useSE == False
            '''This model should not have SE with ReLU'''
            branch_main[2] = nn.ReLU(inplace=True)
            branch_main[-1] = nn.ReLU(inplace=True)
        else:
            branch_main[2] = HS()
            branch_main[-1] = HS()
            if useSE:
                branch_main.append(SELayer(outputs, norm_cfg=norm_cfg))
        self.branch_main = nn.Sequential(*branch_main)

        if projection_shortcut:
            branch_proj = [
                nn.Conv2d(inp, inp, ksize, stride, ksize // 2, groups=inp, bias=False),
                build_norm_layer(norm_cfg, anonymous=True, num_features=inp),
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                build_norm_layer(norm_cfg, anonymous=True, num_features=inp),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HS()
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if not self.projection_shortcut:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        else:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


class Shuffle_Xception(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 base_mid_channels,
                 *,
                 stride,
                 dilation,
                 activation,
                 useSE,
                 projection_shortcut=False,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(Shuffle_Xception, self).__init__()

        assert stride in [1, 2]
        assert base_mid_channels == oup//2

        self.base_mid_channel = base_mid_channels
        self.stride = stride
        ksize = 3
        self.ksize = ksize
        if dilation > 1:
            ksize_effective = ksize + (ksize - 1) * (dilation - 1)
            pad = (ksize_effective - 1) // 2
        else:
            pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.projection_shortcut = projection_shortcut
        outputs = oup - inp

        branch_main = [
            nn.Conv2d(inp,
                      inp,
                      3,
                      stride=stride,
                      dilation=dilation,
                      padding=self.pad,
                      groups=inp,
                      bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=inp),
            nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=base_mid_channels),
            None,
            nn.Conv2d(base_mid_channels,
                      base_mid_channels,
                      3,
                      stride=stride,
                      dilation=dilation,
                      padding=self.pad,
                      groups=base_mid_channels,
                      bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=base_mid_channels),
            nn.Conv2d(base_mid_channels, base_mid_channels, 1, 1, 0, bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=base_mid_channels),
            None,
            nn.Conv2d(base_mid_channels,
                      base_mid_channels,
                      3,
                      stride=stride,
                      dilation=dilation,
                      padding=self.pad,
                      groups=base_mid_channels,
                      bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=base_mid_channels),
            nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=outputs),
            None,
        ]

        if activation == 'ReLU':
            branch_main[4] = nn.ReLU(inplace=True)
            branch_main[9] = nn.ReLU(inplace=True)
            branch_main[14] = nn.ReLU(inplace=True)
        else:
            branch_main[4] = HS()
            branch_main[9] = HS()
            branch_main[14] = HS()
        assert None not in branch_main

        if useSE:
            assert activation != 'ReLU'
            branch_main.append(SELayer(outputs, norm_cfg=norm_cfg))

        self.branch_main = nn.Sequential(*branch_main)

        if projection_shortcut:
            branch_proj = [
                nn.Conv2d(inp, inp, 3, stride, ksize // 2, groups=inp, bias=False),
                build_norm_layer(norm_cfg, anonymous=True, num_features=inp),
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                build_norm_layer(norm_cfg, anonymous=True, num_features=inp),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HS()
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if not self.projection_shortcut:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        else:
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
class ShuffleNetV2Plus(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_classes=1000,
                 architecture=[0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2],
                 size='Small',
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_levels=(1, 2, 3, 4, 5),
                 multi_grid=None,
                 contract_dilation=False,
                 norm_cfg=dict(type='BN', momentum=0.1, requires_grad=True)):
        super(ShuffleNetV2Plus, self).__init__()
        assert architecture is not None
        self.num_classes = num_classes
        self.out_levels = out_levels
        self.multi_grid = multi_grid

        self.stage_repeats = [4, 4, 8, 4]
        if size.lower() == 'large':
            self.stage_out_channels = [-1, 16, 68, 168, 336, 672, 1280]
        elif size.lower() == 'medium':
            self.stage_out_channels = [-1, 16, 48, 128, 256, 512, 1280]
        elif size.lower() == 'small':
            self.stage_out_channels = [-1, 16, 36, 104, 208, 416, 1280]
        else:
            raise NotImplementedError

        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, 2, 1, bias=False),
            build_norm_layer(norm_cfg, anonymous=True, num_features=input_channel),
            HS(),
        )

        self.features = []
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            stride = strides[idxstage]
            stage_dilation = dilations[idxstage]
            if idxstage < (len(self.stage_repeats) - 1):
                multi_grid = None
            else:
                multi_grid = self.multi_grid

            if multi_grid is None:
                if stage_dilation > 1 and contract_dilation:
                    first_dilation = stage_dilation // 2
                else:
                    first_dilation = stage_dilation
            else:
                first_dilation = multi_grid[0]

            activation = 'HS' if idxstage >= 1 else 'ReLU'
            useSE = 'True' if idxstage >= 2 else False

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride, projection = input_channel, output_channel, stride, True
                else:
                    inp, outp, stride, projection = input_channel // 2, output_channel, 1, False

                blockIndex = architecture[archIndex]
                archIndex += 1

                if i == 0:
                    dilation = first_dilation
                else:
                    dilation = stage_dilation if multi_grid is None else multi_grid[i]

                block_cfg = dict(
                    inp=inp,
                    oup=outp,
                    base_mid_channels=outp // 2,
                    stride=stride,
                    dilation=dilation,
                    activation=activation,
                    useSE=useSE,
                    projection_shortcut=projection,
                    norm_cfg=norm_cfg
                )
                if blockIndex == 0:
                    self.features.append(Shufflenet(ksize=3, **block_cfg))
                elif blockIndex == 1:
                    self.features.append(Shufflenet(ksize=5, **block_cfg))
                elif blockIndex == 2:
                    self.features.append(Shufflenet(ksize=7, **block_cfg))
                elif blockIndex == 3:
                    self.features.append(Shuffle_Xception(**block_cfg))
                else:
                    raise NotImplementedError
                input_channel = output_channel

        assert archIndex == len(architecture)
        self.features = nn.Sequential(*self.features)

        from itertools import accumulate
        output_indices = list(accumulate(self.stage_repeats))
        self.block_in_stages = []
        in_idx = 0
        for i in range(len(output_indices)):
            out_idx = output_indices[i]
            self.block_in_stages.append((in_idx, out_idx))
            in_idx = out_idx

        if num_classes is not None:
            self.conv_last = nn.Sequential(
                nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=False),
                build_norm_layer(norm_cfg, anonymous=True, num_features=1280),
                HS()
            )
            self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
            self.LastSE = SELayer(1280, norm_cfg=norm_cfg)
            self.fc = nn.Sequential(
                nn.Linear(1280, 1280, bias=False),
                HS(),
            )
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(1280, self.num_classes, bias=False)

    def forward(self, x):
        outs = []
        if 0 in self.out_levels:
            outs.append(x)
        x = self.first_conv(x)
        if 1 in self.out_levels:
            outs.append(x)
        for i, indices in enumerate(self.block_in_stages):
            x = self.features[indices[0]:indices[1]](x)
            if i+2 in self.out_levels:
                outs.append(x)

        if self.num_classes is not None:
            x = self.conv_last(x)
            x = self.globalpool(x)
            x = self.LastSE(x)

            x = x.contiguous().view(-1, 1280)

            x = self.fc(x)
            x = self.dropout(x)
            x = self.classifier(x)
            return x
        else:
            if len(outs) == 1:
                return outs[0]
            else:
                return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        else:
            self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'SE' in name:
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

    def train(self, mode=True):
        super(ShuffleNetV2Plus, self).train(mode)