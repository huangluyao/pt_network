import torch
import torch.nn as nn
from .decode_head import BaseDecodeHead
from ..builder import HEADS, build_loss
import torch.nn.functional as F
from networks.base.cnn import resize


@HEADS.register_module()
class PRAHead(BaseDecodeHead):
    def __init__(self, in_channels, loss, dr_channels=[64, 128, 256, 512], out_channel=64, **kwargs):
        super(PRAHead, self).__init__(in_channels,loss=loss, **kwargs)

        self.decoder = Decoder(dr_channels, out_channel, f_chn=in_channels)

    def init_weights(self):
        pass

    def forward(self, inputs, return_feat=False):
        features = self.decoder(inputs)
        logits = self.conv_seg(features)
        if return_feat:
            return logits, features
        else:
            return logits

    def losses(self, seg_logit, seg_label):
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        for loss_decode in losses_decode:
            loss[loss_decode.loss_name] = loss_decode(
                 seg_logit,
                 seg_label,
                 weight=None,
                 ignore_label=self.ignore_label)

        return loss

    def forward_train(self, inputs, **kwargs):
        """Forward function for training.

        Parameters
        ----------
        inputs : list[Tensor]
            List of multi-level img features.

        Returns
        -------
        dict[str, Tensor]
            a dictionary of loss components
        """
        return self.forward(inputs)


class Decoder(nn.Module):
    def __init__(self, dr_channels=[64, 128, 256, 512], out_channel=64, f_chn=32):
        super(Decoder, self).__init__()
        self.out_channel = out_channel
        for i in range(len(dr_channels)):
            self.add_module(name=f'dr{i + 2}', module=nn.Sequential(BasicConv2d(dr_channels[i], out_channel, 3, 1, 1)))

        self.last_conv = nn.Sequential(BasicConv2d(384, 256, 3, 1, 1
                                                   ),
                                       nn.Dropout(0.5),
                                       BasicConv2d(256, f_chn, 3, 1, 1
                                                   ), )
        self.pd_module = aggregation(out_channel)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feature_lists: list):
        f2, f3, f4, x = feature_lists

        x2, x3, x4, x = self.dr2(f2), self.dr3(f3), self.dr4(f4), self.dr5(x)
        final = self.pd_module(x4, x3, x2)
        #
        x = F.interpolate(x, size=final.size()[2:], mode='bilinear', align_corners=True)
        # reverse attention
        x = -1 * (torch.sigmoid(x)) + 1
        att = x.expand(-1, self.out_channel, -1, -1).mul(final)
        final = final + att
        return final


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)
        # self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, act=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = act
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x
