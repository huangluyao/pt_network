import torch
import torch.nn as nn
from ..conv_module import ConvModule

class BottleneckPruned(nn.Module):

    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1, **cfg):  # ch_in, ch_out, shortcut, groups, expansion
        super(BottleneckPruned, self).__init__()

        self.cv1 = ConvModule(cv1in, cv1out, 1, 1,**cfg)
        self.cv2 = ConvModule(cv1out, cv2out, 3, 1, 1, groups=g, **cfg)
        self.add = shortcut and cv1in == cv2out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3Pruned(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, cv1in, cv1out, cv2out, cv3out, bottle_args, number=1, shortcut=True, g=1, **cfg):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3Pruned, self).__init__()
        cv3in = bottle_args[-1][-1]
        self.cv1 = ConvModule(cv1in, cv1out, 1, 1, **cfg)
        self.cv2 = ConvModule(cv1in, cv2out, 1, 1, **cfg)
        self.cv3 = ConvModule(cv3in+cv2out, cv3out, 1, **cfg)
        self.m = nn.Sequential(*[BottleneckPruned(*bottle_args[k], shortcut, g, **cfg) for k in range(number)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPPruned(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, cv1in, cv1out, cv2out, k=(5, 9, 13), **kwargs):
        super(SPPPruned, self).__init__()
        self.cv1 = ConvModule(cv1in, cv1out, 1, 1, **kwargs)
        self.cv2 = ConvModule(cv1out * (len(k) + 1), cv2out, 1, 1, **kwargs)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

