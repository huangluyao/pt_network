from ..conv_module import ConvModule
from ..registry import BLOCK_LAYERS
import torch.nn as nn
import torch

class Bottleneck(nn.Module):
    """
    A standard bottlenck from resnet

    parameters
    ----------
    channel_in: input channel dimension of bottleneckcsp structure;

    channel_out: output channel dimension of bottleneckcsp structure;

    Shortcut: whether to add a shortcut connection to the bottleneck structure. After adding, it is the ResNet module;

    g: Groups, the parameters of channel grouping, the number of input channels and the number of output channels must be divisible by groups at the same time;

    e: Expansion: the channel expansion rate of the bottleneck part in the bottleneck structure is 0.5;
    """
    def __init__(self, channel_in, channel_out, shortcut=True, g=1, e=0.5,**cfg):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(channel_out * e)  # hidden channels
        self.cv1 = ConvModule(channel_in, c_, 1, 1,**cfg)
        self.cv2 = ConvModule(c_, channel_out, 3, 1,1, groups=g, **cfg)
        self.add = shortcut and channel_in == channel_out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


@BLOCK_LAYERS.register_module()
class CSP_Bottlenck(nn.Module):
    """
    CSP Bottleneck with 3 convolutions

    The main purpose of designing CSPNet is to enable this architecture to achieve
    a richer gradient combination while reducing the amount of computation. This
    aim is achieved by partitioning feature map of the base layer into two parts
    and then merging them through a proposed crossstage hierarchy

    reference https://github.com/WongKinYiu/CrossStagePartialNetworks

    Parameters
    ----------

    channel_in: input channel dimension of bottleneckcsp structure;

    channel_out: output channel dimension of bottleneckcsp structure;

    n: The number of bottleneck structure;

    Shortcut: whether to add a shortcut connection to the bottleneck structure. After adding, it is the ResNet module;

    g: Groups, the parameters of channel grouping, the number of input channels and the number of output channels must be divisible by groups at the same time;

    e: Expansion: the channel expansion rate of the bottleneck part in the bottleneck structure is 0.5;

    cfg: config parameters in ConvModule

    """

    def __init__(self, channel_in, channel_out, n=1, shortcut=True, g=1, e=0.5, **cfg):
        super(CSP_Bottlenck, self).__init__()
        c_ = int(channel_out * e)  # hidden channels
        self.cv1 = ConvModule(channel_in, c_, 1, 1, **cfg)
        self.cv2 = ConvModule(channel_in, c_, 1, 1, **cfg)
        self.cv3 = ConvModule(2 * c_, channel_out, 1, **cfg)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0, **cfg) for _ in range(n)])
        pass

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))