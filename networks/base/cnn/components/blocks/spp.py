import torch
import torch.nn as nn
from ..conv_module import ConvModule

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), **kwargs):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvModule(c1, c_, 1, 1, **kwargs)
        self.cv2 = ConvModule(c_ * (len(k) + 1), c2, 1, 1, **kwargs)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))