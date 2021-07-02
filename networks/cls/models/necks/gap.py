import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class GlobalAveragePooling(nn.Module):

    def __init__(self):
        super(GlobalAveragePooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, (tuple, list)):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be sequence or torch.tensor')
        return outs
