import torch.nn as nn
from base.cnn import normal_init

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class LinearClsHead(ClsHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 **kwargs):
        super(ClsHead, self).__init__()
        assert isinstance(loss, dict)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def forward(self, inputs):
        logits = self.fc(inputs)
        return logits
