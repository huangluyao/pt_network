import torch.nn as nn
from abc import ABCMeta, abstractmethod

class BaseLoss(nn.Module, metaclass=ABCMeta):
    def __init__(self,
                 loss_name,
                 ignore_label=-100,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs
                 ):
        super(BaseLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.ignore_label = ignore_label

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

    @abstractmethod
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_label=None,
                **kwargs):
        pass