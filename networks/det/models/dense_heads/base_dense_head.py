from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):
        super(BaseDenseHead, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self, inputs, ground_truth, **kwargs):
        preds = self(inputs)
        losses = self.loss(preds, ground_truth, **kwargs)
        return losses
