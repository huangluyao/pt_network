from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseHead(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        super(BaseHead, self).__init__()

    def init_weights(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, gt_labels, **kwargs):
        logits = self.forward(inputs)
        losses = self.losses(logits, gt_labels)
        return losses

    def forward_infer(self, inputs, **kwargs):
        return self.forward(inputs)
