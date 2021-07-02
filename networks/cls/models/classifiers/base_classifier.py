from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from base.utils import print_log


class BaseClassifier(nn.Module, metaclass=ABCMeta):
    """Base class for classifiers"""

    def __init__(self):
        super(BaseClassifier, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        return hasattr(self, 'head') and self.head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, inputs, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def forward_infer(self, inputs, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    def forward(self, inputs, return_metrics=False, **kwargs):
        if return_metrics:
            metrics = self.forward_train(inputs, **kwargs)
            return metrics
        else:
            return self.forward_infer(inputs, **kwargs)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info(f'load model from: {pretrained}')
