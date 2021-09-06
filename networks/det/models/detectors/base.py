from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

import base
from base.utils import get_logger, print_log


class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_bbox(self):
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, inputs):
        """Extract features from images."""
        pass

    @abstractmethod
    def forward_train(self, inputs, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def forward_infer(self, inputs, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    def forward(self, inputs, return_metrics=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_infer` depending
        on whether ``return_metrics`` is ``True``.
        """
        if return_metrics:
            metrics = self.forward_train(inputs, **kwargs)
            return self._parse_metrics(metrics)
        else:
            return self.forward_infer(inputs, **kwargs)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Parameters
        ----------
        pretrained : str, optional
            Path to pre-trained weights.
            Defaults to None.
        """
        if pretrained is not None:
            logger = get_logger('deepcv_det')
            print_log(f'load model from: {pretrained}', logger=logger)

    @staticmethod
    def _parse_metrics(metrics):
        """Parse the raw outputs (metrics) of the network.

        Parameters
        ----------
        metrics : dict
            Raw output of the network, which usually contain
            losses and other necessary information.

        Returns
        -------
        tuple[Tensor, dict]
            (loss, metrics), loss is the loss tensor
            which may be a weighted sum of all losses, metrics contains
            all the metric values.
        """
        parsed_metrics = OrderedDict()
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, torch.Tensor):
                parsed_metrics[metric_name] = metric_value.mean()
            elif isinstance(metric_value, list):
                parsed_metrics[metric_name] = sum(_metric.mean() for _metric in metric_value)
            elif isinstance(metric_value, dict):
                for name, value in metric_value.items():
                    parsed_metrics[name] = value
            else:
                raise TypeError(
                    f'{metric_name} is not a tensor or list(dict) of tensors')

        loss = sum(_value for _key, _value in parsed_metrics.items()
                   if 'loss' in _key)
        parsed_metrics['loss'] = loss

        return parsed_metrics
