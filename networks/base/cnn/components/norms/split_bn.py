import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import NORM_LAYERS


@NORM_LAYERS.register_module('SplitBN')
class SplitBatchNorm(_BatchNorm):
    def __init__(self, num_features, num_splits,
                 eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(SplitBatchNorm, self).__init__(
            num_features=num_features,
            eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats)
        self.num_splits = num_splits
        self.start_train = False

    def train(self, mode=True):
        if (self.training is True) and (mode is False):
            self.running_mean = torch.mean(
                self.running_mean.view(self.num_splits, self.num_features), dim=0)
            self.running_var = torch.mean(
                self.running_var.view(self.num_splits, self.num_features), dim=0)
        if (self.training is False) and (mode is True):
            self.start_train = False
        return super(SplitBatchNorm, self).train(mode)

    def forward(self, input):
        if self.training:
            if not self.start_train:
                self.start_train = True
                self.running_mean = self.running_mean.repeat(self.num_splits)
                self.running_var = self.running_var.repeat(self.num_splits)
            N, C, H, W = input.shape
            return F.batch_norm(
                input.view(N // self.num_splits, C * self.num_splits, H, W),
                running_mean=self.running_mean,
                running_var=self.running_var,
                weight=self.weight.repeat(self.num_splits),
                bias=self.bias.repeat(self.num_splits),
                training=True,
                momentum=self.momentum,
                eps=self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input,
                running_mean=self.running_mean,
                running_var=self.running_var,
                weight=self.weight,
                bias=self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps)
