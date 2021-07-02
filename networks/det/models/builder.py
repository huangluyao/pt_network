import torch.nn as nn

from base.cnn import BACKBONES, build_backbone
from base.utils import Registry, build_from_cfg, build_module

DETECTORS = Registry('detector')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
METRICS = Registry('metric')


def build_neck(cfg):
    return build_module(cfg, NECKS)


def build_head(cfg):
    return build_module(cfg, HEADS)


def build_loss(cfg):
    return build_module(cfg, LOSSES)


def build_metric(cfg):
    return build_module(cfg, METRICS)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    return build_module(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
