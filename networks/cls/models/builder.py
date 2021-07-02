import torch.nn as nn

from base.cnn import BACKBONES, build_backbone
from base.utils import Registry, build_from_cfg, build_module

CLASSIFIERS = Registry('classifier')
HEADS = Registry('head')
NECKS = Registry('neck')
LOSSES = Registry('loss')


def build_neck(cfg):
    return build_module(cfg, NECKS)


def build_head(cfg):
    return build_module(cfg, HEADS)


def build_loss(cfg):
    return build_module(cfg, LOSSES)


def build_classifier(cfg, **kwargs):
    return build_module(cfg, CLASSIFIERS)
