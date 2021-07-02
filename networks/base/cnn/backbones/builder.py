from ...utils import Registry, build_module

BACKBONES = Registry('backbone')


def build_backbone(cfg):
    return build_module(cfg, BACKBONES)