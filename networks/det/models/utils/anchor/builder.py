# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/29 上午10:05
from base.utils import Registry, build_from_cfg

ANCHOR_GENERATORS = Registry('Anchor generator')


def build_anchor_generator(cfg, default_args=None):
    return build_from_cfg(cfg, ANCHOR_GENERATORS, default_args)
