# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/11 下午2:53
import inspect
import math

import torch
from torch.optim import lr_scheduler


def build_optimizer(model, optimizer_cfg):
    if hasattr(model, 'module'):
        model = model.module

    _optimizer_cfg = optimizer_cfg.copy()
    opt_type = _optimizer_cfg.pop('type', 'SGD')
    _optimizer_cfg['params'] = model.parameters()
    for module_name in dir(torch.optim):
        if module_name.startswith('__') or module_name != opt_type:
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            return _optim(**_optimizer_cfg)


def build_scheduler(optimizer, lr_config):
    _lr_config = lr_config.copy()
    warm_up_steps = _lr_config.pop('warm_up_epochs', 1)
    max_steps = _lr_config.pop('max_epochs', 1000)
    _scheduler = lambda step: step / warm_up_steps if step <= warm_up_steps \
                  else 0.5 * (1 + math.cos((step - warm_up_steps) / (max_steps - warm_up_steps) * math.pi))
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=_scheduler)
