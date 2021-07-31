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


class CosineDecayRestart:
    def __init__(self, cycle_epoch, cycle_radio=1.0, warm_up_steps=5):
        self.warm_up_steps = warm_up_steps
        self.cycle_epoch = cycle_epoch * 0.5
        self.cycle_radio = cycle_radio
        self.decay_radio = 1.
        self.cycle_times = 1

    def __call__(self, step):
        if step < self.warm_up_steps:
            return step / self.warm_up_steps if step else 0.1
        elif step - self.warm_up_steps > self.cycle_epoch * self.cycle_times:
            self.cycle_times += 1
            if self.cycle_times % 2 == 0:
                self.decay_radio *= self.cycle_radio
            print(self.decay_radio * 0.5 * (1 + math.cos((step - self.warm_up_steps) / (self.cycle_epoch) * math.pi)))
        return self.decay_radio * 0.5 * (1 + math.cos((step - self.warm_up_steps) / (self.cycle_epoch) * math.pi))


class PolynomialDecay:
    def __init__(self, cycle_epoch, cycle_radio, warm_up_steps=5, power=3):
        self.warm_up_steps = warm_up_steps
        self.cycle_epoch = cycle_epoch
        self.cycle_radio = cycle_radio
        self.decay_radio = 1
        self.decay_step = 0
        self.cycle_times = 1
        self.power = power

    def __call__(self, step):
        if step < self.warm_up_steps:
            return step / self.warm_up_steps if step else 0.1
        elif step - self.warm_up_steps > self.cycle_epoch * self.cycle_times:
            self.cycle_times += 1
            self.decay_step += self.cycle_epoch
            self.decay_radio *= self.cycle_radio
            pass
        return self.decay_radio * (1 - float(step - self.warm_up_steps - self.decay_step) / self.cycle_epoch) ** (
            self.power)


def build_scheduler(optimizer, lr_config):
    _lr_config = lr_config.copy()
    warm_up_steps = _lr_config.pop('warm_up_epochs', 1)
    lr_decay_method = _lr_config.pop("lr_decay_method", "cosine_decay_restarts")
    cycle_epoch = _lr_config.pop("cycle_epoch", 30)
    cycle_radio = _lr_config.pop("cycle_radio", 1.)

    if lr_decay_method == "cosine_decay_restarts":
        _scheduler = CosineDecayRestart(cycle_epoch,cycle_radio,warm_up_steps)
    elif lr_decay_method == "polynomial_decay_restarts":
        _scheduler = PolynomialDecay(cycle_epoch,cycle_radio,_lr_config.pop('power', 1))
    else:
        raise ValueError("unsupported lr decay method: {}".format(lr_decay_method))
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=_scheduler)
