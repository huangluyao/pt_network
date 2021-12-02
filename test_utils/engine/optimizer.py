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


def build_optimizers(model, cfgs):
    optimizers = {}
    for key, cfg in cfgs.items():
        if hasattr(model, key):
            module = getattr(model, key)
            optimizers[key] = build_optimizer(module, cfg)
    return optimizers


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
    elif lr_decay_method == "cosine_annealing_restarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                        T_0=cycle_epoch,
                                                        )

    else:
        raise ValueError("unsupported lr decay method: {}".format(lr_decay_method))
    return lr_scheduler.LambdaLR(optimizer, lr_lambda=_scheduler)


from collections import defaultdict
from torch.optim import Optimizer
import torch


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)