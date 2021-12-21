import torch
from networks.base.utils import Registry, build_from_cfg

RUNNERS = Registry('runner')
OPTIMIZER = Registry('optimizer')
for module_name in dir(torch.optim):
    if module_name.startswith('_') or module_name.islower():
        continue
    optim = getattr(torch.optim, module_name)
    OPTIMIZER.register_module(module_name, module=optim)

def build_runner(cfg, default_args=None):
    return build_from_cfg(cfg, RUNNERS, default_args=default_args)


