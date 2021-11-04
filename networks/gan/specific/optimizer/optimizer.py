import torch
import inspect

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

def build_dict_optimizer(model, cfgs):
    optimizers = {}
    if hasattr(model, 'module'):
        model = model.module

    for key, cfg in cfgs.items():
        cfg_ = cfg.copy()
        module = getattr(model, key)
        optimizers[key] = build_optimizer(module, cfg_)
    return optimizers