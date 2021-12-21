from .build import build_scheduler
from .ranger21 import Ranger21
from .lars import LARS
from .radam import RAdam
from .lookhead import Lookahead
from .yogi import Yogi
from .diffgrad import DiffGrad
from .madgrad import MADGRAD
from .sgdp import SGDP
from ..builder import build_from_cfg, OPTIMIZER


def build_optimizer(model, cfg):
    if hasattr(model, 'module'):
        model = model.module

    _optimizer_cfg = cfg['optimizer'].copy()
    use_lookahead = _optimizer_cfg.pop('use_lookahead', False)
    _optimizer_cfg['params'] = model.parameters()
    if _optimizer_cfg.get('type', 'SGD') == "Ranger21":
        _optimizer_cfg["num_epochs"] = cfg.max_epochs
        _optimizer_cfg["num_batches_per_epoch"] = cfg.loader_cfg.batch_size

    optim = build_from_cfg(_optimizer_cfg, OPTIMIZER)
    if use_lookahead:
        optim = Lookahead(optim, k=5, alpha=0.5)

    return optim

    # for module_name in dir(torch.optim):
    #     if module_name.startswith('__') or module_name != opt_type:
    #         continue
    #     _optim = getattr(torch.optim, module_name)
    #     if inspect.isclass(_optim) and issubclass(_optim,
    #                                               torch.optim.Optimizer):
    #         return _optim(**_optimizer_cfg)


def build_optimizers(model, cfgs):
    optimizers = {}
    for key, cfg in cfgs.items():
        if hasattr(model, key):
            module = getattr(model, key)
            optimizers[key] = build_optimizer(module, cfg)
    return optimizers
