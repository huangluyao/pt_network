import torch

from .hook import HOOKS, Hook
from ...utils import is_list_of
from copy import deepcopy
from functools import partial

@HOOKS.registry()
class ExponentialMovingAverageHook(Hook):

    def __init__(self,
                 module_keys,
                 interp_mode='lerp',
                 interp_cfg=None,
                 interval=-1,
                 start_iter=0
                 ):
        super(ExponentialMovingAverageHook, self).__init__()

        assert isinstance(module_keys, str) or is_list_of(module_keys, str)

        self.module_keys = (module_keys, ) if isinstance(module_keys, str) else module_keys

        # sanity check for the format of module keys
        for k in self.module_keys:
            assert k.endswith(
                '_ema'), 'You should give keys that end with "_ema".'

        self.interp_mode = interp_mode
        self.interp_cfg = dict() if interp_cfg is None else deepcopy(interp_cfg)
        self.interval = interval
        self.start_iter = start_iter

        assert hasattr(
            self, interp_mode
        ), f'Currently, we do not support {self.interp_mode} for EMA.'
        self.interp_func = partial(
            getattr(self, interp_mode), **self.interp_cfg)

    @staticmethod
    def lerp(a, b, momentum=0.999, momentum_nontrainable=0., trainable=True):
        m = momentum if trainable else momentum_nontrainable
        return a + (b-a)*m

    def every_n_iters(self, runner, n):
        if runner.iter < self.start_iter:
            return True
        return (runner.iter + 1 - self.start_iter) % n == 0 if n > 0 else False


    @torch.no_grad()
    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        model = runner.model
        for key in self.module_keys:
            # get current ema states
            ema_net = getattr(model, key)
            states_ema = ema_net.state_dict(keep_vars=False)
            # get currently original states
            net = getattr(model, key[:-4])
            states_orig = net.state_dict(keep_vars=True)
            for k, v in states_orig.items():
                if runner.iter < self.start_iter:
                    states_ema[k].data.copy_(v.data)
                else:
                    states_ema[k] = self.interp_func(
                        v, states_ema[k], trainable=v.requires_grad).detach()

            ema_net.load_state_dict(states_ema, strict=True)

    def before_train(self, runner):
        model = runner.model
        # sanity check for ema model
        for k in self.module_keys:
            if not hasattr(model, k) and not hasattr(model, k[:-4]):
                raise RuntimeError(
                    f'Cannot find both {k[:-4]} and {k} network for EMA hook.')
            if not hasattr(model, k) and hasattr(model, k[:-4]):
                setattr(model, k, deepcopy(getattr(model, k[:-4])))
                runner.logger.info(
                    f'We do not suggest construct and initialize EMA model {k}'
                    ' in hook. You may explicitly define it by yourself.')
