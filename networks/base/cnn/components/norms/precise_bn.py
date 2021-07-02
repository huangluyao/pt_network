import itertools
from tqdm import tqdm
from typing import Any, Iterable, List, Tuple, Type

import torch
from torch import nn

__all__ = ['get_bn_modules', 'update_bn_stats']


BN_MODULE_TYPES: Tuple[Type[nn.Module]] = (
    torch.nn.modules.batchnorm._BatchNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)


class IterLoader:

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


@torch.no_grad()
def update_bn_stats(
    model: nn.Module,
    data_loader: Iterable[Any],
    num_iters: int = 200,
    by_epoch: bool = False):

    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration, so
    the running average can not precisely reflect the actual stats of the
    current model.
    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true average
    of per-batch mean/variance instead of the running average.

    Parameters
    ----------
    model : nn.Module
        the model whose bn stats will be recomputed.
    data_loader : iterator
        an iterator. Produce data as inputs to the model.
    num_iters : int
        number of iterations to compute the stats.
    by_epoch : bool
        When by_epoch == True, num_iters means the number of epochs.

    Notes
    -----
    1. This function will not alter the training mode of the given model.
        Users are responsible for setting the layers that needs
        precise-BN to training mode, prior to calling this function.
    2. Be careful if your models contain other stateful layers in
        addition to BN, i.e. layers whose state can change in forward
        iterations. This function will alter their state. If you wish
        them unchanged, you need to either pass in a submodule without
        those layers, or backup the states.
    """
    bn_layers = get_bn_modules(model)

    if len(bn_layers) == 0:
        return

    momentum_actual = [bn.momentum for bn in bn_layers]
    for bn in bn_layers:
        bn.momentum = 1.0

    running_mean = [
        torch.zeros_like(bn.running_mean) for bn in bn_layers
    ]
    running_var = [torch.zeros_like(bn.running_var) for bn in bn_layers]

    if by_epoch:
        num_iters = num_iters * len(data_loader)

    iter_loader = IterLoader(data_loader)
    ind = -1
    with tqdm(total=num_iters) as pbar:
        pbar.set_description('Calculating running stats')
        while ind < num_iters:
            data_batch = next(iter_loader)
            output = model(data_batch['img'])

            ind += 1
            for i, bn in enumerate(bn_layers):
                running_mean[i] += (bn.running_mean - running_mean[i]) / (ind + 1)
                running_var[i] += (bn.running_var - running_var[i]) / (ind + 1)

            pbar.update(1)

    assert ind == num_iters, (
        "update_bn_stats is meant to run for {} iterations, "
        "but the dataloader stops at {} iterations.".format(num_iters, ind)
    )

    for i, bn in enumerate(bn_layers):
        bn.running_mean = running_mean[i]
        bn.running_var = running_var[i]
        bn.momentum = momentum_actual[i]


def get_bn_modules(model: nn.Module) -> List[nn.Module]:
    """
    Find all BatchNorm (BN) modules that are in training mode. See
    fvcore.precise_bn.BN_MODULE_TYPES for a list of all modules that are
    included in this search.

    Parameters
    ----------
    model : nn.Module
        a model possibly containing BN modules.

    Returns
    -------
    list[nn.Module]
        all BN modules in the model.
    """
    bn_layers = [
        m for m in model.modules() if m.training and isinstance(m, BN_MODULE_TYPES)
    ]
    return bn_layers
