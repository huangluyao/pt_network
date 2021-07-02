import os
import os.path as osp
import pkgutil
import time
import warnings
from collections import OrderedDict

import torch


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Parameters
    ----------
    module : Module
        Module that receives the state_dict.
    state_dict : OrderedDict
        Weights.
    strict : bool
        whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
    logger : :obj:`logging.Logger`, optional
        Logger to log the error message.
        If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        if prefix == 'backbone.':
            _all_prefix = [pre.split('.')[0] for pre in state_dict.keys()]
            if 'backbone' not in _all_prefix:
                prefix = '.'.join(prefix.split('.')[1:])
        module._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')


def _load_checkpoint(filename, map_location=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Parameters
    ----------
    filename : str
        Accept local filepath, URL, ``torchvision://xxx``.
    map_location : str, optional
        Same as :func:`torch.load`. Default: None.

    Returns
    -------
    checkpoint: {dict, OrderedDict}
        The loaded checkpoint. It can be either an OrderedDict storing model weights
        or a dict containing other information, which depends on the checkpoint.
    """
    if filename.startswith('torchvision://'):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    else:
        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Parameters
    ----------
    model : Module
        Module to load checkpoint.
    filename : str
        Accept local filepath, URL, ``torchvision://xxx``.
    map_location : str
        Same as :func:`torch.load`.
    strict : bool
        Whether to allow different params for the model and checkpoint.
    logger : :mod:`logging.Logger`, optional
        The logger for error message.

    Returns
    -------
    checkpoint : dict or OrderedDict
        The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Parameters
    ----------
    state_dict : OrderedDict
        Model weights on GPU.

    Returns
    -------
    state_dict : OrderedDict
        Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def _save_to_state_dict(module, destination, prefix, keep_vars):
    """Saves module state to `destination` dictionary.

    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Parameters
    ----------
    module : nn.Module
        The module to generate state_dict.
    destination : dict
        A dict where state will be stored.
    prefix : str
        The prefix for parameters and buffers used in this module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix='', keep_vars=False):
    """Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.

    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).

    Parameters
    ----------
    module : nn.Module
        The module to generate state_dict.
    destination : OrderedDict
        Returned dict for the state of the module.
    prefix : str
        Prefix of the key.
    keep_vars : bool
        Whether to keep the variable property of the parameters.
        Default: False.

    Returns
    -------
    destination: dict
        A dictionary containing a whole state of the module.
    """

    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(
        version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + '.', keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Parameters
    ----------
    model : Module
        Module whose params are to be saved.
    filename : str
        Checkpoint filename.
    optimizer : :obj:`Optimizer`, optional
        Optimizer to be saved.
    meta : dict, optional
        Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(base_version=BASE_VERSION, time=time.asctime())

    mkdir_or_exist(osp.dirname(filename))
    if is_module_wrapper(model):
        model = model.module

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model))
    }
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()
    with open(filename, 'wb') as f:
        if torch.__version__ >= '1.6':
            torch.save(checkpoint, f, _use_new_zipfile_serialization=False)
        else:
            torch.save(checkpoint, f)
        f.flush()
