import os
import torch

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
    if not os.path.isfile(filename):
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
        if type(checkpoint) == type(model):
            model_state_dict = model.state_dict()
            load_state_dict = checkpoint.state_dict()
            for key, weight in load_state_dict.items():
                if model_state_dict[key].shape == weight.shape:
                    model_state_dict[key] = weight
            model.load_state_dict(model_state_dict)
            return checkpoint
        else:
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