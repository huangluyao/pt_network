import functools
import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean", "sum" and "batchmean".

    Return:
        Tensor: Reduced loss tensor.
    """
    if reduction == 'batchmean':
        return loss.sum() / loss.shape[0]

    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss

    if reduction_enum == 1:
        return loss.mean()

    if reduction_enum == 2:
        return loss.sum()

    raise ValueError(f'reduction type {reduction} not supported')

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def mask_reduce_loss(loss, weight=None, reduction='mean', sample_wise=False):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            "none", "mean" and "sum". Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over masked region
    elif reduction == 'mean':
        # expand weight from N1HW to NCHW
        if weight.size(1) == 1:
            weight = weight.expand_as(loss)
        # small value to prevent division by zero
        eps = 1e-12

        # perform sample-wise mean
        if sample_wise:
            weight = weight.sum(dim=[1, 2, 3], keepdim=True)  # NCHW to N111
            loss = (loss / (weight + eps)).sum() / weight.size(0)
        # perform pixel-wise mean
        else:
            loss = loss.sum() / (weight.sum() + eps)

    return loss