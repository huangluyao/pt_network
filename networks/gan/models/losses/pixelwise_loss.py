import torch.nn as nn
import torch.nn.functional as F
from ..builder import MODULES
from .utils import mask_reduce_loss

def l1_loss(pred, target, weight, reduction, sample_wise):
    """L1 loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated L1 loss.
    """
    loss = F.l1_loss(pred, target, reduction='none')
    loss = mask_reduce_loss(loss, weight, reduction, sample_wise)

    return loss


@MODULES.register_module()
class L1Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Unsupported reduction mode: {reduction}. "
                             f"Supported ones are: {['none', 'mean', 'sum']}")

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(
                pred,
                target,
                weight,
                reduction=self.reduction,
                sample_wise=self.sample_wise)
