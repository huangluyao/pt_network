from .base_loss import BaseLoss
from .functional import *
from ..builder import LOSSES

@LOSSES.register_module()
class SmoothL1Loss(BaseLoss):
    def __init__(self, beta=1.0, **kwargs):
        super(SmoothL1Loss, self).__init__(loss_name='loss_l1' ,**kwargs)
        self.beta = beta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            ignore_label=self.ignore_label,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss
