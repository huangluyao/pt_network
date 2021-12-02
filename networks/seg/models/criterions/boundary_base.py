from .functional import *
from .base_loss import BaseLoss
from ..builder import LOSSES

@LOSSES.register_module()
class BoundaryLoss(BaseLoss):

    def __init__(self, theta0=3, theta=5, **kwargs):
        super(BoundaryLoss, self).__init__(loss_name='loss_boundary',**kwargs)
        self.theta0=theta0
        self.theta = theta

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_label=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        if ignore_label is None:
            ignore_label = self.ignore_label

        loss_cls = self.loss_weight * boundary_loss(
            pred,
            target,
            theta0=self.theta0,
            theta=self.theta,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_label)

        return loss_cls


