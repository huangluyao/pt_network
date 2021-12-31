from .base_loss import BaseLoss
from .functional import *
from ..builder import LOSSES

@LOSSES.register_module()
class DiceLoss(BaseLoss):

    def __init__(self,
                 smooth=1,
                 exponent=2,
                 use_log=False,
                 multi_class=True,
                 **kwargs):
        self.smooth = smooth
        self.exponent = exponent
        self.use_log = use_log
        self.multi_class = multi_class
        super(DiceLoss, self).__init__(loss_name='loss_dice',  **kwargs)

        if multi_class:
            self.cls_criterion = dice_loss
        else:
            self.cls_criterion = binary_dice_loss


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
        if not self.multi_class:
            pred = pred.sigmoid()

        loss = self.loss_weight * self.cls_criterion(
            pred,
            target,
            smooth=self.smooth,
            exponent=self.exponent,
            use_log=self.use_log,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_label)
        return loss

@LOSSES.register_module()
class NRDiceLoss(BaseLoss):
    def __init__(self,gamma=1.5, **kwargs):
        super(NRDiceLoss, self).__init__(loss_name='loss_nr_dice', **kwargs)
        self.gamma = gamma

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

        loss = self.loss_weight * noise_robust_dice_loss(
            pred,
            target,
            gamma=self.gamma,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_label)
        return loss

