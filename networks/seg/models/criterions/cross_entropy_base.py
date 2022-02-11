from .base_loss import BaseLoss
from .functional import *
from ..builder import LOSSES

@LOSSES.register_module()
class CrossEntropyLoss(BaseLoss):

    def __init__(self, class_weight=None, ignore_label=-100, use_sigmoid=False, label_smoothing=0.0, **kwargs):
        super(CrossEntropyLoss, self).__init__(loss_name='loss_ce', **kwargs)
        self.class_weight = class_weight
        self.ignore_label = ignore_label
        self.use_sigmoid = use_sigmoid
        self.label_smoothing = label_smoothing

        if use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_label = None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        if ignore_label is None:
            ignore_label = self.ignore_label

        if self.use_sigmoid:
            loss_cls = self.loss_weight * self.cls_criterion(
                pred,
                target,
                weight=weight,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                ignore_index=ignore_label)
        else:
            loss_cls = self.loss_weight * self.cls_criterion(
                pred,
                target,
                weight=weight,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                ignore_index=ignore_label,
                label_smoothing=self.label_smoothing)

        return loss_cls

@LOSSES.register_module()
class FocalLoss(BaseLoss):
    def __init__(self, multiclass=True, alpha=0.25, gamma=2.0,  **kwargs):
        super(FocalLoss, self).__init__(loss_name='loss_focal_loss',**kwargs)
        self.alpha = alpha
        self.gamma = gamma

        if not multiclass:
            self.criterion = focal_loss_with_logits
        else:
            self.criterion = softmax_focal_loss_with_logits


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
        loss_cls = self.loss_weight * self.criterion(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_label)

        return loss_cls

@LOSSES.register_module()
class TopKLoss(BaseLoss):
    def __init__(self, k=10, class_weight=None, ignore_label=-100,**kwargs):
        self.class_weight = class_weight
        self.ignore_label = ignore_label
        self.k = k
        super(TopKLoss, self).__init__(loss_name='loss_topk',**kwargs)

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_label = None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        if ignore_label is None:
            ignore_label = self.ignore_label

        loss_cls = self.loss_weight * topk_loss(
            pred,
            target,
            k=self.k,
            weight=weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_label)

        return loss_cls

@LOSSES.register_module()
class AsymmetricLoss(BaseLoss):

    def __init__(self,eps=0, gamma_pos=0, gamma_neg=4,  **kwargs):
        super(AsymmetricLoss, self).__init__(loss_name='loss_asl' ,**kwargs)
        self.eps = eps
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

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

        loss_cls = self.loss_weight * asymmetric_loss(
            pred,
            target,
            gamma_pos=self.gamma_pos,
            gamma_neg=self.gamma_neg,
            eps=self.eps,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_label)

        return loss_cls


@LOSSES.register_module()
class GaussianCrossEntropyLoss(BaseLoss):

    def __init__(self, class_weight=None, gamma=1, ignore_label=-100, **kwargs):
        super(GaussianCrossEntropyLoss, self).__init__(loss_name='loss_gaussian_ce', **kwargs)
        self.class_weight = class_weight
        self.ignore_label = ignore_label
        self.gamma = gamma
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_label = None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        if ignore_label is None:
            ignore_label = self.ignore_label

        loss_cls = self.loss_weight * cross_entropy(
            pred,
            target,
            weight=gaussian_transform(target, self.gamma),
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_label)

        return loss_cls
