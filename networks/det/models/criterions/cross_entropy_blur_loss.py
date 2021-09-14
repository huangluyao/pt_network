# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/8/11 下午3:53
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import weight_reduce_loss
from .cross_entropy_loss import _expand_onehot_labels
from networks.det.models.builder import LOSSES


def cross_entropy_blur_loss(pred, label, alpha,
                             weight=None,
                             reduction='mean',
                             avg_factor=None,
                             class_weight=None):
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    pred = torch.sigmoid(pred)  # prob from logits
    dx = pred - label  # reduce only missing label effects
    # dx = (pred - true).abs()  # reduce missing label and false label effects
    alpha_factor = 1 - torch.exp((dx - 1) / (alpha + 1e-4))
    loss *= alpha_factor
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class CrossEntropyBlurLoss(nn.Module):
    def __init__(self,
                 alpha=0.05,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 **kwargs):
        """CrossEntropyLoss.

        Parameters
        ----------
        use_sigmoid : bool, optional
            Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask : bool, optional
            Whether to use mask cross entropy loss.
            Defaults to False.
        reduction : str, optional
            . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight : list[float], optional
            Weight of each class.
            Defaults to None.
        loss_weight : float, optional
            Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyBlurLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.alpha=alpha

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Parameters
        ----------
        cls_score : torch.Tensor
            The prediction.
        label : torch.Tensor
            The learning label of the prediction.
        weight : torch.Tensor, optional
            Sample-wise loss weight.
        avg_factor : int, optional
            Average factor that is used to average
            the loss. Defaults to None.
        reduction : str, optional
            The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        Returns
        -------
        torch.Tensor
            The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * cross_entropy_blur_loss(
            cls_score,
            label,
            self.alpha,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
