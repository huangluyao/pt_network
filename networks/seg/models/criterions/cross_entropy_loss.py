import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100):
    """The wrapper function for :func:`F.cross_entropy`"""
    if ignore_index is None:
        ignore_index = -100
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Parameters
    ----------
    pred : torch.Tensor
        The prediction with shape (N, 1).
    label : torch.Tensor
        The learning label of the prediction.
    weight : torch.Tensor, optional
        Sample-wise loss weight.
    reduction : str, optional
        The method used to reduce the loss.
        Options are "none", "mean" and "sum".
    avg_factor : int, optional
        Average factor that is used to average
        the loss. Defaults to None.
    class_weight : list[float], optional
        The weight for each class.

    Returns
    -------
    torch.Tensor
        The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight=class_weight, reduction='none')
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None):
    """Calculate the CrossEntropy loss for masks.

    Parameters
    ----------
    pred : torch.Tensor
        The prediction with shape (N, C), C is the number
        of classes.
    target : torch.Tensor
        The learning label of the prediction.
    label : torch.Tensor
        ``label`` indicates the class label of the mask'
        corresponding object. This will be used to select the mask in the
        of the class which the object belongs to when the mask prediction
        if not class-agnostic.
    reduction : str, optional
        The method used to reduce the loss.
        Options are "none", "mean" and "sum".
    avg_factor : int, optional
        Average factor that is used to average
        the loss. Defaults to None.
    class_weight : list[float], optional
        The weight for each class.

    Returns
    -------
    torch.Tensor
        The calculated loss
    """
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


def flatten_binary_scores(scores, labels, ignore=None):
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(nn.Module):
    def __init__(self, **kwargs):
        super(StableBCELoss, self).__init__()

    def forward(self, pred, label, ignore=None, **kwargs):
        pred, label = flatten_binary_scores(pred, label, ignore)
        neg_abs = - pred.abs()
        loss = pred.clamp(min=0) - pred * Variable(label.float()) \
               + (1 + neg_abs.exp()).log()
        return loss.mean()


class CrossEntropyLoss(nn.Module):
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

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_label=None):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_label = ignore_label

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                pred,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            pred,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=self.ignore_label)

        return loss_cls


class TopKLoss(CrossEntropyLoss):
    """
    Network has to have NO LINEARITY!
    """
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 class_weight=None,
                 loss_weight=1.0,
                 k=10):
        self.k = k
        super(TopKLoss, self).__init__(
            use_sigmoid=use_sigmoid,
            use_mask=use_mask,
            class_weight=class_weight,
            loss_weight=loss_weight,
            reduction='none')

    def forward(self,
                pred,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        ce_loss = super(TopKLoss, self).forward(pred, label)
        num_voxels = np.prod(ce_loss.shape, dtype=np.int64)
        loss, _ = torch.topk(ce_loss.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return loss.mean()
