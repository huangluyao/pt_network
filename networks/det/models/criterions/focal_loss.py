import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from .utils import weight_reduce_loss


def onehot_target(pred, gt):
    pred_shap = pred.shape
    gt_shap = gt.shape

    with torch.no_grad():
        if len(pred_shap) != len(gt_shap):
            gt = gt.view((gt_shap[0], 1, *gt_shap[1:]))
        if all([i == j for i, j in zip(pred_shap, gt.shape)]):
            gt_onehot = gt
        else:
            gt_onehot = torch.zeros(pred_shap).to(pred.device)
            gt_onehot.scatter_(1, gt.long(), 1)

    return gt_onehot


def det_onehot_target(pred, gt):

    pred_shap = pred.shape
    gt_shap = gt.shape
    num_classes = pred_shap[-1] + 1   # add background

    with torch.no_grad():
        if len(pred_shap) != len(gt_shap):
            gt_onehot = gt.to(dtype=torch.int64)
            gt_onehot = F.one_hot(gt_onehot, num_classes=num_classes)[:, :pred_shap[-1]]
    return gt_onehot


def _sigmoid_focal_loss(input,
                        target,
                        gamma=2.0,
                        alpha=0.25,
                        reduction='mean',
                        *args,
                        **kwargs):
    """Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Parameters
    ----------
    input : A float tensor of arbitrary shape
        The predictions for each example.
    target : A float tensor with the same shape as input
        Stores the binary classification label for each element in input
        (0 for the negative class and 1 for the positive class).
    gamma : float
        Exponent of the modulating factor (1 - p_t) to
        balance easy vs hard examples.
    alpha : float, optional
        Weighting factor in range (0,1) to balance
        positive vs negative examples.
        -1 means no weighting.
    reduction : 'none' | 'mean' | 'sum'
        - 'none': No reduction will be applied to the output.
        - 'mean': The output will be averaged.
        - 'sum': The output will be summed.

    Returns
    -------
    Tensor
        Loss tensor with the reduction option applied.
    """
    # target = onehot_target(input, target)
    target = det_onehot_target(input, target)
    p = torch.sigmoid(input)
    ce_loss = F.binary_cross_entropy_with_logits(input, target.float(), reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    """A warpper _sigmoid_focal_loss

    Parameters
    ----------
    pred : torch.Tensor
        The prediction with shape (N, C), C is the number
        of classes.
    target : torch.Tensor
        The learning label of the prediction.
    weight : torch.Tensor, optional
        Sample-wise loss weight.
    gamma : float, optional
        The gamma for calculating the modulating
        factor. Defaults to 2.0.
    alpha : float, optional
        A balanced form for Focal Loss.
        Defaults to 0.25.
    reduction : str, optional
        The method used to reduce the loss into
        a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
    avg_factor : int, optional
        Average factor that is used to average
        the loss. Defaults to None.
    """
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha, None, 'none')
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
class FocalLoss(nn.Module):

    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs
                 ):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)

        loss_cls = self.loss_weight * sigmoid_focal_loss(
            pred,
            target,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor)

        return loss_cls