import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt
from .utils import *

# ---------------------------------------- CrossEntropy base------------------------------------------
def cross_entropy(pred, label, weight=None, class_weight=None, reduction='mean', avg_factor=None, ignore_index=-100, **kwargs):
    if ignore_index is None:
        ignore_index = -100
    loss = F.cross_entropy(
        pred,
        label.long(),
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    if weight is not None:
        weight = weight.float()

    return  weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)


def focal_loss_with_logits(pred, label, alpha, gamma, weight=None, reduction='mean', avg_factor=None, ignore_index=-100, **kwargs):
    if pred.dim() != label.dim():
        label, _ = expand_onehot_labels(label, pred.shape, ignore_index)

    logpt = F.binary_cross_entropy_with_logits(
        pred, label.float(), reduction='none')

    pt = torch.exp(-logpt)
    focal_term = (1.0 - pt).pow(gamma)

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * label + (1 - alpha) * (1 - label)
    return  weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)


def softmax_focal_loss_with_logits(pred, label, gamma, weight=None, reduction='mean', avg_factor=None, ignore_index=-100, **kwargs):

    log_softmax = F.log_softmax(pred, dim=1)

    loss = F.nll_loss(log_softmax, label, reduction="none", ignore_index=ignore_index)
    pt = torch.exp(-loss)
    focal_term = (1.0 - pt).pow(gamma)
    loss = focal_term * loss
    return  weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)


def topk_loss(pred, label, k, weight=None, class_weight=None, reduction='mean', avg_factor=None, ignore_index=-100, **kwargs):
    if ignore_index is None:
        ignore_index = -100
    loss = F.cross_entropy(
        pred,
        label.long(),
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    num_voxels = np.prod(loss.shape, dtype=np.int64)
    loss, _ = torch.topk(loss.view((-1,)), int(num_voxels * k / 100), sorted=False)

    if weight is not None:
        weight = weight.float()

    return  weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)


def asymmetric_loss(pred, label, gamma_pos, gamma_neg, eps=0, weight=None, reduction='mean', avg_factor=None, ignore_index=-100, **kwargs):
    num_classes = pred.shape[1]
    log_preds = F.log_softmax(pred, dim=1)
    if pred.dim() != label.dim():
        targets_classes, _ = expand_onehot_labels(label, pred.shape, ignore_index)
    else:
        targets_classes = label

    targets = targets_classes
    anti_targets = 1 - targets
    xs_pos = torch.exp(log_preds)
    xs_neg = 1 - xs_pos
    xs_pos = xs_pos * targets
    xs_neg = xs_neg * anti_targets
    asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                             gamma_pos * targets + gamma_neg * anti_targets)

    log_preds = log_preds * asymmetric_w

    if eps > 0:  # label smoothing
        targets_classes.mul_(1 - eps).add_(eps / num_classes)

    # loss calculation
    loss = - targets_classes.mul(log_preds)

    return weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

# ----------------------------------------Dice base----------------------------------------
def dice_loss(pred, target, use_log=False, smooth=1, exponent=2, class_weight=None,  weight=None, reduction='mean', avg_factor=None, ignore_index=-100, **kwargs):
    assert pred.shape[0] == target.shape[0]
    pred = F.softmax(pred, dim=1)
    target, valid_mask = expand_onehot_labels(target.long(), pred.shape, ignore_index)

    total_loss = 0
    num_classes = pred.shape[1]
    for i in range(num_classes):
        if i != ignore_index:
            dice_loss = binary_dice_loss(
                pred[:, i],
                target[:, i],
                use_log=use_log,
                valid_mask=valid_mask,
                smooth=smooth,
                exponent=exponent)
            if class_weight is not None:
                dice_loss *= class_weight[i]
            total_loss += dice_loss

    loss = total_loss / num_classes
    return weight_reduce_loss(loss, weight=None, reduction=reduction, avg_factor=avg_factor)


def binary_dice_loss(pred, target, valid_mask, smooth=1, exponent=2, use_log=False, **kwards):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(pred.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    scores = num / den
    return (1 - scores) if not use_log else -torch.log(scores.clamp_min(1e-8))


def noise_robust_dice_loss(pred, target, gamma=1.5, weight=None, reduction='mean', avg_factor=None, ignore_index=-100, **kwargs):
    if pred.dim() != target.dim():
        target, valid_mask = expand_onehot_labels(target.long(), pred.shape, ignore_index)

    predict = F.softmax(pred, dim=1)

    num_class  = predict.shape[1]
    if predict.dim() == 4:
        predict = predict.permute(0, 2, 3, 1)
    if target.dim()==4:
        target = target.permute(0, 2, 3, 1)

    predict = torch.reshape(predict, (-1, num_class))
    soft_y = torch.reshape(target, (-1, num_class))

    numerator = torch.abs(predict - soft_y)
    numerator = torch.pow(numerator, gamma)
    numerator = torch.sum(numerator, dim=0)
    y_vol = torch.sum(soft_y, dim=0)
    p_vol = torch.sum(predict, dim=0)
    loss = (numerator + 1e-5) / (y_vol + p_vol + 1e-5)
    return weight_reduce_loss(loss, None, reduction, avg_factor)


# ----------------------------------------Boundary base----------------------------------------
def boundary_loss(pred, target, theta0, theta, weight=None, reduction='mean', avg_factor=None, ignore_index=-100, **kwargs):
    n, c, _, _ = pred.shape
    pred = F.softmax(pred, dim=1)
    target, valid_mask = expand_onehot_labels(target.long(), pred.shape, ignore_index)
    target = target.float()
    # boundary map
    gt_b = F.max_pool2d(1-target, kernel_size=theta0, stride=1, padding=(theta0-1) // 2)
    gt_b = gt_b - (1-target)

    pred_b = F.max_pool2d(
        1 - pred, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    pred_b -= 1 - pred

    # extended boundary map
    gt_b_ext = F.max_pool2d(
        gt_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)

    pred_b_ext = F.max_pool2d(
        pred_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)

    # reshape
    gt_b = gt_b.view(n, c, -1)
    pred_b = pred_b.view(n, c, -1)
    gt_b_ext = gt_b_ext.view(n, c, -1)
    pred_b_ext = pred_b_ext.view(n, c, -1)

    # precision
    p = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
    # recall
    r = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

    # Boundary F1 Score
    BF1 = 2 * p * r / (p + r + 1e-7)

    loss = (1-BF1)
    return  weight_reduce_loss(loss, weight=None, reduction=reduction, avg_factor=avg_factor)


# --------------------------------------------Distance Base----------------------------------------------------------
def smooth_l1_loss(pred, target, weight, beta=1.0, ignore_label=-100, reduction='mean', avg_factor=None):

    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    if pred.dim() != target.dim():
        target, valid_mask = expand_onehot_labels(target.long(), pred.shape, ignore_label)

    if pred.shape[1] > 1:
        # multi classes
        pred = F.softmax(pred, dim=1)
    else:
        # one classes
        pred = F.sigmoid(pred).exp()

    # pred = get_region_proportion(pred, valid_mask)
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)

    return weight_reduce_loss(loss, weight, reduction, avg_factor)


def gaussian_transform(batch_mask, gamma=1):

    c, h, w = batch_mask.shape
    dst_trf = torch.zeros_like(batch_mask)
    np_mask = batch_mask.cpu().numpy()

    for b, mask in enumerate(np_mask):
        num_labels, labels = cv2.connectedComponents((mask * 255.0).astype(np.uint8), connectivity=8)
        for idx in range(1, num_labels):
            mask_roi = np.zeros((h, w))
            k = labels == idx
            mask_roi[k] = 1
            dst_trf_roi = cv2.GaussianBlur(mask_roi, (3, 3), gamma) + 1
            dst_trf[b] += torch.tensor(dst_trf_roi,dtype=torch.float32, device=batch_mask.device)

    return dst_trf