import torch
from torch import Tensor
import torchvision


def nms(boxes: Tensor,
        scores: Tensor,
        iou_threshold: float,
        **kwargs):
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Parameters
    ----------
    boxes : Tensor[N, 4])
        boxes to perform NMS on. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping
        boxes with IoU > iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    return indices


def batched_nms(boxes: Tensor,
                scores: Tensor,
                idxs: Tensor,
                nms_cfg: dict,
                class_agnostic: bool = False):
    """Performs non-maximum suppression in a batched fashion.

    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Parameters
    ----------
    boxes : torch.Tensor
        boxes in shape (N, 4).
    scores : torch.Tensor
        scores in shape (N, ).
    idxs : torch.Tensor
        each index value correspond to a bbox cluster,
        and NMS will not be applied between elements of different idxs,
        shape (N, ).
    nms_cfg : dict
        specify nms type and other parameters like iou_thr.
        Possible keys includes the following.

        - iou_thr (float): IoU threshold used for NMS.
        - split_thr (float): threshold number of boxes. In some cases the
            number of boxes is large (e.g., 200k). To avoid OOM during
            training, the users could set `split_thr` to a small value.
            If the number of boxes is greater than the threshold, it will
            perform NMS on each group of boxes separately and sequentially.
            Defaults to 10000.
    class_agnostic : bool
        if true, nms is class agnostic, i.e.
        IoU thresholding happens over all boxes, regardless of the predicted class.

    Returns
    -------
    tuple
        kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    if len(boxes_for_nms) < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = dets[:, -1]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero().view(-1)
            keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero().view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Parameters
    ----------
    multi_bboxes : Tensor
        shape (n, #class*4) or (n, 4)
    multi_scores : Tensor
        shape (n, #class), where the last column
        contains scores of the background class, but this will be ignored.
    score_thr : float
        bbox threshold, bboxes with scores lower than it will not be considered.
    nms_thr : float
        NMS IoU threshold
    max_num : int
        if there are more than max_num bboxes after NMS,
        only top max_num will be kept.
    score_factors : Tensor
        The factors multiplied to scores before applying NMS

    Returns
    -------
    tuple
        (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    valid_mask = scores > score_thr

    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep]
