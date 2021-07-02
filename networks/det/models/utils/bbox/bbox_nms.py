import torch
from torchvision.ops import nms
from .transforms import xywh2xyxy


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=(), score_factors=None):
    """Performs Non-Maximum Suppression (NMS) on inference results by yolo

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        if score_factors is not None:
            # expand the shape to match original shape of score
            score_factors = score_factors.view(-1, 1).expand(
                boxes.shape[0], nc)
            score_factors = score_factors.reshape(-1)
            scores = scores * score_factors

        i = nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output



def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   iou_thres,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Labels are 0-based.
    """

    num_classes = multi_scores.shape[-1] - 1
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]
    # create labels by scores
    labels = torch.arange(num_classes, dtype=torch.long, device= scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr

        # multiply score_factor after threshold to preserve more bboxes, improve
        # mAP by 1% for YOLOv3
        if score_factors is not None:
            # expand the shape to match original shape of score
            score_factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes)
            score_factors = score_factors.reshape(-1)
            scores = scores * score_factors

        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    else:
        raise RuntimeError('[ONNX Error] Can not record NMS '
                           'as it has not been executed this time')

    if bboxes.numel() == 0:
        # return torch.cat([bboxes, labels, scores], dim=-1)
        return torch.zeros([1, max_num, 6], device=multi_scores.device)
    i = nms(bboxes, scores, iou_thres)  # NMS

    if i.shape[0] > max_num:  # limit detections
        i = i[:max_num]
        bboxes = bboxes[i]
        scores = scores[i]
        labels = labels[i] + 1

        labels = labels.unsqueeze(dim=-1)
        scores = scores.unsqueeze(dim=-1)
        result = torch.cat([bboxes, labels, scores], dim=-1).unsqueeze(dim=0)
    else:
        bboxes = bboxes[i]
        scores = scores[i]
        labels = labels[i] + 1
        labels = labels.unsqueeze(dim=-1)
        scores = scores.unsqueeze(dim=-1)
        result = torch.cat([bboxes, labels, scores], dim=-1).unsqueeze(dim=0)

        padding_num = max_num - i.shape[0]
        zero = torch.zeros([1, padding_num, 6])
        zero = bboxes.new_tensor(zero,device=bboxes.device)
        zero[..., -2] = -1      # no label
        result = torch.cat([result, zero], dim=1)

    return result

