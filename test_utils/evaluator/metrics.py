# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/16 上午10:07
import numpy as np
from .boxes_utils import compute_boxes_iou

def calculate_tps_fps(y_score, y_true, pos_label=1):

    y_true = (y_true == pos_label)
    desc_score_indices = np.argsort(y_score, kind='quicksort')[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    distinct_value_indices = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true)[distinct_value_indices]
    fps = 1 + distinct_value_indices - tps
    return fps, tps


def calculate_auc_aupr(y_score, y_true, pos_label=None):
    fps, tps = calculate_tps_fps(y_score, y_true, pos_label=pos_label)
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    fpr = np.r_[0, fpr]
    tpr = np.r_[0, tpr]
    AUC = np.sum((tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1])) / 2.
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tpr[1:]
    last_ind = tps.searchsorted(tps[-1])
    slice_pos = slice(0, last_ind + 1)
    recall = np.r_[0, recall[slice_pos]]
    precision = np.r_[1, precision[slice_pos]]
    AUPR = np.sum(np.diff(recall) * precision[1:])

    return np.round(AUC, 4), np.round(AUPR, 4)


def calcuate_auc_aupr_ks_bestf1(y_score, y_true, pos_label=1):

    fps, tps = calculate_tps_fps(y_score, y_true, pos_label)
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    fpr = np.r_[0, fpr]
    tpr = np.r_[0, tpr]
    AUC = np.sum((tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1])) / 2.
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tpr[1:]
    last_ind = tps.searchsorted(tps[-1])
    slice_pos = slice(0, last_ind + 1)
    recall = np.r_[0, recall[slice_pos]]
    precision = np.r_[1, precision[slice_pos]]
    AUPR = np.sum(np.diff(recall) * precision[1:])

    return np.round(AUC, 4), np.round(AUPR, 4)


def calculate_auc_aupr_ks_bestf1(y_score, y_true, pos_label=None):
    fps, tps = calculate_tps_fps(y_score, y_true, pos_label=pos_label)
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    fpr = np.r_[0, fpr]
    tpr = np.r_[0, tpr]
    AUC = np.sum((tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1])) / 2.
    diff_tpr_fpr = tpr - fpr
    ks_value = np.max(diff_tpr_fpr)
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tpr[1:]
    last_ind = tps.searchsorted(tps[-1])
    slice_pos = slice(0, last_ind + 1)
    recall = np.r_[0, recall[slice_pos]]
    precision = np.r_[1, precision[slice_pos]]
    AUPR = np.sum(np.diff(recall) * precision[1:])
    f1 = 2*precision*recall / (precision + recall)
    bestf1 = np.max(f1)
    if np.isnan(bestf1):
        bestf1 = 0

    return np.round(AUC, 4), np.round(AUPR, 4), np.round(ks_value, 4), np.round(bestf1, 4)


def calculate_precision_recall_f1(y_pred, y_true):
    TP = np.sum(y_pred * y_true)
    FP = np.sum(y_pred * (1-y_true))
    FN = np.sum((1 - y_pred) * y_true)

    Precision = TP / (TP + FP + 1e-7)
    Recall = TP / (TP + FN + 1e-7)

    F1 = 2*Precision*Recall / (Precision + Recall)
    if np.isnan(F1):
        F1 = 0
    return np.round(Precision, 4), np.round(Recall, 4), np.round(F1, 4)


def calculate_precision_recall_f1_per_class(y_pred, y_true, num_classes):
    per_class_metrics = []
    for class_id in range(num_classes):
        P, R, F1 = calculate_precision_recall_f1((y_pred==class_id), (y_true==class_id))
        per_class_metrics.append([P, R, F1])

    per_class_p, per_class_R, per_class_F1= [
        np.stack(metrics) for metrics in zip(*per_class_metrics)
    ]
    return per_class_p, per_class_R, per_class_F1


def calculate_iou(y_pred, y_true):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    iou = intersection / (union + 1e-7)
    return np.round(iou, 4)


def calculate_iou_per_class(y_pred, y_true, num_classes):
    iou_per_class = []
    for class_id in range(num_classes):
        iou = calculate_iou((y_pred==class_id), (y_true==class_id))
        iou_per_class.append(iou)

    return np.round(np.stack(iou_per_class), 4)

def compare_features(y_pred):

    assert y_pred.shape[1] == 2, 'compare features y_pred.shape must be 2 at dim=1 '

    vector_a = y_pred[:,0,:]
    vector_b = y_pred[:,1,:]
    vector_sum = np.sum(vector_a * vector_b, axis=1)

    denom = np.linalg.norm(vector_a,axis=1) * np.linalg.norm(vector_b, axis=1)
    cos = vector_sum / denom
    return cos

def calculate_compare_precision_recall_f1(y_pred, y_true, threshold=0.8):
    cos_smi = compare_features(y_pred)
    P, R, F1, acc = calculate_precision_recall_f1(cos_smi>threshold, y_true)
    return P, R, F1, acc

def calculate_AP_of_detections_per_class(detections_list, gt_boxes_list, class_ids, iou_threshold=0.5, confidence_threshold=0.5):
    num_images = len(detections_list)
    AP_of_classes = dict()

    for class_id in class_ids:
        positives_list = []
        detected_list = []
        confidences_list = []
        box_beg_end = []
        cur_box_idx = 0
        gt_beg_end = []
        cur_gt_idx = 0
        for idx in range(num_images):
            detections = detections_list[idx]
            gt_boxes = gt_boxes_list[idx]
            det_class_spec_indices = np.where(detections[:, 4]==class_id)[0]
            gt_class_spec_indices = np.where(gt_boxes[:, 4]==class_id-1)[0]
            positives, detected, confidences = calculate_positives_detected_of_detections(
                detections[det_class_spec_indices], gt_boxes[gt_class_spec_indices], iou_threshold=iou_threshold
            )
            positives_list.append(positives)
            detected_list.append(detected)
            confidences_list.append(confidences)
            try:
                object_numbers = detected.shape[1]
            except:
                object_numbers = 0

            gt_beg_end.append((cur_gt_idx, cur_gt_idx + object_numbers))
            cur_gt_idx += object_numbers
            box_beg_end.append((cur_box_idx, cur_box_idx + len(confidences)))
            cur_box_idx += len(confidences)
        all_positives = np.concatenate(positives_list, axis=0)
        all_confidences = np.concatenate(confidences_list, axis=0)
        all_detected = np.zeros((len(all_positives), gt_beg_end[-1][1]))
        for idx in range(num_images):
            cur_box_beg_end = box_beg_end[idx]
            cur_gt_beg_end = gt_beg_end[idx]
            if len(detected_list[idx].shape) ==1:
                detected_list[idx] = np.expand_dims(detected_list[idx],axis=-1)
            all_detected[cur_box_beg_end[0]:cur_box_beg_end[1], cur_gt_beg_end[0]:cur_gt_beg_end[1]] = detected_list[idx]
        if confidence_threshold:
            conf_indices = np.where(all_confidences >= confidence_threshold)[0]
            all_confidences = all_confidences[conf_indices]
            all_positives = all_positives[conf_indices]
            all_detected = all_detected[conf_indices, :]
        if len(all_confidences) == 0:
            AP_of_classes[class_id] = 0.0
            continue
        sorted_indices = np.argsort(-all_confidences)
        all_confidences = all_confidences[sorted_indices]
        all_positives = all_positives[sorted_indices]
        all_detected = all_detected[sorted_indices]
        distinct_value_indices = np.where(np.diff(all_confidences))[0]
        distinct_value_indices = np.r_[distinct_value_indices, all_confidences.size - 1]
        tps = np.cumsum(all_positives)[distinct_value_indices]
        fps = 1 + distinct_value_indices - tps
        num_detected = np.cumsum(all_detected, axis=0)
        undetected = np.zeros(num_detected.shape)
        undetected[num_detected == 0] = 1
        fns = np.sum(undetected, axis=1)
        fns = fns[distinct_value_indices]
        precision = tps / (tps + fps)
        precision[np.isnan(precision)] = 0
        recall = tps / (tps + fns)
        last_ind = tps.searchsorted(tps[-1])
        slice_pos = slice(0, last_ind + 1)
        recall = np.r_[0, recall[slice_pos]]
        precision = np.r_[1, precision[slice_pos]]
        AP = np.sum(np.diff(recall) * precision[1:])
        AP_of_classes[class_id] = AP

    return AP_of_classes

def calculate_positives_detected_of_detections(detections, gt_boxes, iou_threshold=0.5):
    confidences = detections[:, 5]
    sorted_indices = np.argsort(-confidences)
    confidences = detections[sorted_indices, 5]
    if len(gt_boxes) == 0:
        return np.zeros(detections.shape[0]), np.zeros(detections.shape[0]), confidences
    boxes = detections[sorted_indices, :4]
    overlaps = compute_boxes_iou(boxes, gt_boxes[:, :4])
    box_max_overlaps = np.amax(overlaps, axis=1)
    positives = np.zeros(detections.shape[0])
    positives_indices = np.where(box_max_overlaps>iou_threshold)[0]
    positives[positives_indices] = 1
    detected = np.zeros(overlaps.shape)
    gt_detected_indices = np.where(overlaps>iou_threshold)
    detected[gt_detected_indices] = 1
    return positives, detected, confidences


def model_info(model, img_size=640):

    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients

    try:  # FLOPS
        import torch
        from thop import profile
        from copy import deepcopy
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        img = torch.zeros((1, 3, img_size[0], img_size[1]), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
        fs = ', %.1f GFLOPS' % (flops)
    except (ImportError, Exception):
        fs = ''

    return f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, size {(n_p * 4 / 1024) /1024:.2f} (MB) {fs}"
