# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/21 下午6:01

import cv2
import math
import os
import numpy as np

class Visualizer:

    def __init__(self, task, class_names, input_size, vis_score_threshold=None):
        self._task = task
        self._class_names = class_names
        self._input_size = input_size
        self._vis_score_threshold = vis_score_threshold

        if self._task == 'classification':
            self.visualize = self._visualize_cls
        elif self._task == 'segmentation':
            self.visualize = self._visualize_seg
        elif self._task == 'detection':
            self.visualize = self._visualize_det
        else:
            raise ValueError('Not support task %s' %self._task)

    def _visualize_cls(self, img_paths, predictions, vis_predictions_dir):
        num_preds = len(img_paths)
        if not os.path.exists(vis_predictions_dir):
            os.makedirs(vis_predictions_dir)
        vis_paths = [os.path.join(vis_predictions_dir, 'pred_%03d.png'%(idx)) for idx in range(num_preds)]
        for idx in range(num_preds):
            ori_image = cv2.imread(img_paths[idx])
            index = np.argmax(predictions[idx])
            result_txt = "%s: %.4f" % (self._class_names[index], predictions[idx][index])

            ((text_width, text_height), _) = cv2.getTextSize(result_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            cv2.putText(ori_image, result_txt, (10, text_height+10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 0))

            cv2.imwrite(vis_paths[idx], ori_image)

    def _visualize_det(self, img_paths, predictions, vis_predictions_dir, image_names=None):
        num_preds = len(img_paths)
        if not os.path.exists(vis_predictions_dir):
            os.makedirs(vis_predictions_dir)
        if image_names is None:
            vis_paths = [os.path.join(vis_predictions_dir, 'pred_%03d.png'%(idx)) for idx in range(num_preds)]

        for idx in range(num_preds):
            ori_image = cv2.imread(img_paths[idx])
            ori_size = ori_image.shape[:2]
            boxes = predictions[idx]
            boxes = resize_box(ori_size , (self._input_size[1], self._input_size[0]), boxes)
            vis_result = show_detections(ori_image, boxes, self._class_names, self._vis_score_threshold)
            cv2.imwrite(vis_paths[idx], vis_result)

    def _visualize_seg(self, img_paths, predictions, vis_predictions_dir):
        num_preds = len(img_paths)
        num_classes = len(self._class_names)
        if not os.path.exists(vis_predictions_dir):
            os.makedirs(vis_predictions_dir)
        vis_paths = [os.path.join(vis_predictions_dir, 'pred_%03d.png'%(idx)) for idx in range(num_preds)]

        bgr_values = get_BGR_values(num_classes)

        for idx in range(num_preds):
            ori_image = cv2.imread(img_paths[idx])
            ori_size = ori_image.shape[:2]
            image = cv2.resize(ori_image, (self._input_size[1], self._input_size[0]))
            mask = get_pred(predictions[idx], self._vis_score_threshold)
            image, mask = resize_mask(image, mask, ori_size)
            vis_preds = np.zeros_like(ori_image)
            for class_id in range(1, num_classes+1):
                vis_preds[:,:,0][mask==class_id] = bgr_values[class_id-1][0]
                vis_preds[:,:,1][mask==class_id] = bgr_values[class_id-1][1]
                vis_preds[:,:,2][mask==class_id] = bgr_values[class_id-1][2]

            mask = (mask==0)[:,:,None]
            vis_preds = (ori_image*mask+(1-mask)*(vis_preds*0.5+ori_image*0.5)).astype(np.uint8)
            vis_result = np.hstack((ori_image, vis_preds))
            cv2.imwrite(vis_paths[idx], vis_result)



def resize_box(org_size, inptut_size, boxes):
    input_h, input_w = inptut_size
    im_h, im_w = org_size
    resize_ratio = min(input_w / im_w, input_h / im_h)
    new_w = round(im_w * resize_ratio)
    new_h = round(im_h * resize_ratio)

    del_h = (input_h - new_h)/2
    del_w = (input_w - new_w)/2
    boxes = boxes[(boxes[:,2]>0) * (boxes[:,3] > 0)]

    boxes[:,0:4:2] = boxes[:,0:4:2] - del_w
    boxes[:,1:4:2] = boxes[:,1:4:2] - del_h
    boxes[:,:4] /= resize_ratio

    return boxes


def resize_mask(image, mask, size):
    resize_h, resize_w = size
    im_h, im_w, im_c = image.shape
    resize_ratio = min(resize_w / im_w, resize_h / im_h)
    new_w = round(im_w * resize_ratio)
    new_h = round(im_h * resize_ratio)
    im_resized = cv2.resize(image, (new_w, new_h))
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    im_padding = np.full([resize_h, resize_w, im_c], 0)
    im_padding[(resize_h-new_h)//2:(resize_h-new_h)//2 + new_h, (resize_w-new_w)//2:(resize_w-new_w)//2 + new_w,  :] = im_resized
    mask_padding = np.full([resize_h, resize_w], 0)
    mask_padding[(resize_h-new_h)//2:(resize_h-new_h)//2 + new_h, (resize_w-new_w)//2:(resize_w-new_w)//2 + new_w] = mask_resized
    # im_padding = cv2.resize(image, (resize_w, resize_h))
    # mask_padding = cv2.resize(mask, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
    return im_padding, mask_padding


def show_detections(image, detections, class_names, score_threshold=None):
    detections = detections[~np.all(detections==0, axis=1)]
    if score_threshold is not None:
        indices = np.where(detections[:, -1]>=score_threshold)[0]
        detections = detections[indices, :]
    class_ids = np.unique(detections[:, -2])
    class_ids = class_ids[class_ids>0].astype(np.int32)
    if len(class_ids)==0:
        return image
    bgr_values = get_BGR_values(len(class_names))
    # class_color_dict = dict(zip(class_ids, bgr_values))

    img_to_draw = image.copy()
    num_boxes = detections.shape[0]
    for i in range(num_boxes):
        class_id = int(detections[i][-2])
        if class_id == 0:
            continue
        confidence_score = detections[i][-1]
        box = np.array(detections[i][:-2], np.int32)
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(
            img_to_draw,
            (box[0], box[1]),
            (box[2], box[3]),
            bgr_values[class_id-1],
            2
        )

        result_txt = "%s: %.4f"%(class_names[class_id-1], confidence_score)

        ((text_width, text_height), _) = cv2.getTextSize(result_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img_to_draw, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), (255,255,255), -1)
        cv2.putText(
            img_to_draw,
            text=result_txt,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=bgr_values[class_id-1],
            lineType=cv2.LINE_AA,
        )
    return img_to_draw


_RGB_LIST = [
    [255,215,0],
    [0,255,255],
    [255,0,255],
    [148,0,211],
    [0,191,255],
    [255,69,0],
    [255,105,180],
    [147,112,219],
    [220,20,60],
    [46,139,87],
    [218,165,32],
    [139,0,0],
    [0,0,128],
    [184,134,11],
    [128,128,0],
    [0,139,139],
    [255,140,0],
    [128,0,128],
    [255,0,0],
    [0,0,255],
    [255,255,0],
    [0,255,0]
]


def get_BGR_values(n_colors):
    if n_colors <= len(_RGB_LIST):
        bgr_list = [(v[2], v[1], v[0]) for v in _RGB_LIST[0:n_colors]]
    else:
        multiple = int(math.ceil(n_colors // len(_RGB_LIST)))
        rgb_list = _RGB_LIST * multiple
        bgr_list = [(v[2], v[1], v[0]) for v in rgb_list[0:n_colors]]
    return bgr_list

def get_pred(y_prob, threshold=None):
    if threshold is not None:
        y_pred = (y_prob>=threshold)*1
    else:
        y_pred = np.argmax(y_prob, axis=-1)

    if y_pred.ndim > 2:
        y_pred = np.squeeze(y_pred, axis=-1)

    return y_pred