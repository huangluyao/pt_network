# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/8/4 上午9:42
import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from test_utils.transforms.augmentations import RandomCrop
from test_utils.datasets import DATASET


def get_mosaic_coordinate(mosaic_index, xc, yc, input_h, input_w):
    if mosaic_index == 0:
        x1, y1, x2, y2 = 0, 0, xc, yc
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, 0, input_w, yc
    elif mosaic_index == 2:
        x1, y1, x2, y2 = 0, yc, xc, input_h
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, input_w, input_h
    return x1, y1, x2, y2


@DATASET.registry()
class MosaicDetection(Dataset):

    def __init__(self, dataset, img_size):
        super(MosaicDetection, self).__init__()
        self.dataset = dataset

        if isinstance(img_size, list) or isinstance(img_size, tuple):
            self.img_size = img_size[:2]
        elif isinstance(img_size, int):
            self.img_size = [img_size, img_size]
        else:
            raise ValueError("img_size must be list, tuple or int")

        self.class_names = self.dataset.class_names
        self.max_number_object_per_img = self.dataset.max_number_object_per_img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):

        mosaic_boxes = []
        mosaic_index = []
        input_h, input_w = self.img_size

        # yc, xc = s, s  # mosaic center x, y
        yc = int(random.uniform(0.25 * input_h, 0.75 * input_h))
        xc = int(random.uniform(0.25 * input_w, 0.75 * input_w))

        indices = [item] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        for i_mosaic, index in enumerate(indices):
            img, labels = self.dataset.__getitem__(index)
            (h, w, c) = img.shape[:3]
            if i_mosaic == 0:
                mosaic_img = np.zeros_like(img)

            (x1, y1, x2, y2) = get_mosaic_coordinate(i_mosaic, xc, yc, input_h, input_w)
            crop_result = RandomCrop(y2-y1, x2-x1, always_apply=True)(image=img, **labels)

            mosaic_img[y1:y2, x1:x2] = crop_result["image"]

            bboxes = np.array(crop_result["bboxes"])
            b_x1 = bboxes[:, 0] + x1
            b_y1 = bboxes[:, 1] + y1
            b_x2 = bboxes[:, 2] + x1
            b_y2 = bboxes[:, 3] + y1

            old_bbox_area = (b_x2-b_x1) * (b_y2-b_y1)
            b_x1 = b_x1.clip(x1, x2)
            b_y1 = b_y1.clip(y1, y2)
            b_x2 = b_x2.clip(x1, x2)
            b_y2 = b_y2.clip(y1, y2)
            new_bbox_area = (b_x2-b_x1) * (b_y2-b_y1)

            radio = new_bbox_area / old_bbox_area
            bboxes = np.concatenate([b_x1[:, None], b_y1[:, None], b_x2[:, None], b_y2[:, None]], axis=-1)
            bboxes = bboxes[radio>0.5]
            indexs = np.array(crop_result["label_index"])[radio>0.5]
            for bbox, id in zip(bboxes, indexs):
                mosaic_boxes.append(bbox)
                mosaic_index.append(id)

        info_data = dict()
        info_data["bboxes"] = mosaic_boxes
        info_data["label_index"] = mosaic_index
        return mosaic_img, info_data

def draw_boxe(img, box, name):
    img_to_draw = img.copy()
    box = [int(x) for x in box]
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(
        img_to_draw,
        (box[0], box[1]),
        (box[2], box[3]),
        (255, 255, 0),
        2
    )

    ((text_width, text_height), _) = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img_to_draw, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), (255,255,255), -1)
    cv2.putText(
        img_to_draw,
        text=name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=(255, 255, 0),
        lineType=cv2.LINE_AA,
    )
    return img_to_draw

