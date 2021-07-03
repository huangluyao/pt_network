# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/19 下午7:28
import os
import cv2
import json

import numpy as np
from .BaseDetectionDataset import BaseDataset
from test_utils.datasets import DATASET


@DATASET.registry()
class LabelmeTxtDataset(BaseDataset):
    """
    Detects the reading of the dataset

    args:
        data_path:      The txt file path
        augmentation:   A dictionary that holds all the methods of data augmentations
        mode:           augmentation mode be used

    read txt file like:
    images/yashang_img_00001_10.png	labelme/yashang_img_00001_10.json
    images/yashang_img_00001_13.png	labelme/yashang_img_00001_13.json
    images/yashang_img_00001_14.png	labelme/yashang_img_00001_14.json
    images/yashang_img_00007_10.png	labelme/yashang_img_00007_10.json
    images/yashang_img_00007_12.png	labelme/yashang_img_00007_12.json
    images/yashang_img_00007_14.png	labelme/yashang_img_00007_14.json
    """

    def __init__(self, data_path, augmentations, mode='train', **kwargs):
        super(LabelmeTxtDataset, self).__init__(data_path, augmentations, mode, **kwargs)

    def get_image_info(self, txt_path, mode, **kwargs):

        image_infos = list()
        with open(txt_path, 'r') as f:
            infos = f.readlines()

        for image_info in infos:
            image_path, annotation_path = image_info.split('\t')
            annotation_path = annotation_path.split('\n')[0]
            if annotation_path.endswith('.json'):
                with open(annotation_path) as f:
                    info = json.load(f)
                labels_info = info.get('shapes', None)
                bboxes = []
                label_names = []

                for label_info in labels_info:
                    label_name = label_info.get('label', None)
                    if not label_name:
                        continue
                    bbox = cv2.boundingRect(np.array(label_info['points']))
                    if label_name not in self.class_names:
                        self.class_names.append(label_name)

                    bboxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                    label_names.append(self.class_names.index(label_name))

                if self.max_number_object_per_img < len(label_names):
                    self.max_number_object_per_img = len(label_names)

                image_info = dict(image_path=image_path,
                                  bboxes=bboxes,
                                  label_index=np.array(label_names)
                                  )
                image_infos.append(image_info)
        return image_infos
