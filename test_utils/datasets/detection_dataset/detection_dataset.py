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
class DetectionDataset(BaseDataset):
    """
    Detects the reading of the dataset

    args:
        data_path:      The dataset path
        augmentation:   A dictionary that holds all the methods of data augmentations
        mode:           augmentation mode be used

    The folder format is as follows:
    ├─images
    └─annotations
    """

    def __init__(self, data_path, augmentations, mode='train', **kwargs):
        super(DetectionDataset, self).__init__(data_path, augmentations, mode, **kwargs)

    def get_image_info(self, data_path, mode, **kwargs):

        annotations_path = os.path.join(data_path, 'annotations')
        annotation_files = os.listdir(annotations_path)

        image_infos = list()

        # parser annotations
        for annotation_file in annotation_files:
            annotation_file_path = os.path.join(annotations_path, annotation_file)

            if annotation_file_path.endswith('.json'):
                with open(annotation_file_path) as f:
                    info = json.load(f)
                labels_info = info.get('shapes', None)

                bboxes = []
                label_names = []

                for label_info in labels_info:
                    label_name = label_info.get('label', None)
                    if not label_name:
                        continue
                    bbox = cv2.boundingRect(np.array(label_info['points']).astype(np.float32))
                    if label_name not in self.class_names:
                        self.class_names.append(label_name)

                    bboxes.append([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                    label_names.append(self.class_names.index(label_name))

                if self.max_number_object_per_img < len(label_names):
                    self.max_number_object_per_img = len(label_names)

                image_info = dict(image_path=annotation_file_path.replace('annotations', 'images').replace('json', 'png'),
                                  bboxes=bboxes,
                                  label_index=np.array(label_names)
                                  )
                image_infos.append(image_info)

        return image_infos

