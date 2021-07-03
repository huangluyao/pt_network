import os
import cv2
import json

import numpy as np
import xml.etree.ElementTree as ET
from .BaseDetectionDataset import BaseDataset
from test_utils.datasets import DATASET


@DATASET.registry()
class VOCDataset(BaseDataset):
    """
    Detects the reading of the dataset

    args:
        data_path:      The txt file path
        augmentation:   A dictionary that holds all the methods of data augmentations
        mode:           augmentation mode be used

    The dataset in voc format
    """

    def __init__(self, data_path, augmentations, mode='train', **kwargs):
        super(VOCDataset, self).__init__(data_path, augmentations, mode, **kwargs)

    def get_image_info(self, data_path, mode, **kwargs):

        if mode == "train":
            image_name_txt = os.path.join(data_path,"ImageSets","Main", "train.txt")
        else:
            image_name_txt = os.path.join(data_path,"ImageSets","Main", "val.txt")

        with open(image_name_txt, 'r') as f:
            image_names = f.readlines()

        image_infos = list()

        for image_name in image_names:
            image_name = image_name.strip()
            annotation_path = os.path.join(data_path, "Annotations", image_name+".xml")
            image_path = os.path.join(data_path,"JPEGImages", image_name+".jpg")

            with open(annotation_path, 'r', encoding='utf-8') as in_file:
                tree = ET.parse(in_file)
                root = tree.getroot()

            bboxes = []
            label_index = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in self.class_names:
                    self.class_names.append(cls)
                label_index.append(self.class_names.index(cls))



                xmlbox = obj.find('bndbox')
                b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                     int(xmlbox.find('ymax').text))
                bboxes.append(b)

            if self.max_number_object_per_img < len(label_index):
                self.max_number_object_per_img = len(label_index)
            image_info = dict(
                image_path=image_path,
                bboxes=bboxes,
                label_index=np.array(label_index)
                )
            image_infos.append(image_info)

        return image_infos

