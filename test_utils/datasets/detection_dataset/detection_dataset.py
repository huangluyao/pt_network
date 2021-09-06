# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/19 下午7:28
import os
import cv2
import json

import numpy as np
from test_utils.datasets.detection_dataset.BaseDetectionDataset import BaseDataset
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

        images_path = os.path.join(data_path, 'images')
        image_files = os.listdir(images_path)
        IMAGE_FORMER = ['JPEG', 'JPG', 'JPE', 'BMP', 'PNG', 'JP2', 'PBM', 'PGM', 'PPM']
        image_infos = list()

        # parser annotations
        for image_file in image_files:

            image_name, suffix = image_file.split('.')
            if suffix.upper() in IMAGE_FORMER:
                image_file_path = os.path.join(images_path, image_file)
                annotation_file_path = os.path.join(images_path.replace('images', 'annotations'), image_name+'.json')

                if os.path.exists(annotation_file_path) and os.path.exists(image_file_path):
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

                    image_info = dict(image_path=image_file_path,
                                      bboxes=bboxes,
                                      label_index=np.array(label_names)
                                      )
                    image_infos.append(image_info)

        return image_infos


    @staticmethod
    def dataset_collate(batch):
        batch_img, batch_info = zip(*batch)

        batch_labels = dict()
        keys = batch_info[0].keys()
        for info in batch_info:
            for key in keys:
                if key in batch_labels:
                    batch_labels[key].append(info[key])
                else:
                    batch_labels[key] = [info[key]]
        batch_img = np.transpose(np.array(batch_img), [0, 3, 1, 2])
        return batch_img, batch_labels


if __name__=="__main__":
    data_path = "D:\\datasets\\huangluyao\\detection\\plane_car\\train"
    json_path = "tools/config/augmentation/base_augmentation.json"
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    test = DetectionDataset(data_path, cfg["dataset"]["augmentations"], "train")

    std = np.array([0.229,0.224,0.225])
    mean =np.array([0.485,0.456,0.406])

    for image, label in test:
        image = np.transpose(image, [1, 2, 0])
        image = image*std +mean

        for boxes, index in zip(label["bboxes"], label["label_index"]):
            x1, y1, x2, y2 = [int(x) for x in boxes]
            cv2.rectangle(image,(x1, y1), (x2,y2),(0,255,0))

        cv2.imshow("result", image.astype(np.uint8))
        cv2.waitKey()
