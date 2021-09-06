# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/17 上午10:01

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from test_utils.transforms import Compose
from test_utils.datasets import DATASET


@DATASET.registry()
class ClsDataSet(Dataset):

    class_names = []

    def __init__(self, data_path, augmentations, input_shape=[224, 224, 3], mode='train', **kwargs):
        self.mode = mode
        self.image_paths, self.image_labels, self.num_classes = self._get_info(data_path)
        self.input_shape = input_shape
        self.transfrom = Compose(augmentations[mode])

    def _get_info(self, data_path):
        IMAGE_FORMER = ['JPEG', 'JPG', 'JPE', 'BMP', 'PNG', 'JP2', 'PBM', 'PGM', 'PPM']
        image_paths = list()
        image_labels = list()
        num_classes = 0
        categories_folder = os.listdir(data_path)
        self.class_names = categories_folder
        for cls_indx, category_folder in enumerate(categories_folder):
            category_path = os.path.join(data_path, category_folder)
            if os.path.isdir(category_path):
                num_classes += 1
                images_name = os.listdir(category_path)
                for image_name in images_name:
                    suffix = image_name.split('.')[-1].upper()
                    if suffix in IMAGE_FORMER:
                        image_path = os.path.join(category_path, image_name)
                        image_paths.append(image_path)
                        image_labels.append(cls_indx)
        return image_paths, image_labels, num_classes

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        label = self.image_labels[item]

        data = dict(image=cv2.imread(image_path),
                    label=label
                    )
        data = self.transfrom(**data)

        image = np.transpose(data['image'], [2, 0, 1])
        label = data['label']

        label_dict = dict(gt_labels=label,
                          image_path=image_path
                          )

        return image, label_dict

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def dataset_collate(batch):
        batch_images = np.array([b[0] for b in batch])
        dict_labels = dict()
        for key in batch[0][1].keys():
            dict_labels[key] = np.array([b[1][key] for b in batch])

        return batch_images, dict_labels