import os
import json
import numpy as np
from test_utils.datasets import DATASET
from .base_dataset import BaseDataset
import torch

@DATASET.registry()
class ImageInpaintingDataset(BaseDataset):

    class_names = []

    def __init__(self, data_path, augmentations, **kwargs):

        super(ImageInpaintingDataset, self).__init__(augmentations=augmentations)
        self.data_path = data_path
        self.data_infos = self.load_annotations()

    def load_annotations(self):

        images_path = os.path.join(self.data_path, 'images')
        image_files = os.listdir(images_path)

        image_infos = list()

        # parser annotations
        for image_file in image_files:
            image_name, suffix = image_file.split('.')
            if suffix.upper() in self.IMAGE_FORMER:

                image_file_path = os.path.join(images_path, image_file)
                annotation_file_path = os.path.join(images_path.replace('images', 'annotations'), image_name+'.json')

                if os.path.exists(annotation_file_path) and os.path.exists(image_file_path):
                    with open(annotation_file_path) as f:
                        info = json.load(f)
                        labels_info = info.get('shapes', None)

                    label_points = []
                    label_names = []

                    for label_info in labels_info:
                        label_name = label_info.get('label', None)
                        if not label_name:
                            continue
                        points = np.array(label_info['points'])
                        if label_name not in self.class_names:
                            self.class_names.append(label_name)

                        label_points.append(points)
                        label_names.append(self.class_names.index(label_name)+1)

                    image_info = dict(
                        image_path=annotation_file_path.replace('annotations', 'images').replace('json', 'png'),
                        label_points=label_points,
                        label_index=np.array(label_names)
                        )
                    image_infos.append(image_info)

        return image_infos


    @staticmethod
    def dataset_collate(batch):
        dict_labels = dict()
        for key in batch[0].keys():
            if key in ["image", "mask", "masked_img"]:
                if key == "mask":
                    dict_labels[key] = torch.tensor(np.array([b[key] for b in batch])).unsqueeze(dim=1).float()
                else:
                    dict_labels[key] = torch.tensor(np.array([b[key] for b in batch]).transpose(0, 3, 1, 2)).float()
            else:
                dict_labels[key] = [b[key] for b in batch]

        return dict_labels