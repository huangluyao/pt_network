import copy
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
from test_utils.transforms import Compose
from torch.utils.data import Dataset

class BaseDataset(Dataset, metaclass=ABCMeta):

    IMAGE_FORMER = ['JPEG', 'JPG', 'JPE', 'BMP', 'PNG', 'JP2', 'PBM', 'PGM', 'PPM']

    def __init__(self, augmentations):
        self.pipeline = Compose(augmentations)

    @abstractmethod
    def load_annotations(self):
        """Abstract function for loading annotation.

        All subclasses should overwrite this function
        """

    def __getitem__(self, idx):

        image_info = copy.deepcopy(self.data_infos[idx])

        image = cv2.imread(image_info["image_path"])
        if image is None:
            raise ValueError(f"Not such image file {image_info['image_path']}")

        mask = np.zeros(image.shape[:2])
        for label_point, index in zip(image_info["label_points"], image_info["label_index"]):
            mask = cv2.fillPoly(mask, [np.int0(label_point)], (1))

        image_info.update(dict(image=image,
                               src_shape=image.shape,
                               mask=mask))

        image_info = self.pipeline(**image_info)


        masked_img = image_info["image"] * (1. - image_info["mask"])[...,None]
        image_info['masked_img'] = masked_img

        return image_info

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_infos)