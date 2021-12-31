import cv2
import numpy as np

from test_utils.transforms import Compose
from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset


class BaseDataset(Dataset, metaclass=ABCMeta):

    class_names = []

    def __init__(self,data_path, augmentations, mode='train', **kwargs):

        self.transforms = Compose(augmentations[mode])
        self.images_info = self.get_image_info(data_path, mode, **kwargs)

    def __len__(self):
        return len(self.images_info)

    @abstractmethod
    def get_image_info(self, data_path, mode, **kwargs):

        pass

    @staticmethod
    def dataset_collate(batch):
        batch_images = np.array([b[0] for b in batch])
        dict_labels = dict()
        for key in batch[0][1].keys():
            dict_labels[key] = np.array([b[1][key] for b in batch])

        return batch_images, dict_labels
