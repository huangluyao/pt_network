import cv2
import numpy as np

from test_utils.transforms import Compose
from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset


class BaseDataset(Dataset, metaclass=ABCMeta):

    class_names = []

    def __init__(self,data_path, augmentations, mode, **kwargs):

        self.transforms = Compose(augmentations[mode])
        self.images_info = self.get_image_info(data_path, mode, **kwargs)


    @abstractmethod
    def get_image_info(self, data_path, mode, **kwargs):

        pass