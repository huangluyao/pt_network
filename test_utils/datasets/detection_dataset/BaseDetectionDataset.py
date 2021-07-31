import cv2
import numpy as np

from test_utils.transforms import Compose
from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset


class BaseDataset(Dataset, metaclass=ABCMeta):

    class_names = []
    max_number_object_per_img = 0

    def __init__(self,data_path, augmentations, mode, **kwargs):

        self.transforms = Compose(augmentations[mode])
        self.images_info = self.get_image_info(data_path, mode, **kwargs)

    def __getitem__(self, item):
        image_info = self.images_info[item]
        img = cv2.imread(image_info["image_path"])
        image_info.update(dict(image=img,
                               src_shape=img.shape))

        data = self.transforms(**image_info)
        image = np.transpose(data.pop('image'), [2, 0, 1])
        data['bboxes'] = np.array(data['bboxes']).astype(np.float32)
        return image, data

    def __len__(self):
        return len(self.images_info)

    @abstractmethod
    def get_image_info(self, data_path, mode, **kwargs):

        pass