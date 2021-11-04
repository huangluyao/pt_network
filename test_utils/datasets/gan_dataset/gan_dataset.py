import os
import cv2
import numpy as np
from test_utils.transforms import Compose
from torch.utils.data import Dataset
from test_utils.datasets import DATASET


@DATASET.registry()
class GANDataset(Dataset):

    def __init__(self, data_path, augmentations, input_shape=[224, 224, 3]):

        super(GANDataset, self).__init__()

        self.image_paths = self._get_info(data_path)
        self.input_shape = input_shape
        self.transfrom = Compose(augmentations)

    def _get_info(self, data_path):
        IMAGE_FORMER = ['JPEG', 'JPG', 'JPE', 'BMP', 'PNG', 'JP2', 'PBM', 'PGM', 'PPM']
        image_paths = list()
        categories_folder = os.listdir(data_path)
        for cls_indx, category_folder in enumerate(categories_folder):
            category_path = os.path.join(data_path, category_folder)
            suffix = category_path.split('.')[-1].upper()
            if suffix in IMAGE_FORMER:
                image_paths.append(category_path)
        return image_paths

    def __getitem__(self, item):
        image_path = self.image_paths[item]

        data = dict(image=cv2.imread(image_path))
        data = self.transfrom(**data)

        image = np.transpose(data['image'], [2, 0, 1])

        label_dict = dict(image_path=image_path)

        return image, label_dict

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def dataset_collate(batch):
        batch_images = np.array([b[0] for b in batch])
        image_dict = dict(real_img=batch_images)
        for key in batch[0][1].keys():
            image_dict[key] = np.array([b[1][key] for b in batch])

        return image_dict