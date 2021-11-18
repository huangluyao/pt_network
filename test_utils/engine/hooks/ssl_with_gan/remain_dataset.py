import os
import cv2
from torch.utils.data import Dataset, DataLoader
from test_utils.transforms import Compose

class RemainDataset(Dataset):

    def __init__(self, images_path, augmentations):
        self.image_paths = [os.path.join(images_path, image_name) for image_name in os.listdir(images_path)
                            if image_name.split('.')[-1].upper() in ['JPEG', 'JPG', 'JPE', 'BMP', 'PNG', 'JP2', 'PBM', 'PGM', 'PPM']
                            ]
        if augmentations is not None:
            self.transfrom = Compose(augmentations)
        else:
            self.transfrom = None
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        if self.transfrom:
            result = self.transfrom(image=image)["image"]
        else:
            result = image
        return result.transpose([2, 0, 1])

def get_remain_loader(image_path, augmentation, loader_cfg):
    dataset = RemainDataset(image_path, augmentation)
    return DataLoader(dataset,**loader_cfg)

