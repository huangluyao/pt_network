from .base_seg_dataset import BaseDataset
import os
import glob
import cv2
import numpy as np
from test_utils.datasets import DATASET


@DATASET.registry()
class ChangeSegDataset(BaseDataset):
    class_names = ['changed']

    def __init__(self, data_root=None, **kwargs):
        self.data_root = data_root
        super(ChangeSegDataset, self).__init__(**kwargs)

    def get_image_info(self, data_path, mode, **kwargs):

        with open(data_path, 'r') as f:
            model_list = f.readlines()
        model_list = [name.strip() for name in model_list]

        ng_files = []
        for d in glob.glob(self.data_root + '/NG/*'):
            if not os.path.isdir(d):
                continue
            d = d.replace('\\', '/')
            model_name = d.split('/')[-1]

            if model_list is not None and model_name not in model_list:
                continue

            for f in glob.glob(d + '/*'):
                if not os.path.isdir(f):
                    continue
                for h in glob.glob(f + '/*.jpg'):
                    h2 = h[:-4] + '_mask.bmp'
                    h2, h = h2.replace('\\', '/'), h.replace('\\', '/')
                    ng_files.append((h, h2))

        pairs = []
        for d in ng_files:
            f = d[0].split('/')
            ok_dir = os.path.join(self.data_root, 'OK', f[-3], f[-2])
            for f in glob.glob(ok_dir + '/*.jpg'):
                pairs.append((f, d[0], d[1]))

        return pairs

    def __getitem__(self, item):

        ok, ng, mask = self.images_info[item]

        im1 = cv2.imread(ok, cv2.IMREAD_GRAYSCALE) #(512, 512)
        im2 = cv2.imread(ng, cv2.IMREAD_GRAYSCALE) #(512, 512)
        msk = cv2.imread(mask, cv2.IMREAD_GRAYSCALE) #msk
        msk = msk.astype(np.float32) / 255

        image_info = self.transforms(images=[im1, im2], mask=msk)
        image_info["label_index"] = 1
        image_info["image_path"] = ng

        image = np.stack(image_info["images"], axis=-1).transpose([2, 0, 1])
        image_info["gt_masks"] = image_info.pop("mask")

        return image, image_info