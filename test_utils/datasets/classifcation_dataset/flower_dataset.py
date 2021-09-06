import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from test_utils.datasets import DATASET

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def letterbox_image(image, size):
    image = image.convert("RGB")
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


@DATASET.registry(name='FS_flower17')
class FlowerDataset_FS(Dataset):

    class_names = []

    def __init__(self, data_path, input_shape=[224, 224, 3], mode='train', **kwargs):
        self.mode = mode
        self.image_paths, self.image_labels, self.num_classes = self._get_info(data_path)
        self.input_shape = input_shape

        pass

    def _get_info(self, data_path):

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
                    image_path = os.path.join(category_path, image_name)
                    image_paths.append(image_path)
                    image_labels.append(cls_indx)
        return image_paths, image_labels, num_classes

    def __getitem__(self, item):
        if self.mode == 'test':
            return self._get_test_data()
        else:
            return self._get_train_val_data()

    def _get_train_val_data(self):
        images = np.zeros((3, self.input_shape[2], self.input_shape[0], self.input_shape[1]))
        labels = np.zeros((3))

        # select random category as anchor and positive image
        c = random.randint(0, self.num_classes-1)
        select_category = [self.image_paths[i] for i, image_label in enumerate(self.image_labels) if image_label == c]
        while len(select_category) < 2:
            c = random.randint(0, self.num_classes - 1)
            select_category = [self.image_paths[i] for i, image_label in enumerate(self.image_labels) if image_label == c]

        images_indexs = np.random.choice(range(0, len(select_category)), 2)

        # get anchor image and label
        image = Image.open(select_category[images_indexs[0]])
        image = self.get_random_data(image, self.input_shape[:2])
        image = np.transpose(np.asarray(image).astype(np.float64),[2,0,1]) / 255
        images[0, :, :, :] = image
        labels[0] = c
        # get positive image and label
        image = Image.open(select_category[images_indexs[1]])
        image = self.get_random_data(image, self.input_shape[:2])
        image = np.transpose(np.asarray(image).astype(np.float64),[2,0,1]) / 255
        images[1, :, :, :] = image
        labels[1] = c

        # get negative image and label
        different_c= list(range(0,self.num_classes))
        different_c.pop(c)
        different_c_index = random.choice(different_c)
        select_different_category = [self.image_paths[i] for i, image_label in enumerate(self.image_labels) if image_label == different_c_index]
        while len(select_different_category)<1:
            different_c_index = random.choice(different_c)
            select_different_category = [self.image_paths[i] for i, image_label in enumerate(self.image_labels) if image_label == different_c_index]

        neg_image_index = random.randint(0, len(select_different_category)-1)
        neg_image = Image.open(select_different_category[neg_image_index])
        neg_image = self.get_random_data(neg_image, self.input_shape[:2])
        neg_image = np.transpose(np.asarray(neg_image).astype(np.float64), [2, 0, 1]) / 255.
        images[2, ...] = neg_image
        labels[2] = different_c_index

        return images, labels

    def _get_test_data(self):

        # select one category
        list_labels = list(range(0, self.num_classes))
        c = random.randint(0, self.num_classes-1)
        select_category = [self.image_paths[i] for i, image_label in enumerate(self.image_labels) if image_label == c]
        while len(select_category)<1:
            c = random.randint(0, self.num_classes - 1)
            select_category = [self.image_paths[i] for i, image_label in enumerate(self.image_labels) if image_label == c]

        if random.random() > 0.5:
            image_paths = random.choices(select_category, k=2)
            image1, image2 = Image.open(image_paths[0]), Image.open(image_paths[1])
            image1 = letterbox_image(image1, self.input_shape[:2])
            image2 = letterbox_image(image2, self.input_shape[:2])
            issame = True
        else:
            image1_path = random.choice(select_category)
            list_labels.pop(c)
            different_c = random.choice(list_labels)
            select_different_category = [self.image_paths[i] for i, image_label in enumerate(self.image_labels) if image_label == different_c]
            while len(select_different_category)<1:
                different_c = random.choice(list_labels)
                select_different_category = [self.image_paths[i] for i, image_label in enumerate(self.image_labels) if
                                             image_label == different_c]
            image2_path = random.choice(select_different_category)
            image1, image2 = Image.open(image1_path), Image.open(image2_path)
            image1 = letterbox_image(image1, self.input_shape[:2])
            image2 = letterbox_image(image2, self.input_shape[:2])
            issame = False

        img1, img2 = np.array(image1)/255, np.array(image2)/255
        img1 = np.transpose(img1,[2,0,1])
        img2 = np.transpose(img2,[2,0,1])

        return img1, img2, issame

    def get_random_data(self, image, input_shape, jitter=.1, hue=.05, sat=1.3, val=1.3, flip_signal=True):
        image = image.convert("RGB")

        h, w = input_shape
        rand_jit1 = rand(1 - jitter, 1 + jitter)
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = rand(0.9, 1.1)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        flip = rand() < .5
        if flip and flip_signal:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        rotate = rand() < .5
        if rotate:
            angle = np.random.randint(-5, 5)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])

        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        # if self.channel == 1:
        #     image_data = Image.fromarray(np.uint8(image_data)).convert("L")
        # cv2.imshow("TEST",np.uint8(cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)))
        # cv2.waitKey(0)
        return image_data

    def __len__(self):
        return len(self.image_paths)