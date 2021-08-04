import os.path

import cv2
import torch
import numpy as np
from ..datasets import build_dataset
from ..datasets.detection_dataset import MosaicDetection

def calc_mean_std(images_path):
    # ---------------calc mean-------------------
    R_channel_sum = []
    G_channel_sum = []
    B_channel_sum = []
    num_pixels = 0
    for image_path in images_path:
        img = cv2.imread(image_path)
        if img is not None:
            R_channel_sum.append(np.sum(img[:, :, 2]))
            G_channel_sum.append(np.sum(img[:, :, 1]))
            B_channel_sum.append(np.sum(img[:, :, 0]))
            num_pixels += img.shape[0] * img.shape[1]

    R_mean = round(sum(R_channel_sum) / num_pixels, 3)
    G_mean = round(sum(G_channel_sum) / num_pixels, 3)
    B_mean = round(sum(B_channel_sum) / num_pixels, 3)
    means = [B_mean, G_mean, R_mean]

    # ---------------calc std-------------------
    R_channel_sum = []
    G_channel_sum = []
    B_channel_sum = []
    for image_path in images_path:
        img = cv2.imread(image_path)
        if img is not None:
            R_channel_sum.append(np.sum((img[:, :, 2] - R_mean) ** 2))
            G_channel_sum.append(np.sum((img[:, :, 1] - G_mean) ** 2))
            B_channel_sum.append(np.sum((img[:, :, 0] - B_mean) ** 2))

    R_var = sum(R_channel_sum) / num_pixels
    G_var = sum(G_channel_sum) / num_pixels
    B_var = sum(B_channel_sum) / num_pixels

    R_std = round(np.sqrt(R_var), 3)
    G_std = round(np.sqrt(G_var), 3)
    B_std = round(np.sqrt(B_var), 3)

    stds = [B_std, G_std, R_std]
    return means, stds


def statistics_data(image_paths):
    merge_paths = []
    if isinstance(image_paths, list):
        for image_path in image_paths:
            if image_path.endswith('.txt'):
                with open(image_path, 'r') as f:
                    image_infos = f.readlines()
                images_path = [image_info.split('\t')[0] for image_info in image_infos]
                pass
            else:
                images_path = os.path.join(image_path, 'images')
                images_path = [os.path.join(images_path, image_file) for image_file in os.listdir(images_path)]
            merge_paths += images_path
    elif isinstance(image_paths, str):
            image_paths = os.path.join(image_paths, 'images')
            images_path = [os.path.join(image_paths, image_file) for image_file in os.listdir(image_paths)]
            merge_paths = images_path

    return calc_mean_std(merge_paths)


def build_data_loader(task, type, input_size, train_data_path, val_data_path, loader_cfg, **cfg):

    if task == "detection":
        mosaic = cfg.pop("use_mosaic", None)
    train_cfg_dict = dict(type=type, data_path=train_data_path, **cfg)
    val_cfg_dict = dict(type=type, data_path=val_data_path, mode='val', **cfg)

    train_dataset = build_dataset(train_cfg_dict)
    val_dataset = build_dataset(val_cfg_dict)

    if task == "classification" or task=="segmentation":
        if 'FS' in type:
            dataset_collate = few_shot_dataset_collate
        else:
            dataset_collate = classification_dataset_collate
    elif task=="detection":
        dataset_collate = detection_dataset_collate
        if mosaic:
            train_dataset = MosaicDetection(train_dataset, input_size)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=dataset_collate, **loader_cfg)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, collate_fn=dataset_collate, **loader_cfg)
    return train_data_loader, val_data_loader


def classification_dataset_collate(batch):
    batch_images = np.array([b[0] for b in batch])
    dict_labels = dict()
    for key in batch[0][1].keys():
        dict_labels[key] = np.array([b[1][key] for b in batch])

    return batch_images, dict_labels


def few_shot_dataset_collate(batch):
    images = []
    labels = []

    # todo debug
    # for images in batch:
    #     for image in images[0]:
    #         img = np.transpose(image, [1, 2, 0])*255
    #         img = img.astype(np.int8)
    #         cv2.imshow('tse', img)
    #         cv2.waitKey()

    for img, label in batch:
        images.append(img)
        labels.append(label)

    images1 = np.array(images)[:, 0, :, :, :]
    images2 = np.array(images)[:, 1, :, :, :]
    images3 = np.array(images)[:, 2, :, :, :]
    images = np.concatenate([images1, images2, images3], 0)

    labels1 = np.array(labels)[:, 0]
    labels2 = np.array(labels)[:, 1]
    labels3 = np.array(labels)[:, 2]
    labels = np.concatenate([labels1, labels2, labels3], 0)
    gt = dict(gt_labels=labels)
    return images, gt


def detection_dataset_collate(batch):
    batch_img, batch_info = zip(*batch)

    batch_labels = dict()
    keys = batch_info[0].keys()
    for info in batch_info:
        for key in keys:
            if key in batch_labels:
                batch_labels[key].append(info[key])
            else:
                batch_labels[key] = [info[key]]
    batch_img = np.transpose(np.array(batch_img),[0, 3, 1, 2])
    return batch_img, batch_labels
