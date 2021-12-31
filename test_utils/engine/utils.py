import time
from getpass import getuser
from socket import gethostname
import os.path
import cv2
import torch
import numpy as np



def get_host_info():
    return f'{getuser()}@{gethostname()}'

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

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


def put_data(dataset, index_queue, image_queue, collate_fn):
    try:
        while True:
            batch_index = index_queue.get()
            data = [dataset[i] for i in batch_index]
            data = collate_fn(data)
            image_queue.put(data)
            del data, batch_index
    except KeyboardInterrupt:
        print("process exit")


def default_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy':
        return default_collate([torch.as_tensor(b) for b in batch])
    else:
        raise NotImplementedError

class BatchSampler:

    def __init__(self, sampler, batch_size, drop_last):
        super(BatchSampler, self).__init__()
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):

        batch = []
        for idx in self.sampler:

            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
