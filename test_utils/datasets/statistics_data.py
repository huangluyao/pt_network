import os.path
import cv2
import numpy as np

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


def statistics_data(image_paths, dataset_type):

    if dataset_type=="DetectionDataset" or dataset_type=="LabelMeSegDataset":
        return statistics_detection_data(image_paths)
    elif dataset_type == "LabelmeTxtDataset":
        return statistics_labelme_txt_data(image_paths)
    elif dataset_type =="VOCDataset":
        return statistics_voc_data(image_paths)
    elif dataset_type == "ClsDataSet":
        return statistics_cls_data(image_paths)


def statistics_labelme_txt_data(image_paths):
    merge_paths = []
    if isinstance(image_paths, list):
        for image_path in image_paths:
            if image_path.endswith('.txt'):
                with open(image_path, 'r') as f:
                    image_infos = f.readlines()
                images_path = [image_info.split('\t')[0] for image_info in image_infos]
                merge_paths += images_path
    elif isinstance(image_paths, str):
            image_paths = os.path.join(image_paths, 'images')
            images_path = [os.path.join(image_paths, image_file) for image_file in os.listdir(image_paths)]
            merge_paths = images_path
    return calc_mean_std(merge_paths)


def statistics_detection_data(image_paths):
    merge_paths = []
    if isinstance(image_paths, list):
        for image_path in image_paths:
            images_path = os.path.join(image_path, 'images')
            images_path = [os.path.join(images_path, image_file) for image_file in os.listdir(images_path)]
            merge_paths += images_path
    elif isinstance(image_paths, str):
            image_paths = os.path.join(image_paths, 'images')
            images_path = [os.path.join(image_paths, image_file) for image_file in os.listdir(image_paths)]
            merge_paths = images_path
    return calc_mean_std(merge_paths)


def statistics_voc_data(image_paths):
    merge_paths = []
    if isinstance(image_paths, list):
            images_path = os.path.join(image_paths[0], 'JPEGImages')
            images_path = [os.path.join(images_path, image_file) for image_file in os.listdir(images_path)]
            merge_paths += images_path
    elif isinstance(image_paths, str):
            image_paths = os.path.join(image_paths[0], 'JPEGImages')
            images_path = [os.path.join(image_paths, image_file) for image_file in os.listdir(image_paths)]
            merge_paths = images_path
    return calc_mean_std(merge_paths)

def statistics_cls_data(image_paths):
    merge_paths = []
    if isinstance(image_paths, list):
        for image_path in image_paths:
            category_folders = [os.path.join(image_path, image_file) for image_file in os.listdir(image_path) if not os.path.isfile(os.path.join(image_path, image_file))]
            for category_folder in category_folders:
                merge_paths += [os.path.join(category_folder, file_name) for file_name in os.listdir(category_folder)]
    elif isinstance(image_paths, str):
            category_folders = [os.path.join(image_paths, image_file) for image_file in os.listdir(image_paths) if not os.path.isfile(os.path.join(image_paths, image_file))]
            for category_folder in category_folders:
                merge_paths += [os.path.join(category_folder, file_name) for file_name in os.listdir(category_folder)]
    return calc_mean_std(merge_paths)