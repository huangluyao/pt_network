import os
import json
import cv2
import numpy as np
from test_utils.transforms import Compose
from test_utils.datasets import DATASET


@DATASET.registry()
class LabelMeSegDataset():
    class_names = ['background']

    def __init__(self,data_path, augmentations, mode="train", **kwargs):
        super(LabelMeSegDataset, self).__init__()

        self.transforms = Compose(augmentations[mode])
        self.images_info = self.get_image_info(data_path, mode, **kwargs)

    def __getitem__(self, item):

        image_info = self.images_info[item]
        image = cv2.imread(image_info["image_path"])
        if image is None:
            raise ValueError(f"Not such image file {image_info['image_path']}")

        mask = np.zeros(image.shape[:2])
        for label_point, index in zip(image_info["label_points"], image_info["label_index"]):
            mask = cv2.fillPoly(mask, [np.int0(label_point)], (int(index)))

        image_info.update(dict(image=image,
                               src_shape=image.shape,
                               mask=mask))

        image_info = self.transforms(**image_info)
        image = np.transpose(image_info.pop('image'), [2, 0, 1])
        image_info["gt_masks"] = image_info.pop("mask")
        return image, image_info

    def __len__(self):
        return len(self.images_info)

    def get_image_info(self, data_path, mode, **kwargs):
        annotations_path = os.path.join(data_path, 'annotations')
        annotation_files = os.listdir(annotations_path)

        image_infos = list()

        # parser annotations
        for annotation_file in annotation_files:
            annotation_file_path = os.path.join(annotations_path, annotation_file)

            if annotation_file_path.endswith('.json'):
                with open(annotation_file_path) as f:
                    info = json.load(f)
                labels_info = info.get('shapes', None)

                label_points = []
                label_names = []

                for label_info in labels_info:
                    label_name = label_info.get('label', None)
                    if not label_name:
                        continue
                    points = np.array(label_info['points'])
                    if label_name not in self.class_names:
                        self.class_names.append(label_name)

                    label_points.append(points)
                    label_names.append(self.class_names.index(label_name))

                image_info = dict(image_path=annotation_file_path.replace('annotations', 'images').replace('json', 'png'),
                                  label_points=label_points,
                                  label_index=np.array(label_names)
                                  )
                image_infos.append(image_info)

        return image_infos

if __name__=="__main__":
    data_path = "D:\\datasets\\huangluyao\\detection\\df_new\\val"
    json_path = "tools/config/augmentation/base_augmentation.json"
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    test = LabelMeSegDataset(data_path, cfg["dataset"]["augmentations"], "train")

    std = np.array([0.229,0.224,0.225])
    mean =np.array([0.485,0.456,0.406])

    for image, label in test:
        image = np.transpose(image, [1, 2, 0])
        image = image*std +mean
        label = label["gt_masks"].astype(np.uint8)
        label[label == 1] = 125
        cv2.imshow("result", image.astype(np.uint8))
        cv2.imshow("gt_masks", label)
        cv2.waitKey()



        pass