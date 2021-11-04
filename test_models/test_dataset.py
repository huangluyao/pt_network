import json
import cv2
import numpy as np
from test_utils.datasets import ImageInpaintingDataset

if __name__=="__main__":

    data_path = "E:\\datasets\\huangluyao\\segmentation\\df_new\\val"
    json_path = "tools/config/__base__/augmentation/gan_augmentation.json"
    with open(json_path, 'r') as f:
        cfg = json.load(f)
    test = ImageInpaintingDataset(data_path, cfg["dataset"]["augmentations"]["train"])

    std = np.array([127.5, 127.5, 127.5])
    mean = np.array([127.5, 127.5, 127.5])

    from imgviz import color as color_module


    def label_colormap(n_label=256, value=None):
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        cmap = np.zeros((n_label, 3), dtype=np.uint8)
        for i in range(0, n_label):
            id = i
            r, g, b = 0, 0, 0
            for j in range(0, 8):
                r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
                g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
                b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

        if value is not None:
            hsv = color_module.rgb2hsv(cmap.reshape(1, -1, 3))
            if isinstance(value, float):
                hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
            else:
                assert isinstance(value, int)
                hsv[:, 1:, 2] = value
            cmap = color_module.hsv2rgb(hsv).reshape(-1, 3)
        return cmap


    cm = label_colormap()

    for image_info in test:
        image = (image_info["image"] * std + mean).astype(np.uint8)
        mask_image = (image_info["masked_img"]* std + mean).astype(np.uint8)
        mask = (image_info["mask"] * 255).astype(np.uint8)
        cv2.imshow("result", image)
        cv2.imshow("masks", mask)
        cv2.imshow("masked_img", mask_image)
        cv2.waitKey()
