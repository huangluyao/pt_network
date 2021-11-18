import json
import numpy as np
import cv2
import torch
import os
from copy import deepcopy
from collections import OrderedDict
from test_utils.utils.utils import build_gan_model
from test_utils.transforms import Compose, Resize, Normalize, ToTensor
from test_utils.utils.misc import tensor2img
from test_utils.utils.checkpoint import load_checkpoint

class ImageInpaintingInfer:

    def __init__(self, cfg):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 读取配置文件
        with open(cfg, 'r', encoding='utf-8') as f:
            config_dict = json.load(f, object_pairs_hook=OrderedDict)

        # 构建网络模型
        model_cfg = config_dict.get('model')
        self.network = build_gan_model(model_cfg).to(self.device)
        # 寻找网络权重并加载
        weight_dir = os.path.join(config_dict.get("output_dir"), "weights")
        iter_weight = {float(iter.split('_')[1].split('.')[0]):iter for iter in os.listdir(weight_dir)}
        weights = sorted(iter_weight.keys(), reverse=True)
        load_checkpoint(self.network, os.path.join(weight_dir, iter_weight[weights[0]]))


        resize_cfg = None
        norm_cfg = None
        for aug_cfg in config_dict["dataset"]["augmentations"]:
            type = aug_cfg.pop("type", None)
            if type == "Resize":
                resize_cfg = aug_cfg
            elif type == "Normalize":
                norm_cfg = aug_cfg
            else:
                continue
        self.pipeline = Compose([Resize(**resize_cfg),
                                 Normalize(**norm_cfg),
                                 ToTensor(transpose_mask=True, always_apply=True)
                                 ])
        pass


    def __call__(self, image, mask):

        result = self.pipeline(image=image, mask=mask)
        result["masked_img"] = (result["image"]*(1-result["mask"][None,...])).unsqueeze(0).to(self.device).float()
        result["mask"] = result["mask"][None, None, ...].to(self.device).float()
        result = self.network(result, return_loss=False)
        return result


class ImageInpatint:

    def __init__(self, cfg_file):

        self.inpaintor = ImageInpaintingInfer(cfg_file)

        cv2.namedWindow("test")
        cv2.setMouseCallback("test", self.onMouse)
        self.src_image = None
        self.mask = None
        self.mask_img = None

        self.drag_start = None

    def onMouse(self, event, x, y, flags, param):

        if event == cv2.EVENT_MOUSEMOVE:
            if self.drag_start is not None:
                self.drag_start.append([x, y])
                if self.show_image is not None:
                    cv2.fillPoly(self.show_image, [np.int0(self.drag_start)], color=(255,255,0))

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = [[x, y]]
        elif event == cv2.EVENT_LBUTTONUP:
            if self.src_image is not None:
                # 制作mask
                self.mask = np.zeros(self.src_image.shape[:2])
                cv2.fillPoly(self.mask, [np.int0(self.drag_start)], 1)
                # 膨胀
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mask = cv2.dilate(self.mask, kernel=kernel)

                result = self.inpaintor(self.src_image, mask)

                result = tensor2img(result["fake_img"], min_max=(-1, 1))[..., ::-1]
                height, width = self.mask.shape
                show_image = cv2.resize(result, (width, height))
                self.show_image = show_image

            self.drag_start = None


    def onMouse_draw_rectangle(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)

        if event == cv2.EVENT_LBUTTONUP:

            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)

            if self.src_image is not None:
                self.mask = np.zeros(self.src_image.shape[:2])
                cv2.rectangle(self.mask,(xmin, ymin), (xmax, ymax),
                              color=1, thickness=cv2.FILLED)

                result = self.inpaintor(self.src_image, self.mask)

                result = tensor2img(result["fake_img"], min_max=(-1, 1))[..., ::-1]
                height, width = self.mask.shape
                show_image = cv2.resize(result, (width, height))
                self.show_image = show_image

    def __call__(self, image):
        self.src_image = image
        self.show_image = deepcopy(self.src_image)

        while True:
            cv2.imshow("test", self.show_image)
            if cv2.waitKey(1)==27:
                break


if __name__=="__main__":
    cfg_file = "export/gan/TwoStageInpaintor_None_ImageInpaintingDataset/2021-11-04-07-17-30/config.json"
    app = ImageInpatint(cfg_file)
    image = cv2.imread("C:/workspace/data/classification/df/train/ok/20210426201651795490_deliver_flex_roi0_c4_8.png")
    app(image)