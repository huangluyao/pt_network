import os
import torch, cv2
import numpy as np
from networks.cls import build_classifier
from networks.det import build_detector
from networks.seg import build_segmentor
from test_utils.transforms import Compose
from test_utils.utils.checkpoint import load_checkpoint
from test_utils.evaluator.visualizer import resize_box, resize_mask, show_detections, get_pred, get_BGR_values
from tools.train import parse_config_file, fromfile
from test_utils.utils.file_io import mkdir
from other_code.dark import deHaze
temp = True
if temp:
    import csv

    f = open('test.csv', 'w', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['name', 'image_id', 'confidence', 'xmin',
                         'ymin', 'xmax', 'ymax'
                         ])


    def save_csv(image_path, boxes, ori_image, input_size, classes_name):

        image_id = os.path.basename(image_path).split('.')[0]
        # image_id = int(image_id)
        ori_size = ori_image.shape[:2]
        boxes = boxes[0]
        boxes = resize_box(ori_size, (input_size[1], input_size[0]), boxes)

        detections = boxes[~np.all(boxes == 0, axis=1)]
        indices = np.where(detections[:, -1] >= 0.3)[0]
        detections = detections[indices, :]
        class_ids = np.unique(detections[:, -2])
        class_ids = class_ids[class_ids > 0].astype(np.int32)
        if len(class_ids) == 0:
            return
        for detection in detections:
            confidence = detection[-1]
            name = classes_name[int(detection[4])-1]
            info = [name, image_id, confidence, int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])]
            csv_writer.writerow(info)


class Predict:
    def __init__(self, cfg):
        self._task = cfg['task']
        # init process image
        self.transformer = Compose(cfg["dataset"]["augmentations"]["val"])
        # init model
        self.model = self.build_model(cfg)
        self._device, gpu_ids = self.get_device(gpu_id=cfg.get('gpu_id', 0))
        self.model = self.model.to(self._device)
        self.classes_name = cfg["dataset"]["classes_name"]
        self.input_size = cfg["dataset"]["input_size"]
        self.model.eval()

    def __call__(self, src, image_path=None):

        # process image
        image = src.copy()
        image = self.transformer(image=image)["image"]

        input = np.transpose(image, [2, 0, 1])[None, :]
        input = torch.from_numpy(input).to(device=self._device)
        # infer
        predict = self.model(input)

        predict = predict.detach().cpu().numpy()
        result = self.get_result_img(src, predict)
        if temp:
            save_csv(image_path, predict, src, self.input_size, self.classes_name)
        # get results visualization
        return result

    def get_result_img(self, image, predict):
        ori_image = image.copy()
        if self._task == "classification":

            index = np.argmax(predict)
            result_txt = "%s: %.4f" % (self.classes_name[index], predict[index])
            ((text_width, text_height), _) = cv2.getTextSize(result_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            cv2.putText(ori_image, result_txt, (10, text_height+10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 0))
            return ori_image

        if self._task == "detection":
            ori_size = ori_image.shape[:2]
            boxes = predict[0]
            boxes = resize_box(ori_size, (self.input_size[1], self.input_size[0]), boxes)
            vis_result = show_detections(ori_image, boxes, self.classes_name, 0.3)
            return vis_result

        if self._task == "segmentation":
            mask = get_pred(predict)
            bgr_values = get_BGR_values(len(self.classes_name))
            ori_size = ori_image.shape[:2]
            image, mask = resize_mask(image, mask, ori_size)
            vis_preds = np.zeros_like(ori_image)
            for class_id in range(1, len(self.classes_name)):
                vis_preds[:,:,0][mask==class_id] = bgr_values[class_id-1][0]
                vis_preds[:,:,1][mask==class_id] = bgr_values[class_id-1][1]
                vis_preds[:,:,2][mask==class_id] = bgr_values[class_id-1][2]

            mask = (mask==0)[:,:,None]
            vis_preds = (ori_image*mask+(1-mask)*(vis_preds*0.5+ori_image*0.5)).astype(np.uint8)
            return vis_preds


    def build_model(self, cfg):
        model_cfg = cfg.get('model')
        if cfg.get('task') == "classification":
            model = build_classifier(model_cfg)
        elif cfg.get('task') == "detection":
            model = build_detector(model_cfg)
        elif cfg.get('task') == "segmentation":
            model = build_segmentor(model_cfg)
        else:
            raise TypeError(f"task must be classification, detection or segmentation, but go {cfg['task']}")
        checkpoint_path = os.path.join(cfg["output_dir"], "checkpoints", "val_best.pth")
        if os.path.exists(checkpoint_path):
            checkpoint_path = os.path.expanduser(checkpoint_path)
            load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True)
            print(f"load checkpoint from {checkpoint_path}")
        return model

    def get_device(self, gpu_id):
        gpu_count = torch.cuda.device_count()
        if gpu_id > 0 and gpu_count == 0:
            print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            num_gpus = 0
        if gpu_id > gpu_count:
            print("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                  "on this machine.".format(gpu_id, gpu_count))
            gpu_id = gpu_count
        device = torch.device('cuda:%d'%(gpu_id) if torch.cuda.is_available() else 'cpu')
        return device, gpu_id


if __name__=="__main__":
    import argparse

    def setup():
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config',
                            default="export/detection/FCOS_DSNet_CSP_PAN_DetectionDataset/2021-07-27-11-54-28/config.json",
                            type=str)
        parser.add_argument('-f', '--file', type=str, default='', help='initial weights path')
        args = parser.parse_args()
        cfg = parse_config_file(args.config)
        cfg = fromfile(cfg)
        return cfg, args.file

    cfg, file = setup()

    if file == '':
        file = "/media/hly/Samsung_T51/datasets/huangluyao/standard/under_water/test-A-image/test-A-image"

    pred = Predict(cfg)
    IMAGE_FORMER = ['JPEG', 'JPG', 'JPE', 'BMP', 'PNG', 'JP2', 'PBM', 'PGM', 'PPM']
    result_dir = os.path.join(cfg["output_dir"], "predict")
    mkdir(result_dir)

    if os.path.isdir(file):
        image_names = [name for name in sorted(os.listdir(file)) if name.split('.')[-1].upper() in IMAGE_FORMER]

        for image_name in image_names:
            image_path = os.path.join(file, image_name)
            img = cv2.imread(image_path)
            img = deHaze(img /255.0)*255
            result = pred(img, image_path)
            print(f"predict image from {image_path}")
            cv2.imwrite(os.path.join(result_dir, image_name), result)
