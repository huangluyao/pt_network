import os
from pprint import pprint
from pyexpat import model
from re import S
import torch, cv2
import numpy as np
from networks.cls import build_classifier
from networks.det import build_detector
from networks.seg import build_segmentor
from test_utils.transforms import Compose
from test_utils.utils.checkpoint import load_checkpoint
from test_utils.evaluator.eval_hooks.det_eval_hooks import resize_box, show_detections, get_BGR_values
from test_utils.evaluator.eval_hooks.seg_eval_hooks import get_pred
from tools.train import parse_config_file, fromfile
from test_utils.utils.file_io import mkdir
import onnx, onnxruntime

def resize_mask(mask, size):
    resize_h, resize_w = size
    im_h, im_w  = mask.shape
    resize_ratio = min(resize_w / im_w, resize_h / im_h)
    new_w = round(im_w * resize_ratio)
    new_h = round(im_h * resize_ratio)
    mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    mask_padding = np.full([resize_h, resize_w], 0)
    mask_padding[(resize_h-new_h)//2:(resize_h-new_h)//2 + new_h, (resize_w-new_w)//2:(resize_w-new_w)//2 + new_w] = mask_resized
    return  mask_padding


class Exporter:
    def __init__(self, cfg):
        self._task = cfg['task']
        # init process image
        self.transformer = Compose(cfg["dataset"]["augmentations"]["val"])
        # init model
        self.model = self.build_model(cfg)
        self._device, gpu_ids = self.get_device(gpu_id=cfg.get('gpu_id', 0))
        # self.model = self.model.to(self._device)
        self.classes_name = cfg["class_names"]
        self.input_size = cfg["input_size"]
        self.model.eval()

    def __call__(self, src, image_path=None):

        # process image
        image = src.copy()
        image = self.transformer(image=image)["image"]

        input = np.transpose(image, [2, 0, 1])[None, :]
        input = torch.from_numpy(input).to(device=self._device)
        # infer
        predict = self.model(input)["preds"]

        predict = predict.detach().cpu().numpy()
        result = self.get_result_img(src, predict)
        # get results visualization
        return result
    def export(self, img):
        image = img.copy()
        image = self.transformer(image=image)["image"]

        input = np.transpose(image, [2, 0, 1])[None, :]
        input = torch.from_numpy(input)
        # input = torch.randn(1, 3, 512, 512)
        fake_data = torch.randn(1, 3, self.input_size[1], self.input_size[0], device="cpu")
        self.model(fake_data, logits=False)
        torch.onnx.export(self.model, (fake_data), output_path,
            input_names=["input_image"], output_names=["output_names"] , opset_version=opset
            # ,dynamic_axes={"input_image":{0: "batch_size"}, "output_names":{0: "batch_size"},}
        )

        # net = onnx.load(output_path)
        # res = onnx.checker.check_model(net)

        inter_session = onnxruntime.InferenceSession(output_path)
        inputs = {"input_image": input.numpy()}
        outs = inter_session.run(["output_names"], inputs)[0]
        result = self.get_result_img(img, outs)
        # print(outs)
        return result

        # print(res)

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
            result = np.transpose(predict, [0, 2, 3, 1])[0]
            mask = get_pred(result)
            bgr_values = get_BGR_values(len(self.classes_name))
            ori_size = ori_image.shape[:2]
            mask = resize_mask(mask, ori_size)
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
        number_classes_model = {"classification": ["backbone"],
                                "detection": ["bbox_head"],
                                "segmentation": ["decode_head", "auxiliary_head"]
                                }
        num_classes_cfgs = [model_cfg.get(head_cfg, None) for head_cfg in number_classes_model.get(cfg.get("task"), [])]
        if num_classes_cfgs:
            for c in num_classes_cfgs:
                if c is not None:
                    c.update(dict(num_classes=len(cfg.class_names)))

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
                            default="export/gan/TwoStageInpaintor_None_ImageInpaintingDataset/2021-12-30-11-39-48/config.json",
                            type=str)
        parser.add_argument('-f', '--file', type=str, default='', help='input_image')
        parser.add_argument('--opset', type=int, default=12, help='onnx opset version')
        parser.add_argument('--output_path', type=str, default='export/model.onnx', help='onnx file save path')
        args = parser.parse_args()
        cfg = parse_config_file(args.config)
        cfg = fromfile(cfg)
        return cfg, args.file, args.opset, args.output_path

    cfg, file, opset, output_path = setup()

    if file == '':
        file = "D:\data\seg\dos_stc\\val\images\\20210925173114031_1_8_4.png"

    pred = Exporter(cfg)
    # IMAGE_FORMER = ['JPEG', 'JPG', 'JPE', 'BMP', 'PNG', 'JP2', 'PBM', 'PGM', 'PPM']
    # result_dir = os.path.join(cfg["output_dir"], "predict")
    # mkdir(result_dir)
    
    img = cv2.imread(file)
    # result = pred(img,)
    result = pred.export(img)
    print(f"predict image from {file}")
    cv2.imwrite(os.path.join("/zhubin/pt_network/output", "rpredict.jpg"), result)

    # if os.path.isdir(file):
        # image_names = [name for name in sorted(os.listdir(file)) if name.split('.')[-1].upper() in IMAGE_FORMER]

        # for image_name in image_names:
        #     image_path = os.path.join(file, image_name)
        #     img = cv2.imread(image_path)
        #     result = pred(img, image_path)
        #     print(f"predict image from {image_path}")
        #     cv2.imwrite(os.path.join(result_dir, image_name), result)
