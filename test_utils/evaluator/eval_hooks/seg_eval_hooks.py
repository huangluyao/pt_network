import os
import cv2
import math
import time
import numpy as np
import torch
from .base_eval_hooks import BaseEvalHook
from ..metrics import *
from ..draw_plot import draw_plot
from ...engine.hooks import HOOKS


@HOOKS.registry()
class SegEvalHook(BaseEvalHook):
    def __init__(self, class_names, performance_dir,
                 input_size,
                 model_name, metric='f1', **kwargs):
        super(SegEvalHook, self).__init__(**kwargs)
        assert metric in ["f1", "accuracy", "acc"], "metric best be f1 or accuracy but got {}".format(metric)
        self.input_size = input_size
        self.vis_score_threshold = kwargs.get("vis_score_threshold", None)
        self._auc_per_epoch = []
        self._aupr_per_epoch = []
        self._ks_per_epoch = []
        self._best_f1_per_epoch = []
        self._precision_per_epoch = []
        self._recall_per_epoch = []
        self._f1_per_epoch = []
        self._iou_per_epoch = []
        self._accuracy_per_epoch = []
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.metric = metric
        self.performance_dir = performance_dir
        self._metric_vs_epoch_file = "%s/metric_vs_epoch_%s.png" %(self.performance_dir, model_name)
        self._record_file = self._metric_vs_epoch_file.replace("png", "txt")
        self._avg_loss_per_epoch = []
        self._lr_per_epoch = []
        self.vis_predictions_dir = os.path.join(self.performance_dir, 'vis_predictions')
        self._checkpoint_dir = os.path.join(self.performance_dir, 'checkpoints')
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)
        if not os.path.exists(self.performance_dir):
            os.makedirs(self.performance_dir)

    def evaluate(self,learning_rate, avg_losses, dataloader, model, threshold=None, logger=None, **kwargs):

        model.eval()
        img_paths = []
        infer_times = []
        for step, (img_batch, label_batch) in enumerate(dataloader):
            img_paths += [img_path for img_path in label_batch['image_path']]
            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.from_numpy(img_batch).to(model.device, dtype=torch.float32)

            with torch.no_grad():
                torch.cuda.synchronize()
                time_start = time.time()
                pred = model(img_batch)["preds"]
                torch.cuda.synchronize()
                time_end = time.time()
            infer_times.append(time_end - time_start)
            if step == 0:
                pred_array = pred.detach().cpu().numpy()
                pred_array = np.transpose(pred_array, [0, 2, 3, 1])
                gt_array = label_batch['gt_masks']
            else:
                pre_array1 = np.transpose(pred.detach().cpu().numpy(), [0, 2, 3, 1])
                pred_array = np.concatenate([pred_array, pre_array1], axis=0)
                gt_array = np.concatenate([gt_array, label_batch['gt_masks']], axis=0)

        y_prob = pred_array
        y_true = gt_array

        self._lr_per_epoch.append(learning_rate)
        self._avg_loss_per_epoch.append(avg_losses)
        if self.num_classes > 1:
            if threshold is not None:
                y_pred = (y_prob >= threshold) * 1
            else:
                y_pred = np.argmax(y_prob, axis=-1)
        else:
            y_prob = np.squeeze(y_prob, axis=-1)

        logger.info("infer time for each image %.4fs" % (sum(infer_times) / len(infer_times)))
        if self.num_classes==1:
            auc, aupr, ks, bestf1 = calculate_auc_aupr_ks_bestf1(y_prob, y_true)
            self._auc_per_epoch.append(auc)
            self._aupr_per_epoch.append(aupr)
            self._ks_per_epoch.append(ks)
            self._best_f1_per_epoch.append(bestf1)
            self._cur_epoch = len(self._best_f1_per_epoch)

            logger.info("Validation AUC = %.4f" %auc)
            logger.info("Validation AUPR = %.4f" %aupr)
            logger.info("Validation KS = %.4f" %ks)
            logger.info("Validation BestF1 = %.4f" %bestf1)
        else:
            precision_per_class, recall_per_class, f1_per_class1 = calculate_precision_recall_f1_per_class(
                y_pred, y_true, num_classes=self.num_classes
            )

            iou_per_class = calculate_iou_per_class(y_pred, y_true, num_classes=self.num_classes)
            accuracy = np.sum(y_true==y_pred) /( y_pred.shape[0] * y_pred.shape[1] * y_pred.shape[2])
            self._precision_per_epoch.append(precision_per_class)
            self._recall_per_epoch.append(recall_per_class)
            self._f1_per_epoch.append(f1_per_class1)
            self._iou_per_epoch.append(iou_per_class)
            self._accuracy_per_epoch.append(accuracy)
            self._cur_epoch = len(self._f1_per_epoch)
            class_name_len = [len(name) for name in self.class_names]
            max_len = max(class_name_len)+4
            logger.info("Validation metric per class:")
            logger.info(
                            'class_name'.ljust(max_len,' ') +
                            'Precision'.ljust(15,' ') +
                            'Recall'.ljust(15,' ') +
                            'F1'.ljust(15,' ') +
                            'IoU'.ljust(15,' ')
                        )

            metric_per_class = np.stack([
                precision_per_class,
                recall_per_class,
                f1_per_class1,
                iou_per_class], axis=0)
            for idx in range(self.num_classes):
                metric_list = list(metric_per_class[:,idx])
                metric_list = list(map(lambda x: '{:6.4f}'.format(x).ljust(15,' ') ,metric_list))
                metric_str = ''.join(metric_list)
                logger.info(self.class_names[idx].ljust(max_len,' ') + metric_str)

            if self.is_val_best_epoch():
                prefix = 'Best performance so far, '
                self.visualize(img_paths, y_prob)
                logger.info("Saving checkpoint of best validation metrics.")
                # save checkpoint
                state = {
                    'arch': type(model).__name__,
                    'epoch': self._cur_epoch,
                    'state_dict': model.state_dict(),
                }
                torch.save(state, "%s/val_best.pth"%(self._checkpoint_dir))
            else:
                prefix = ''

            logger.info(prefix + 'mIoU = %.4f' % np.mean(iou_per_class))
            mean_iou_per_epoch = [np.mean(iou) for iou in self._iou_per_epoch]
            metrices = dict(accuracy=self._accuracy_per_epoch, iou=mean_iou_per_epoch,
                            loss=self._avg_loss_per_epoch,
                            learning_rate=self._lr_per_epoch
                            )
            draw_plot(metrices, os.path.dirname(self._metric_vs_epoch_file))


    def is_val_best_epoch(self):
        if self.num_classes == 1:
            if self.metric == 'f1':
                best_metric = max(self._best_f1_per_epoch)
                if self._best_f1_per_epoch[-1] >= best_metric:
                    return True
            else:
                best_metric = max(self._accuracy_per_epoch)
                if self._accuracy_per_epoch[-1] >= best_metric:
                    return True
            return False
        else:
            if self.metric == 'f1':
                major_metrics = self._f1_per_epoch
                mean_metrics = [sum(x) for x in major_metrics]
                best_metric = max(mean_metrics)
            else:
                mean_metrics = self._accuracy_per_epoch
                best_metric = max(self._accuracy_per_epoch)

            if mean_metrics[-1] >= best_metric:
                return True
            else:
                return False

    def visualize(self, img_paths, predictions):
        num_preds = len(img_paths)
        num_classes = len(self.class_names)
        if not os.path.exists(self.vis_predictions_dir):
            os.makedirs(self.vis_predictions_dir)
        vis_paths = [os.path.join(self.vis_predictions_dir, 'pred_%03d.png'%(idx)) for idx in range(num_preds)]

        bgr_values = get_BGR_values(num_classes)

        for idx in range(num_preds):
            ori_image = cv2.imread(img_paths[idx])
            mask = get_pred(predictions[idx], self.vis_score_threshold)
            image = resize_img(ori_image, (self.input_size[1], self.input_size[0]))
            vis_preds = np.zeros_like(image)
            for class_id in range(1, num_classes+1):
                vis_preds[:,:,0][mask==class_id] = bgr_values[class_id-1][0]
                vis_preds[:,:,1][mask==class_id] = bgr_values[class_id-1][1]
                vis_preds[:,:,2][mask==class_id] = bgr_values[class_id-1][2]

            mask = (mask==0)[:,:,None]
            vis_preds = (image*mask+(1-mask)*(vis_preds*0.5+image*0.5)).astype(np.uint8)
            vis_result = np.hstack((image, vis_preds))
            cv2.imwrite(vis_paths[idx], vis_result)


    def after_train(self, runner):

        mean_metrics = [sum(x) / len(x) for x in self._iou_per_epoch]
        best_iou = max(mean_metrics)
        best_index = mean_metrics.index(best_iou)
        runner.logger.info("the best metric iou is {:.4f}, at epoch {}".format(best_iou, best_index))


_RGB_LIST = [
    [255,215,0],
    [0,255,255],
    [255,0,255],
    [148,0,211],
    [0,191,255],
    [255,69,0],
    [255,105,180],
    [147,112,219],
    [220,20,60],
    [46,139,87],
    [218,165,32],
    [139,0,0],
    [0,0,128],
    [184,134,11],
    [128,128,0],
    [0,139,139],
    [255,140,0],
    [128,0,128],
    [255,0,0],
    [0,0,255],
    [255,255,0],
    [0,255,0]
]


def get_BGR_values(n_colors):
    if n_colors <= len(_RGB_LIST):
        bgr_list = [(v[2], v[1], v[0]) for v in _RGB_LIST[0:n_colors]]
    else:
        multiple = int(math.ceil(n_colors // len(_RGB_LIST)))
        rgb_list = _RGB_LIST * multiple
        bgr_list = [(v[2], v[1], v[0]) for v in rgb_list[0:n_colors]]
    return bgr_list

def get_pred(y_prob, threshold=None):
    if threshold is not None:
        y_pred = (y_prob>=threshold)*1
    else:
        y_pred = np.argmax(y_prob, axis=-1)

    if y_pred.ndim > 2:
        y_pred = np.squeeze(y_pred, axis=-1)

    return y_pred

def resize_img(image, size):
    resize_h, resize_w = size
    im_h, im_w, im_c = image.shape
    resize_ratio = min(resize_w / im_w, resize_h / im_h)
    new_w = round(im_w * resize_ratio)
    new_h = round(im_h * resize_ratio)
    im_resized = cv2.resize(image, (new_w, new_h))
    im_padding = np.full([resize_h, resize_w, im_c], 0)
    im_padding[(resize_h-new_h)//2:(resize_h-new_h)//2 + new_h, (resize_w-new_w)//2:(resize_w-new_w)//2 + new_w,  :] = im_resized
    return im_padding