import os
import cv2
import math
import torch
from .base_eval_hooks import BaseEvalHook
from ..metrics import *
from ..draw_plot import draw_plot, draw_per_classes
from ...engine.hooks import HOOKS


@HOOKS.registry()
class DetEvalHook(BaseEvalHook):
    def __init__(self, class_names, performance_dir, model_name, input_size,
                 metric='f1',
                 vis_score_threshold=0.3,
                 **kwargs):
        super(DetEvalHook, self).__init__(**kwargs)
        assert metric in ["f1", "accuracy", "acc"], "metric best be f1 or accuracy but got {}".format(metric)
        self.vis_score_threshold = vis_score_threshold
        self.max_number_gt = kwargs.get("max_number_gt", 100)
        self.input_size = input_size
        self._ap_per_epoch = []
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

    def evaluate(self,learning_rate, avg_losses, dataloader, model, runner,
                 threshold=None, logger=None, **kwargs):
        confidence_threshold = kwargs.get("confidence_threshold", self.vis_score_threshold)
        model.eval()
        img_paths = []
        for step, (img_batch, label_batch) in enumerate(dataloader):
            img_paths += label_batch['image_path']
            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.from_numpy(img_batch).to(runner.device, dtype=torch.float32)
            pred = model(img_batch)

            batch_label = []
            for boxes, index in zip(label_batch['bboxes'], label_batch['label_index']):
                index = np.expand_dims(index, axis=-1)
                if len(boxes) ==0 :
                    boxes = np.empty(shape=(0, 4))
                gt_label = np.concatenate([boxes, index], axis=-1)
                if gt_label.shape[0] < self.max_number_gt:
                    padding = np.zeros([self.max_number_gt - gt_label.shape[0], gt_label.shape[1]])
                    padding[...,-1] =-1
                    label = np.concatenate([gt_label, padding], axis=0)
                batch_label.append(label)

            if step ==0:
                pred_array = pred.detach().cpu().numpy()
                gt_array = np.array(batch_label)
            else:
                pred_array = np.concatenate([pred_array, pred.detach().cpu().numpy()], axis=0)
                gt_array = np.concatenate([gt_array, np.array(batch_label)], axis=0)

        y_prob = pred_array
        y_true = gt_array


        self._lr_per_epoch.append(learning_rate)
        self._avg_loss_per_epoch.append(avg_losses)

        AP_of_classes = calculate_AP_of_detections_per_class(y_prob, y_true,
                                                             class_ids=list(range(1, len(self.class_names)+1)),
                                                             iou_threshold=0.5,
                                                             confidence_threshold=confidence_threshold)

        AP_of_classes = sorted(AP_of_classes.items(), key=lambda item:item[0], reverse=False)
        AP_of_classes = np.array([v[1] for v in AP_of_classes], dtype=np.float32)

        self._ap_per_epoch.append(AP_of_classes)
        self._cur_epoch = len(self._ap_per_epoch)


        if runner.rank ==0:
            logger.info("Validation metric per class:")
            logger.info('class_name'.ljust(15,' ') + 'AP'.ljust(15,' '))

            metric_per_class = np.stack([
                AP_of_classes
            ], axis=0)
            for idx in range(self.num_classes):
                metric_list = list(metric_per_class[:, idx])
                metric_list = list(map(lambda x: '{:6.4f}'.format(x).ljust(15, ' '), metric_list))
                metric_str = ''.join(metric_list)
                logger.info(self.class_names[idx].ljust(15, ' ') + metric_str)
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
                torch.save(state, "%s/val_best.pth" % (self._checkpoint_dir))
            else:
                prefix = ''
            logger.info(prefix + 'mAP = %.4f' % np.mean(AP_of_classes))

            mean_metrics = [sum(x) / len(x) for x in self._ap_per_epoch]
            metrices = dict(mAP=mean_metrics,
                            loss=self._avg_loss_per_epoch,
                            learning_rate=self._lr_per_epoch
                            )
            draw_plot(metrices, os.path.dirname(self._metric_vs_epoch_file))

            metrics = np.stack(self._ap_per_epoch, axis=0)

            draw_per_classes("ap_per_classes", metrics, self.class_names,  os.path.dirname(self._metric_vs_epoch_file))

    def is_val_best_epoch(self):
        major_metrics = self._ap_per_epoch
        mean_metrics = [sum(x) for x in major_metrics]
        best_metric = max(mean_metrics)
        if mean_metrics[-1] >= best_metric:
            return True
        else:
            return False

    def visualize(self, img_paths, predictions):
        num_preds = len(img_paths)

        if not os.path.exists(self.vis_predictions_dir):
            os.makedirs(self.vis_predictions_dir)
        vis_paths = [os.path.join(self.vis_predictions_dir, 'pred_%03d.png'%(idx)) for idx in range(num_preds)]

        for idx in range(num_preds):
            ori_image = cv2.imread(img_paths[idx])
            ori_size = ori_image.shape[:2]
            boxes = predictions[idx]
            boxes = resize_box(ori_size , (self.input_size[1], self.input_size[0]), boxes)
            vis_result = show_detections(ori_image, boxes, self.class_names, self.vis_score_threshold)
            cv2.imencode('.jpg', vis_result)[1].tofile(vis_paths[idx])


    def after_train_epoch(self, runner):
        if not self.evaluation_flag(runner):
            return
        epoch_loss = sum(self.total_epoch_losses)/(len(self.total_epoch_losses)+1e-6)
        param_group = runner.optimizer.param_groups[0]

        self.evaluate(param_group["lr"],
                epoch_loss, self.dataloader,
                runner.model,
                runner,
                logger=runner.logger)

        self.total_epoch_losses = []

    def after_train(self, runner):
        major_metrics = self._ap_per_epoch
        mean_metrics = [sum(x) / len(x) for x in major_metrics]
        best_metric = max(mean_metrics)
        best_index = mean_metrics.index(best_metric)
        runner.logger.info("the best metric acc is %.4f, at epoch %d" % (best_metric, best_index))

@HOOKS.registry()
class PrunedDetEvalHook(DetEvalHook):
    def __init__(self, **kwargs):
        super(PrunedDetEvalHook, self).__init__(**kwargs)
        self.epochs = []
        self.flops = []
        self.sizes = []
        self.mAPs = []

    def after_train_epoch(self, runner):
        super(PrunedDetEvalHook, self).after_train_epoch(runner)

        model_params = model_info(runner.model, runner.cfg.input_size)
        runner.logger.info(model_params)

        info_list = model_params.split(" ")
        self.flops.append(float(info_list[-2]))
        self.sizes.append(float(info_list[-5]))
        self.mAPs.append(self._ap_per_epoch[-1].mean() * 100)

        metircs = np.array([self.flops, self.sizes, self.mAPs]).transpose([1, 0])
        save_path = self.vis_predictions_dir = os.path.join(self.performance_dir)
        draw_per_classes("pruned", metrics=metircs, class_names=["flops", "size", "mAP"], save_path=save_path)


def resize_box(org_size, inptut_size, boxes):
    input_h, input_w = inptut_size
    im_h, im_w = org_size
    resize_ratio = min(input_w / im_w, input_h / im_h)
    new_w = round(im_w * resize_ratio)
    new_h = round(im_h * resize_ratio)

    del_h = (input_h - new_h)/2
    del_w = (input_w - new_w)/2
    boxes = boxes[(boxes[:,2]>0) * (boxes[:,3] > 0)]

    boxes[:,0:4:2] = boxes[:,0:4:2] - del_w
    boxes[:,1:4:2] = boxes[:,1:4:2] - del_h
    boxes[:,:4] /= resize_ratio

    return boxes


def show_detections(image, detections, class_names, score_threshold=None):
    detections = detections[~np.all(detections==0, axis=1)]
    if score_threshold is not None:
        indices = np.where(detections[:, -1]>=score_threshold)[0]
        detections = detections[indices, :]
    class_ids = np.unique(detections[:, -2])
    class_ids = class_ids[class_ids>0].astype(np.int32)
    if len(class_ids)==0:
        return image
    bgr_values = get_BGR_values(len(class_names))
    # class_color_dict = dict(zip(class_ids, bgr_values))

    img_to_draw = image.copy()
    num_boxes = detections.shape[0]
    for i in range(num_boxes):
        class_id = int(detections[i][-2])
        if class_id == 0:
            continue
        confidence_score = detections[i][-1]
        box = np.array(detections[i][:-2], np.int32)
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(
            img_to_draw,
            (box[0], box[1]),
            (box[2], box[3]),
            bgr_values[class_id-1],
            2
        )

        result_txt = "%s: %.4f"%(class_names[class_id-1], confidence_score)

        ((text_width, text_height), _) = cv2.getTextSize(result_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img_to_draw, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), (255,255,255), -1)
        cv2.putText(
            img_to_draw,
            text=result_txt,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=bgr_values[class_id-1],
            lineType=cv2.LINE_AA,
        )
    return img_to_draw

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
