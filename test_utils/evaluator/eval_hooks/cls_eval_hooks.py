import os
import cv2
import torch
from .base_eval_hooks import BaseEvalHook
from ..metrics import *
from ..draw_plot import draw_plot
from ...engine.hooks import HOOKS


@HOOKS.registry()
class ClsEvalHook(BaseEvalHook):
    def __init__(self, class_names, performance_dir, model_name, metric='f1', **kwargs):
        super(ClsEvalHook, self).__init__(**kwargs)
        assert metric in ["f1", "accuracy", "acc"], "metric best be f1 or accuracy but got {}".format(metric)

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
        for step, (img_batch, label_batch) in enumerate(dataloader):
            img_paths += [img_path for img_path in label_batch['image_path']]
            if not isinstance(img_batch, torch.Tensor):
                img_batch = torch.from_numpy(img_batch).to(model.device, dtype=torch.float32)
            pred = model(img_batch)
            if step ==0:
                pred_array = pred.detach().cpu().numpy()
                gt_array = label_batch['gt_labels']
            else:
                pred_array = np.concatenate([pred_array, pred.detach().cpu().numpy()], axis=0)
                gt_array = np.concatenate([gt_array, label_batch['gt_labels']], axis=0)

        y_prob = np.reshape(pred_array, [-1, self.num_classes])
        y_true = np.reshape(gt_array, [-1,])


        self._lr_per_epoch.append(learning_rate)
        self._avg_loss_per_epoch.append(avg_losses)
        if self.num_classes > 1:
            if threshold is not None:
                y_pred = (y_prob >= threshold) * 1
            else:
                y_pred = np.argmax(y_prob, axis=-1)
        else:
            y_prob = np.squeeze(y_prob, axis=-1)

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
            accuracy = np.sum(y_true==y_pred) / len(y_pred)
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
                logger.info(model.class_names[idx].ljust(max_len,' ') + metric_str)

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

            logger.info(prefix + 'mF1 = %.4f, accuracy = %.4f' % (np.mean(f1_per_class1), accuracy))
            mean_f1_per_epoch = [np.mean(f1) for f1 in self._f1_per_epoch]
            metrices = dict(accuracy=self._accuracy_per_epoch, f1=mean_f1_per_epoch,
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
        if not os.path.exists(self.vis_predictions_dir):
            os.makedirs(self.vis_predictions_dir)
        vis_paths = [os.path.join(self.vis_predictions_dir, 'pred_%03d.png'%(idx)) for idx in range(num_preds)]
        for idx in range(num_preds):
            ori_image = cv2.imread(img_paths[idx])
            index = np.argmax(predictions[idx])
            result_txt = "%s: %.4f" % (self.class_names[index], predictions[idx][index])

            ((text_width, text_height), _) = cv2.getTextSize(result_txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            cv2.putText(ori_image, result_txt, (10, text_height+10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 255, 0))
            cv2.imencode('.jpg', ori_image)[1].tofile(vis_paths[idx])


    def after_train(self, runner):
        if self.num_classes == 1:
            best_F1 = max(self._best_f1_per_epoch)
            best_index = self._best_f1_per_epoch.index(best_F1)
        else:
            mean_metrics = [sum(x) / len(x) for x in self._f1_per_epoch]
            best_F1 = max(mean_metrics)
            best_index = mean_metrics.index(best_F1)

        best_acc = max(self._accuracy_per_epoch)
        best_acc_index = self._accuracy_per_epoch.index(best_acc)
        runner.logger.info(f"the best metric F1 is {best_F1}, at epoch {best_index}")
        runner.logger.info(f"the best metric acc is {best_acc}, at epoch {best_acc_index}")

