# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/16 上午9:24
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from .metrics import *

class Evaluator:

    def __init__(self, task, model_name,
                 num_classes, class_names, performance_dir,
                 logger,
                 threshold=None
                 ):

        self._task = task
        self.logger = logger
        self._num_classes = num_classes
        self._class_names = class_names
        self._model_name = model_name
        self._performance_dir = performance_dir
        self._threshold = threshold
        if not os.path.exists(performance_dir):
            os.makedirs(performance_dir)

        self._lr_per_epoch = []
        self._avg_loss_per_epoch = []

        self._metric_vs_epoch_file = "%s/metric_vs_epoch_%s.png" %(self._performance_dir, self._model_name)
        self._record_file = self._metric_vs_epoch_file.replace("png", "txt")

        if task == 'classification' or task == 'segmentation':
            self._auc_per_epoch = []
            self._aupr_per_epoch = []
            self._ks_per_epoch = []
            self._best_f1_per_epoch = []
            self._precision_per_epoch = []
            self._recall_per_epoch = []
            self._f1_per_epoch = []
            self._iou_per_epoch = []
            self.record_cur_epoch = self._record_cur_epoch_cls
            if task == 'segmentation' and class_names[0] == 'background':
                self._ignore_background = True
            else:
                self._ignore_background = False
        elif task == 'detection':
            self._ap_per_epoch = []
            self.record_cur_epoch = self._record_cur_epoch_det
        else:
            self.record_cur_epoch = None
            raise TypeError('Only support classification')

    def _record_cur_epoch_cls(self, learning_rate, avg_loss, y_prob, y_true, y_feature=None, **kwargs):

        self._lr_per_epoch.append(learning_rate)
        self._avg_loss_per_epoch.append(avg_loss)

        y_prob = np.reshape(y_prob, [-1, self._num_classes])
        y_true = np.reshape(y_true, [-1,])
        if self._num_classes > 1:
            if self._threshold is not None:
                y_pred = (y_prob >= self._threshold) * 1
            else:
                y_pred = np.argmax(y_prob, axis=-1)
        else:
            y_prob = np.squeeze(y_prob, axis=-1)

        if self._num_classes==1:
            auc, aupr, ks, bestf1 = calculate_auc_aupr_ks_bestf1(y_prob, y_true)
            self._auc_per_epoch.append(auc)
            self._aupr_per_epoch.append(aupr)
            self._ks_per_epoch.append(ks)
            self._best_f1_per_epoch.append(bestf1)
            self._cur_epoch = len(self._best_f1_per_epoch)

            self.logger.info("Validation AUC = %.4f" %auc)
            self.logger.info("Validation AUPR = %.4f" %aupr)
            self.logger.info("Validation KS = %.4f" %ks)
            self.logger.info("Validation BestF1 = %.4f" %bestf1)

        else:
            precision_per_class, recall_per_class, f1_per_class1 = calculate_precision_recall_f1_per_class(
                y_pred, y_true, num_classes=self._num_classes
            )
            iou_per_class = calculate_iou_per_class(y_pred, y_true, num_classes=self._num_classes)


            self._precision_per_epoch.append(precision_per_class)
            self._recall_per_epoch.append(recall_per_class)
            self._f1_per_epoch.append(f1_per_class1)
            self._iou_per_epoch.append(iou_per_class)
            self._cur_epoch = len(self._f1_per_epoch)
            self.logger.info("Validation metric per class:")
            self.logger.info(
                            'class_name'.ljust(15,' ') +
                            'Precision'.ljust(15,' ') +
                            'Recall'.ljust(15,' ') +
                            'F1'.ljust(15,' ') +
                            'IoU'.ljust(15,' ')
                        )

            # print info
            metric_per_class = np.stack([
                precision_per_class,
                recall_per_class,
                f1_per_class1,
                iou_per_class], axis=0)
            for idx in range(self._num_classes):
                metric_list = list(metric_per_class[:,idx])
                metric_list = list(map(lambda x: '{:6.4f}'.format(x).ljust(15,' ') ,metric_list))
                metric_str = ''.join(metric_list)
                self.logger.info(self._class_names[idx].ljust(15,' ') + metric_str)

            if self.is_val_best_epoch():
                prefix = 'Best performance so far, '
            else:
                prefix = ''
            if self._task == 'segmentation':
                self.logger.info(prefix + 'mIoU = %.4f' % np.mean(iou_per_class))
            else:
                self.logger.info(prefix + 'mF1 = %.4f' % np.mean(f1_per_class1))


        xmult = 1
        if self._cur_epoch > 20:
            xmult = self._cur_epoch // 20
            rest = self._cur_epoch % 20
            if rest // xmult > 5:
                xmult += 1
        x_epoch = range(1, self._cur_epoch+1)
        xytext = (+30,+10)
        if self._num_classes == 1:
            plt.figure(figsize=(10,12))
            xytexts = [(xytext[0], xytext[1]+20*idx) for idx in range(4)]

            plt.suptitle(self._model_name,fontsize='xx-large',fontweight='heavy',verticalalignment='baseline')
            ax = plt.subplot(211)
            y_loss = np.array(self._avg_loss_per_epoch)
            if np.median(y_loss) < 2.0:
                y_loss = np.clip(y_loss, 0.0, 2.0)
            else:
                y_loss = 2 * y_loss / np.max(y_loss)
            plt.plot(x_epoch,y_loss,'',label='loss')
            plt.legend(loc='best')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            pos_loss = np.where(y_loss==min(y_loss))[0]
            if len(pos_loss) > 1:
                pos_loss = [pos_loss[-1]]
            for x in pos_loss:
                draw_point(x_epoch[x],min(y_loss),string="(%s,%.4f)"%('min',min(self._avg_loss_per_epoch)),xytext=xytext)
            xmajorLocator = MultipleLocator(xmult)
            ax.xaxis.set_major_locator(xmajorLocator)
            plt.grid()

            ax = plt.subplot(212)
            y_auc = np.array(self._auc_per_epoch)
            y_aupr = np.array(self._aupr_per_epoch)
            y_ks = np.array(self._ks_per_epoch)
            y_f1 = np.array(self._best_f1_per_epoch)

            plt.plot(x_epoch,y_auc,'',label='AUC')
            plt.plot(x_epoch,y_aupr,'',label='AUPR')
            plt.plot(x_epoch,y_ks,'',label='ks')
            plt.plot(x_epoch,y_f1,'',label='best_f1')
            plt.legend(loc='best')
            plt.xlabel('epoch')
            plt.ylabel('metric')
            plt.ylim(0,1)
            pos_auc = np.where(y_auc==max(y_auc))[0]
            if len(pos_auc) > 1:
                pos_auc = [pos_auc[-1]]
            pos_aupr = np.where(y_aupr==max(y_aupr))[0]
            if len(pos_aupr) > 1:
                pos_aupr = [pos_aupr[-1]]
            pos_ks = np.where(y_ks==max(y_ks))[0]
            if len(pos_ks) > 1:
                pos_ks = [pos_ks[-1]]
            pos_f1 = np.where(y_f1==max(y_f1))[0]
            if len(pos_f1) > 1:
                pos_f1 = [pos_f1[-1]]

            for x in pos_auc:
                draw_point(x_epoch[x],max(y_auc),string="(%s,%.4f)"%('max',max(y_auc)),xytext=xytexts[3])
            for x in pos_aupr:
                draw_point(x_epoch[x],max(y_aupr),string="(%s,%.4f)"%('max',max(y_aupr)),xytext=xytexts[2])
            for x in pos_ks:
                draw_point(x_epoch[x],max(y_ks),string="(%s,%.4f)"%('max',max(y_ks)),xytext=xytexts[1])
            for x in pos_f1:
                draw_point(x_epoch[x],max(y_f1),string="(%s,%.4f)"%('max',max(y_f1)),xytext=xytexts[0])
            xmajorLocator = MultipleLocator(xmult)
            ax.xaxis.set_major_locator(xmajorLocator)
            ymajorLocator = MultipleLocator(0.1)
            ax.yaxis.set_major_locator(ymajorLocator)
            plt.grid()
        else:
            precisions = np.stack(self._precision_per_epoch, axis=0)
            recalls = np.stack(self._recall_per_epoch, axis=0)
            if self._task == 'classification':
                metric_name = 'F1'
                metrics = np.stack(self._f1_per_epoch, axis=0)
            else:
                metric_name = 'IoU'
                metrics = np.stack(self._iou_per_epoch, axis=0)
            xytexts = [(xytext[0]+10*idx, xytext[1]+20*idx) for idx in range(self._num_classes)]

            plt.figure(figsize=(20,12))
            plt.suptitle(self._model_name,fontsize='xx-large',fontweight='heavy',verticalalignment='baseline')
            ax = plt.subplot(221)
            ymult = 0.1
            y_loss = np.array(self._avg_loss_per_epoch)
            if np.median(y_loss) < 2.0:
                y_loss = np.clip(y_loss, 0.0, 2.0)
            else:
                y_loss = 2 * y_loss / np.max(y_loss)
            plt.title('Loss')
            plt.plot(x_epoch,y_loss,'',label='loss')
            plt.legend(loc='best')
            plt.xlabel('epoch')
            plt.ylabel('Loss')
            pos_loss = np.where(y_loss==min(y_loss))[0]
            if len(pos_loss) > 1:
                pos_loss = [pos_loss[-1]]
            for x in pos_loss:
                draw_point(x_epoch[x],min(y_loss),string="(%s,%.4f)"%('min',min(self._avg_loss_per_epoch)),xytext=xytext)
            xmajorLocator = MultipleLocator(xmult)
            ax.xaxis.set_major_locator(xmajorLocator)
            ymajorLocator = MultipleLocator(ymult)
            ax.yaxis.set_major_locator(ymajorLocator)
            plt.grid()

            ax = plt.subplot(223)
            plt.title(metric_name)
            for idx in range(self._num_classes):
                plt.plot(x_epoch,metrics[:,idx],'',label=self._class_names[idx])
                pos_iou = np.where(metrics[:,idx]==max(metrics[:,idx]))[0]
                if len(pos_iou) > 1:
                    pos_iou = [pos_iou[-1]]
                for x in pos_iou:
                    draw_point(x_epoch[x],max(metrics[:,idx]),string="(%s,%.4f)"%('max',max(metrics[:,idx])),xytext=xytexts[idx])
            plt.legend(loc='best')
            plt.xlabel('epoch')
            plt.ylabel(metric_name)
            plt.ylim(0,1)
            xmajorLocator = MultipleLocator(xmult)
            ax.xaxis.set_major_locator(xmajorLocator)
            ymajorLocator = MultipleLocator(0.1)
            ax.yaxis.set_major_locator(ymajorLocator)
            plt.grid()

            ax = plt.subplot(222)
            plt.title('Precision')
            for idx in range(self._num_classes):
                plt.plot(x_epoch,precisions[:,idx],'',label=self._class_names[idx])
                pos_prec = np.where(precisions[:,idx]==max(precisions[:,idx]))[0]
                if len(pos_prec) > 1:
                    pos_prec = [pos_prec[-1]]
                for x in pos_prec:
                    draw_point(x_epoch[x],max(precisions[:,idx]),string="(%s,%.4f)"%('max',max(precisions[:,idx])),xytext=xytexts[idx])
            plt.legend(loc='best')
            plt.xlabel('epoch')
            plt.ylabel('Precision')
            plt.ylim(0,1)
            xmajorLocator = MultipleLocator(xmult)
            ax.xaxis.set_major_locator(xmajorLocator)
            ymajorLocator = MultipleLocator(0.1)
            ax.yaxis.set_major_locator(ymajorLocator)
            plt.grid()

            ax = plt.subplot(224)
            plt.title('Recall')
            for idx in range(self._num_classes):
                plt.plot(x_epoch,recalls[:,idx],'',label=self._class_names[idx])
                pos_recall = np.where(recalls[:,idx]==max(recalls[:,idx]))[0]
                if len(pos_recall) > 1:
                    pos_recall = [pos_recall[-1]]
                for x in pos_recall:
                    draw_point(x_epoch[x],max(recalls[:,idx]),string="(%s,%.4f)"%('max',max(recalls[:,idx])),xytext=xytexts[idx])
            plt.legend(loc='best')
            plt.xlabel('epoch')
            plt.ylabel('Recall')
            plt.ylim(0,1)
            xmajorLocator = MultipleLocator(xmult)
            ax.xaxis.set_major_locator(xmajorLocator)
            ymajorLocator = MultipleLocator(0.1)
            ax.yaxis.set_major_locator(ymajorLocator)
            plt.grid()

        plt.tight_layout()
        plt.savefig(self._metric_vs_epoch_file)

        with open(self._record_file, 'w') as f:
            if self._num_classes == 1:
                f.write("epoch lr loss auc aupr ks best_f1\n")
                for epoch in range(1, self._cur_epoch+1):
                    f.write(
                        str(epoch) + " " +
                        format(self._lr_per_epoch[epoch-1], '.6f') + " " +
                        format(self._avg_loss_per_epoch[epoch-1], '.4f') + " " +
                        format(self._auc_per_epoch[epoch-1], '.4f') + " " +
                        format(self._aupr_per_epoch[epoch-1], '.4f') + " " +
                        format(self._ks_per_epoch[epoch-1], '.4f') + " " +
                        format(self._best_f1_per_epoch[epoch-1], '.4f') + "\n"
                    )
            else:
                f.write("epoch lr loss | precision recall f1 iou |\n")
                for epoch in range(1, self._cur_epoch+1):
                    f.write(str(epoch) + " " + format(self._lr_per_epoch[epoch-1], '.6f'))
                    f.write(" " + format(self._avg_loss_per_epoch[epoch-1], '.4f'))
                    for idx in range(self._num_classes):
                        f.write(
                            " | " +
                            format(self._precision_per_epoch[epoch-1][idx], '.4f') + " " +
                            format(self._recall_per_epoch[epoch-1][idx], '.4f') + " " +
                            format(self._f1_per_epoch[epoch-1][idx], '.4f') + " " +
                            format(self._iou_per_epoch[epoch-1][idx], '.4f')
                        )
                    f.write("\n")

        if y_feature is not None:
            match_label = np.array([[x==y] for x in y_true for y in y_true])
            match_feature = np.array([[x, y] for x in y_feature for y in y_feature])
            metric_list = calculate_compare_precision_recall_f1(match_feature, match_label)
            metric_list = list(map(lambda x: '{:6.4f}'.format(x).ljust(15, ' '), metric_list))
            metric_str = ''.join(metric_list)
            self.logger.info('compare'.ljust(15, ' ') + metric_str)

    def is_val_best_epoch(self):

        if self._task == 'detection':
            major_metrics = self._ap_per_epoch
            mean_metrics = [sum(x) for x in major_metrics]
            best_metric = max(mean_metrics)
            if mean_metrics[-1] >= best_metric:
                return True
            else:
                return False
        else:
            if self._num_classes == 1:
                best_metric = max(self._best_f1_per_epoch)
                if self._best_f1_per_epoch[-1] >= best_metric:
                    return True
                else:
                    return False
            else:
                if self._task == 'classification':
                    major_metrics = self._f1_per_epoch
                else:
                    major_metrics = self._iou_per_epoch
                mean_metrics = [sum(x) for x in major_metrics]
                best_metric = max(mean_metrics)
                if mean_metrics[-1] >= best_metric:
                    return True
                else:
                    return False

    def is_train_best_epoch(self):
        min_loss = min(self._avg_loss_per_epoch)
        if self._avg_loss_per_epoch[-1] <= min_loss:
            return True
        else:
            return False

    def get_best_metric(self):
        if self._task == 'detection':
            major_metrics = self._ap_per_epoch
            mean_metrics = [sum(x)/len(x) for x in major_metrics]
            best_metric = max(mean_metrics)
        else:
            if self._num_classes == 1:
                best_metric = max(self._best_f1_per_epoch)
            else:
                if self._task == 'classification':
                    major_metrics = self._f1_per_epoch
                else:
                    major_metrics = self._iou_per_epoch
                mean_metrics = [sum(x)/len(x) for x in major_metrics]
                best_metric = max(mean_metrics)
        return best_metric

    def _record_cur_epoch_det(self, learning_rate,
                                avg_loss,
                                y_prob,
                                y_true,
                                confidence_threshold,
                                **kwargs):

        self._lr_per_epoch.append(learning_rate)
        self._avg_loss_per_epoch.append(avg_loss)

        AP_of_classes = calculate_AP_of_detections_per_class(y_prob, y_true, class_ids=list(range(1, len(self._class_names)+1)), iou_threshold=0.5, confidence_threshold=confidence_threshold)
        AP_of_classes = sorted(AP_of_classes.items(), key=lambda item:item[0], reverse=False)
        AP_of_classes = np.array([v[1] for v in AP_of_classes], dtype=np.float32)

        self._ap_per_epoch.append(AP_of_classes)
        self._cur_epoch = len(self._ap_per_epoch)

        self.logger.info("Validation metric per class:")
        self.logger.info('class_name'.ljust(15,' ') + 'AP'.ljust(15,' '))
        metric_per_class = np.stack([
            AP_of_classes
        ], axis=0)
        for idx in range(self._num_classes):
            metric_list = list(metric_per_class[:,idx])
            metric_list = list(map(lambda x: '{:6.4f}'.format(x).ljust(15,' ') ,metric_list))
            metric_str = ''.join(metric_list)
            self.logger.info(self._class_names[idx].ljust(15,' ') + metric_str)
        if self.is_val_best_epoch():
            prefix = 'Best performance so far, '
        else:
            prefix = ''
        self.logger.info(prefix+'mAP = %.4f' %np.mean(AP_of_classes))
        xmult = 1
        if self._cur_epoch > 20:
            xmult = self._cur_epoch // 20
            rest = self._cur_epoch % 20
            if rest // xmult > 5:
                xmult += 1
        x_epoch = range(1, self._cur_epoch+1)
        xytext = (+30,+10)
        metric_name = 'AP'
        metrics = np.stack(self._ap_per_epoch, axis=0)
        plt.figure(figsize=(10,12))
        plt.suptitle(self._model_name,fontsize='xx-large',fontweight='heavy',verticalalignment='baseline')
        xytexts = [(xytext[0]+10*idx, xytext[1]+20*idx) for idx in range(self._num_classes)]
        ymult = 0.1
        ax = plt.subplot(211)
        y_loss = np.array(self._avg_loss_per_epoch)
        y_loss = np.clip(y_loss, 0.0, 2.0)
        plt.plot(x_epoch,y_loss,'',label='loss')
        plt.legend(loc='best')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        pos_loss = np.where(y_loss==min(y_loss))[0]
        if len(pos_loss) > 1:
            pos_loss = [pos_loss[-1]]
        for x in pos_loss:
            draw_point(x_epoch[x],min(y_loss),string="(%s,%.4f)"%('min',min(y_loss)),xytext=xytext)
        xmajorLocator = MultipleLocator(xmult)
        ax.xaxis.set_major_locator(xmajorLocator)
        ymajorLocator = MultipleLocator(ymult)
        ax.yaxis.set_major_locator(ymajorLocator)
        plt.grid()
        ax = plt.subplot(212)
        plt.title(metric_name)
        for idx in range(self._num_classes):
            plt.plot(x_epoch,metrics[:,idx],'',label=self._class_names[idx])
            pos_iou = np.where(metrics[:,idx]==max(metrics[:,idx]))[0]
            if len(pos_iou) > 1:
                pos_iou = [pos_iou[-1]]
            for x in pos_iou:
                draw_point(x_epoch[x],max(metrics[:,idx]),string="(%s,%.4f)"%('max',max(metrics[:,idx])),xytext=xytexts[idx])
        plt.legend(loc='best')
        plt.xlabel('epoch')
        plt.ylabel(metric_name)
        plt.ylim(0,1)
        xmajorLocator = MultipleLocator(xmult)
        ax.xaxis.set_major_locator(xmajorLocator)
        ymajorLocator = MultipleLocator(0.1)
        ax.yaxis.set_major_locator(ymajorLocator)
        plt.grid()

        plt.tight_layout()
        plt.savefig(self._metric_vs_epoch_file)

        with open(self._record_file, 'w') as f:
            f.write("epoch lr loss | AP |\n")
            for epoch in range(1, self._cur_epoch+1):
                f.write(str(epoch) + " " + format(self._lr_per_epoch[epoch-1], '.6f'))
                f.write(" " + format(self._avg_loss_per_epoch[epoch-1], '.4f'))
                for idx in range(self._num_classes):
                    f.write(
                        " | " +
                        format(self._ap_per_epoch[epoch-1][idx], '.4f')
                    )
                f.write("\n")




def draw_point(x, y, string, xytext):
    plt.plot(x,y,"ro")
    plt.annotate(
        s=string,
        xy=(x,y),
        xycoords="data",
        xytext=xytext,
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2")
    )