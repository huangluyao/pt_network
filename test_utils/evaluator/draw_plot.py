# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/8/10 下午6:30
import os.path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


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


def draw_plot(metrics, save_path):
    xytext = (+30, +10)
    for key in metrics.keys():
        plt.figure(figsize=(20, 12))
        plt.title(key)
        x_epoch = range(1, len(metrics[key])+1)
        plt.plot(x_epoch, metrics[key], '', label=key)

        if key == "loss":
            pos_index = metrics[key].index(min(metrics[key]))
            xxx = "min"
        else:
            pos_index = metrics[key].index(max(metrics[key]))
            xxx = "max"
        draw_point(pos_index+1, metrics[key][pos_index],
                   string="(%s,%.4f)" % (xxx, metrics[key][pos_index]),
                   xytext=xytext)
        plt.legend(loc='best')
        plt.xlabel('epoch')
        plt.ylabel(key)
        plt.grid()
        file_name = os.path.join(save_path, key)
        plt.savefig(file_name)
        plt.clf()
        plt.close()

def draw_per_classes(metric_name, metrics, class_names, save_path):
    plt.figure(figsize=(20, 12))
    plt.title(metric_name)
    x_epoch = range(1, len(metrics) + 1)
    xytext = (+30, +10)
    for idx in range(len(class_names)):
        xytexts = [(xytext[0] + 10 * idx, xytext[1] + 20 * idx) for idx in range(len(class_names))]
        plt.plot(x_epoch, metrics[:, idx], '', label=class_names[idx])
        pos_iou = np.where(metrics[:, idx] == max(metrics[:, idx]))[0]
        if len(pos_iou) > 1:
            pos_iou = [pos_iou[-1]]
        for x in pos_iou:
            draw_point(x_epoch[x], max(metrics[:, idx]), string="(%s,%.4f)" % ('max', max(metrics[:, idx])),
                       xytext=xytexts[idx])

    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.grid()
    file_name = os.path.join(save_path, metric_name)
    plt.savefig(file_name)
    plt.clf()
    plt.close()


