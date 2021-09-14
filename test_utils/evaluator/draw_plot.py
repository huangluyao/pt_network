# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/8/10 下午6:30
import os.path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from collections import  Counter


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

def draw_per_classes(metric_name, metrics, class_names, save_path, is_max=True):
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
            if is_max:
                draw_point(x_epoch[x], max(metrics[:, idx]), string="(%s,%.4f)" % ('max', max(metrics[:, idx])),
                       xytext=xytexts[idx])
            else:
                draw_point(x_epoch[x], min(metrics[:, idx]), string="(%s,%.4f)" % ('min', max(metrics[:, idx])),
                       xytext=xytexts[idx])

    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.grid()
    file_name = os.path.join(save_path, metric_name)
    plt.savefig(file_name)
    plt.clf()
    plt.close()


def static_bn(bn_numpy, save_path):
    round_bn = np.round(bn_numpy, 1)
    round_bn = Counter(round_bn)

    keys = sorted(round_bn.keys())

    x = np.round(np.arange(keys[0], keys[-1], 0.1),1).astype(bn_numpy.dtype)

    y = []
    index = 0
    for key in x:
        if key in keys:
            y.append(round_bn[key])
            index +=1
        else:
            y.append(0)

    plt.title("bn static")
    plt.figure(figsize=(8, 4))
    # 一个图表中画多个图
    plt.xlabel ("number")
    plt.ylabel ("count")

    plt.bar(x, y, width=0.07)
    plt.savefig(save_path)
    plt.close()

    txtresutl = save_path.replace("png", "txt")
    with open(txtresutl, "w") as f:
        for number_x, number_y in zip(x, y):
            number_x = str(number_x)
            number_y = str(number_y)

            f.write(number_x+"\t"+number_y+"\n")

