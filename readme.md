## Introduction

![image](https://github.com/huangluyao/pt_network/blob/master/image/pred_001.png))

PT_network is a framwrok for deep learning experiments base on pytorch， which implements the following algorithms:

[YOLOF](https://arxiv.org/abs/2103.09460):You Only Look One-level Feature(CVPR'2021)

[FCOS](https://arxiv.org/abs/1904.01355): Fully Convolutional One-Stage Object Detection(ICCV'2019)

## DataSets

The dataset type has VOCDataset and DetectionDataset.

##### VOCDataset

Follow the VOC dataset format:

```
├─Annotations
├─ImageSets
│  └─Main
└─JPEGImages
```

The json file is configured as follows:

```
    "dataset": {
        "input_size": [640, 640],
        "type": "VOCDataset",
        "train_data_path": "~/VOCdevkit/VOC2012",
        "val_data_path": "~/VOCdevkit/VOC2012"
    }
```

##### DetectionDataset 

The annotations for the DetectionDataset is a json file generated with [Labelme](https://github.com/wkentaro/labelme).

Images path and annotations path format is as follows:

    ├─train
    │  ├─images
    │  ├─annotations
    └─val
        ├─annotations
        └─images

The json file is configured as follows:

```
"dataset": {
    "input_size": [640, 640],
    "type": "DetectionDataset",
    "train_data_path": "~/minidataset/train",
    "val_data_path": "~/minidataset/val"
},
```

## Train

you can set parameters in "tools/config/det/yolof/yolof_resnet18_voc.json"

```
python /tools/train.py --config=tools/config/det/yolof/yolof_resnet18_voc.json
```

## Result

| Model | Backbone | Neck           | FLOPS | DataSet(Close source) | AP     |
| ----- | -------- | -------------- | ----- | :-------------------: | ------ |
| YOLOF | resnet18 | DilatedEncoder | 44.1  |  Industrial Dataset   | 0.883  |
| FCOS  | resnet18 | NASFCOS_FPN    | 98.7  |  Industrial Dataset   | 0.8595 |
| FCOS  | resnet18 | FPN            | 80.2  |  Industrial Dataset   | 0.9097 |

## Special Thanks

[Detectron2](https://github.com/facebookresearch/detectron2)

[MMDetection](https://github.com/open-mmlab/mmdetection)

[YOLOF](https://github.com/chensnathan/YOLOF)

[FCOS](https://github.com/tianzhi0549/FCOS)

