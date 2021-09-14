import os
import random
import tempfile
import numpy as np
import torch
from ..utils.checkpoint import load_checkpoint
from networks.cls import build_classifier
from networks.det import build_detector
from networks.seg import build_segmentor
from test_utils.utils import init_distributed_mode, dist

def build_model(cfg, num_classes, logger):
    model_cfg = cfg.get('model')
    number_classes_model ={"classification":"backbone",
                           "detection":"bbox_head",
                           "segmentation":"decode_head"
                           }
    num_classes_cfg = model_cfg.get(number_classes_model[cfg.get("task")], None)
    if num_classes_cfg:
        num_classes_cfg.update(dict(num_classes=num_classes))
    if cfg.get('task') == "classification":
        model = build_classifier(model_cfg)
    elif cfg.get('task') == "detection":
        model_cfg["train_cfg"]["output_dir"] = cfg["output_dir"]
        model = build_detector(model_cfg)
    elif cfg.get('task') == "segmentation":
        model = build_segmentor(model_cfg)
    else:
        raise TypeError(f"task must be classification, detection or segmentation, but go {cfg['task']}")

    checkpoint_path = cfg.get("pretrained", "")
    if os.path.exists(checkpoint_path):
        checkpoint_path = os.path.expanduser(checkpoint_path)
        load_checkpoint(model, checkpoint_path, map_location='cpu', strict=True)
        logger.info(f"load checkpoint from {checkpoint_path}")

    if cfg.distributed:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        if cfg.rank == 0:
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"save temp pth at {checkpoint_path}")
        dist.barrier()
        # 指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
        dist.barrier()
        if cfg.rank == 0:
            if os.path.exists(checkpoint_path) is True:
                os.remove(checkpoint_path)
                logger.info(f"remove temp pth at {checkpoint_path}")

    return model

def update_dateset_info(pipelines, means, stds, input_size):
    if isinstance(pipelines, list):
        for augmentation in pipelines:
            if isinstance(augmentation, dict):
                if "Normalize" == augmentation.get('type', None):
                    augmentation.update(dict(mean=means, std=stds))
                elif "Resize" == augmentation.get('type', None) and input_size is not None:
                    augmentation.update(dict(width=input_size[0], height=input_size[1]))
                elif "RandomCrop" == augmentation.get('type', None) and input_size is not None:
                    augmentation.update(dict(width=input_size[0], height=input_size[1]))
    elif isinstance(pipelines, dict):
        for k, v in pipelines.items():
            update_dateset_info(v, means, stds, input_size)
    else:
        pass