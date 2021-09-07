import os
import torch
import random
import tempfile
import numpy as np
from networks.cls import build_classifier
from networks.det import build_detector
from networks.seg import build_segmentor
from test_utils.utils import init_distributed_mode, dist
from test_utils.datasets import build_dataset, statistics_data
from ..utils.checkpoint import load_checkpoint
from ..engine.data_loader import build_data_loader
from ..evaluator import model_info
from ..engine.optimizer import build_optimizer, build_scheduler
from ..engine.simple_trainer import SimplerTrainer

def cfg2trainer(cfg, logger):

    # 设置随机种子
    seed = cfg.get("seed", None)
    if seed is not None:
        logger.info('state random seed : %d' % (seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的   　　
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子；
        torch.backends.cudnn.deterministic = True

    # 初始化训练设备
    init_distributed_mode(cfg)

    # 统计数据集信息
    input_size = cfg.dataset.pop("input_size")
    logger.info('-' * 25 + 'statistics_data:' + '-' * 25)
    if cfg["dataset"].get("statistics_data", None):
        means = cfg["dataset"].get("statistics_data", None).get("means")
        stds = cfg["dataset"].get("statistics_data", None).get("stds")
    else:
        means, stds = statistics_data([cfg['dataset']['train_data_path'], cfg['dataset']['val_data_path']],
                                      cfg['dataset']['type'])

    update_dateset_info(cfg['dataset'], means, stds, input_size)
    # 创建dataloader
    train_dataloader, val_dataloader = build_data_loader(dataset_cfg=cfg.dataset,
                                   loader_cfg=cfg['loader_cfg'],
                                   is_dist=cfg.distributed)
    logger.info(f'data means: {means}')
    logger.info(f'data stds: {stds}')
    logger.info("categories: %s", ", ".join(train_dataloader.dataset.class_names))

    # 创建网络模型
    logger.info('-' * 25 + 'model info' + '-' * 25)
    if cfg.distributed == False:
        logger.info('Not using distributed mode')
    else:
        logger.info('distributed init (rank {}): {}'.format(
                        cfg.rank, cfg.dist_url), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, len(train_dataloader.dataset.class_names), logger).to(device)
    model.class_names = train_dataloader.dataset.class_names
    model.device = device

    # 只有训练带有BN结构的网络时使用SyncBatchNorm
    if cfg.distributed:
        # 使用SyncBatchNorm后训练会更耗时
        mode = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        # 转为DDP模型
        mode = torch.nn.parallel.DistributedDataParallel(mode, device_ids=[cfg.gpus])

    logger.info(model_info(model, input_size))

    # 创建优化器和学习率下降策略
    optimizer = build_optimizer(model, cfg['optimizer'])

    trainer = SimplerTrainer(
        train_dataloader=train_dataloader,
        model=model,
        optimizer=optimizer,
        work_dir=cfg["output_dir"],
        logger=logger,
        max_epochs=cfg.max_epochs,
        max_iters=cfg.max_iters
    )

    # 加载网络需要的钩子
    for hook_cfg in cfg.hook_cfgs:
        # 如果钩子是验证集
        if "EvalHook" in hook_cfg["type"]:
            hook_cfg["class_names"] = model.class_names
            hook_cfg["performance_dir"] = cfg.output_dir
            hook_cfg["model_name"] = type(model).__name__
            hook_cfg["dataloader"] = val_dataloader
            hook_cfg["input_size"] = input_size
        priority = hook_cfg.get("priority", "NORMAL")
        trainer.register_hook(hook_cfg, priority)

    return trainer

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
        dist.barrier()
        # 指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
        if cfg.rank == 0:
            if os.path.exists(checkpoint_path) is True:
                os.remove(checkpoint_path)
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