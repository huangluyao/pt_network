import os
import torch
from test_utils.datasets import build_dataset, statistics_data
from .utils import update_dateset_info, build_model, set_random_seed
from ..engine.data_loader import build_data_loader, build_gan_loader
from ..evaluator import model_info
from ..engine.optimizer import build_optimizer, build_scheduler, build_optimizers
from ..engine import SimplerTrainer, DynamicIterTrainer

def cfg2trainer(cfg, logger):

    # 初始化训练设备
    # init_distributed_mode(cfg)
    set_random_seed(2021)

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
    cfg.input_size = input_size
    # 创建dataloader
    if cfg.task == "gan":
        train_dataloader = build_gan_loader(dataset_cfg=cfg.dataset, loader_cfg=cfg['loader_cfg'],
                                            is_dist=cfg.distributed
                                            )
    else:
        train_dataloader, val_dataloader = build_data_loader(dataset_cfg=cfg.dataset,
                                   loader_cfg=cfg['loader_cfg'],
                                   is_dist=cfg.distributed)
    if hasattr(train_dataloader.dataset, 'class_names'):
        cfg.class_names = train_dataloader.dataset.class_names
    else:
        cfg.class_names = []
    if cfg.rank == 0:
        logger.info(f'data means: {means}')
        logger.info(f'data stds: {stds}')
        if len(cfg.class_names):
            logger.info("categories: %s", ", ".join(train_dataloader.dataset.class_names))
        # 创建网络模型
        logger.info('-' * 25 + 'model info' + '-' * 25)
    if cfg.distributed == False:
        logger.info('Not using distributed mode')
    elif cfg.rank==0:
        logger.info('distributed init (rank {}): {}'.format(cfg.rank, cfg.dist_url))
    else:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, len(cfg.class_names), logger).to(device)
    model.device = device

    # 只有训练带有BN结构的网络时使用SyncBatchNorm
    if cfg.distributed:
        # 使用SyncBatchNorm后训练会更耗时
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        # 转为DDP模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu],
                                                          find_unused_parameters=True
                                                          )

    logger.info(model_info(model, input_size))



    if cfg.task == "gan":

        optimizers = build_optimizers(model, cfg.optimizers)


        trainer = DynamicIterTrainer(cfg=cfg,
                device=device,
                train_dataloader=train_dataloader,
                optimizer=optimizers,
                model=model,
                work_dir=cfg["output_dir"],
                logger=logger,
                max_epochs=cfg.max_epochs,
                max_iters=cfg.max_iters,
                rank=cfg.rank)
    else:
        # 创建优化器和学习率下降策略
        optimizer = build_optimizer(model, cfg['optimizer'])
        trainer = SimplerTrainer(
            cfg=cfg,
            device=device,
            train_dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            work_dir=cfg["output_dir"],
            logger=logger,
            max_epochs=cfg.max_epochs,
            max_iters=cfg.max_iters,
            rank=cfg.rank)

    # 加载网络需要的钩子
    for hook_cfg in cfg.hook_cfgs:
        # 如果钩子是验证集
        if "EvalHook" in hook_cfg["type"]:
            hook_cfg["class_names"] = train_dataloader.dataset.class_names
            hook_cfg["performance_dir"] = cfg.output_dir
            hook_cfg["model_name"] = type(model).__name__
            hook_cfg["dataloader"] = val_dataloader
            hook_cfg["input_size"] = input_size
        priority = hook_cfg.get("priority", "NORMAL")
        trainer.register_hook(hook_cfg, priority)

    return trainer

