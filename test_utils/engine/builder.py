import torch
from networks.base.utils import Registry, build_from_cfg
from ..datasets import build_dataset
from .data_loader import SimpleDataLoader
from copy import deepcopy

RUNNERS = Registry('runner')
OPTIMIZER = Registry('optimizer')
for module_name in dir(torch.optim):
    if module_name.startswith('_') or module_name.islower():
        continue
    optim = getattr(torch.optim, module_name)
    OPTIMIZER.register_module(module_name, module=optim)

def build_runner(cfg, default_args=None):
    return build_from_cfg(cfg, RUNNERS, default_args=default_args)


def build_gan_loader(dataset_cfg, loader_cfg,
                     is_dist, **cfg
                     ):
    data_type = dataset_cfg.type
    train_data_path = dataset_cfg.train_data_path
    augmentations = dataset_cfg.augmentations

    train_cfg_dict = dict(type=data_type, data_path=train_data_path, augmentations=augmentations)
    train_dataset = build_dataset(train_cfg_dict)

    if is_dist:
        # 给每个rank对应的进程分配训练的样本索引
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        # 将样本索引每batch_size个元素组成一个list
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, loader_cfg["batch_size"], drop_last=True)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=loader_cfg["num_workers"],
                                                        collate_fn=train_dataset.dataset_collate)
    else:
        train_data_loader = SimpleDataLoader(train_dataset, shuffle=True,
                                                        collate_fn=train_dataset.dataset_collate, **loader_cfg)

    return train_data_loader


def build_data_loader(dataset_cfg, loader_cfg,
                      is_dist,
                      **cfg):

    data_type = dataset_cfg.type
    train_data_path = dataset_cfg.train_data_path
    val_data_path = dataset_cfg.val_data_path
    augmentations = dataset_cfg.augmentations


    train_cfg_dict = deepcopy(dataset_cfg)
    train_cfg_dict.update(dict(type=data_type, data_path=train_data_path, augmentations=augmentations))

    val_cfg_dict = deepcopy(dataset_cfg)
    val_cfg_dict.update(type=data_type, data_path=val_data_path, mode='val',augmentations=augmentations)


    train_dataset = build_dataset(train_cfg_dict)
    val_dataset = build_dataset(val_cfg_dict)


    if is_dist:
        # 给每个rank对应的进程分配训练的样本索引
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

        # 将样本索引每batch_size个元素组成一个list
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, loader_cfg["batch_size"], drop_last=True)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=loader_cfg["num_workers"],
                                                        collate_fn=train_dataset.dataset_collate)
        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=loader_cfg["batch_size"],
                                                      sampler=val_sampler,
                                                      pin_memory=True,
                                                      num_workers=loader_cfg["num_workers"],
                                                      collate_fn=train_dataset.dataset_collate)

    else:
        train_data_loader = SimpleDataLoader(train_dataset, shuffle=True,
                                                        collate_fn=train_dataset.dataset_collate, **loader_cfg)
        val_data_loader = SimpleDataLoader(val_dataset,
                                                      collate_fn=train_dataset.dataset_collate, **loader_cfg)
    return train_data_loader, val_data_loader
