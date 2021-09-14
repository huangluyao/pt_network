import os
import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = dist.get_world_size() if is_dist_avail_and_initialized() else 1
    if world_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value

def init_distributed_mode(cfg):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        cfg.gpu = int(os.environ['LOCAL_RANK'])
        cfg.distributed = True
    elif 'SLURM_PROCID' in os.environ:
        cfg.rank = int(os.environ['SLURM_PROCID'])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
        cfg.distributed = True
    else:
        cfg.distributed = False
        cfg.rank = 0
        return

    cfg.distributed = True

    torch.cuda.set_device(cfg.gpu)
    cfg.dist_backend = 'nccl'
    cfg.dist_url = cfg.get("dist_url", "env://")
    cfg.world_size = cfg.get("world_size", 1)
    dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                            world_size=cfg.world_size, rank=cfg.rank)
    dist.barrier()

