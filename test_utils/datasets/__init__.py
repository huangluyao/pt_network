from ..utils.registry import Registry, build_from_cfg
DATASET = Registry('Dataset_registry')
from .statistics_data import statistics_data
from .classifcation_dataset import *
from .detection_dataset import *
from .segmentation_dataset import *

def build_dataset(cfg):
    return build_from_cfg(cfg, DATASET)