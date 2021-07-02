from ..builder import LOSSES

from .cross_entropy_loss import CrossEntropyLoss

LOSSES.register_module('CrossEntropyLoss', module=CrossEntropyLoss)
