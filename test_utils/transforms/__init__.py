from .registry import Registry

TRANSFORM = Registry('Transform_registry')
from .compose import Compose
from .augmentations import *

