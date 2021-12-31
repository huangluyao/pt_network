from .bisenet_head import BiSeNetHead
from .deeplabv3_head import DeepLabV3Head
from .deeplabv3_plus_head import DeepLabV3PlusHead
from .fcn_head import FCNHead
from .unet_head import UNetHead
from .dla_head import DLAHead
from .stdc_head import STDCHead
from .memory_head import MemoryHead
from .dahead import DAHead
from .dla_tf_head import *
from .pra_head import PRAHead
__all__ = [k for k in globals().keys() if not k.startswith("_")]
