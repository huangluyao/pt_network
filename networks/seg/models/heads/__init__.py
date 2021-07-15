from .bisenet_head import BiSeNetHead
from .deeplabv3_head import DeepLabV3Head
from .deeplabv3_plus_head import DeepLabV3PlusHead
from .fcn_head import FCNHead
from .unet_head import UNetHead
from .dla_head import DLAHead
__all__ = [k for k in globals().keys() if not k.startswith("_")]
