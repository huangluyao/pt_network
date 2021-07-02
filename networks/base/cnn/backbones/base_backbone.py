from abc import ABCMeta, abstractmethod
import torch.nn as nn

class BaseBackbone(nn.Module, metaclass=ABCMeta):
    """Base backbone.

    This class defines the basic functions of a backbone.
    Any backbone that inherits this class should at least
    define its own `forward` function.

    """

    def __init__(self):
        super(BaseBackbone, self).__init__()

    def init_weights(self, pretrained=None):
        """Init backbone weights

        Parameters
        ----------
        pretrained : str | None
            If pretrained is a string, then it
            initializes backbone weights by loading the pretrained
            checkpoint. If pretrained is None, then it follows default
            initializer or customized initializer in subclasses.
        """
        pass

    @abstractmethod
    def forward(self, x):
        """Forward computation

        Parameters
        ----------
        x : tensor | tuple[tensor]
            x could be a Torch.tensor or a tuple of
            Torch.tensor, containing input data for forward computation.
        """
        pass

    def train(self, mode=True):
        """Set module status before forward computation

        Parameters
        ----------
        mode : bool
            Whether it is train_mode or test_mode
        """
        super(BaseBackbone, self).train(mode)
