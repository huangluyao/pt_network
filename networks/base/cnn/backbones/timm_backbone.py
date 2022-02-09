import torch.nn as nn
from .timm_models import create_model
from ..components import NORM_LAYERS
from .builder import BACKBONES


@BACKBONES.register_module()
class TIMMBackbone(nn.Module):
    """Wrapper to use backbones from timm library. More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_ .

    Args:
        model_name (str): Name of timm model to instantiate.
        pretrained (bool): Load pretrained weights if True.
        checkpoint_path (str): Path of checkpoint to load after
            model is initialized.
        in_channels (int): Number of input image channels. Default: 3.
        init_cfg (dict, optional): Initialization config dict
        **kwargs: Other timm & model specific arguments.
    """

    def __init__(
        self,
        model_name,
        features_only=True,
        pretrained=True,
        checkpoint_path='',
        in_channels=3,
        init_cfg=None,
        **kwargs,
    ):
        super(TIMMBackbone, self).__init__()
        if 'norm_layer' in kwargs:
            kwargs['norm_layer'] = NORM_LAYERS.get(kwargs['norm_layer'])
        self.timm_model = create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )
        filters = ('num_classes', 'num_features', 'head_conv', 'global_pool', 'fc', 'drop_rate')

        # Make unused parameters None

        for fitter in filters:
            if hasattr(self.timm_model, fitter): delattr(self.timm_model, fitter)

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True

    def forward(self, x):
        features = self.timm_model(x)
        return features

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
