import torchvision
import torch
import torch.nn as nn
import math
from torchvision.models.utils import load_state_dict_from_url
from .builder import BACKBONES

@BACKBONES.register_module()
class MobileNetV2(torchvision.models.MobileNetV2):
    model_urls = {
        'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    }

    def __init__(self,in_channels=3, out_levels=[1, 2, 3, 4, 5], out_stride=32, num_classes=None, **kwargs):
        if num_classes is not None:
            super(MobileNetV2, self).__init__(num_classes)
        else:
            super(MobileNetV2, self).__init__()
            del self.classifier
            if out_stride in [8, 16]:
                self.make_dilated(out_stride)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_levels = out_levels

        self.features[0][0].in_channels = in_channels
        out_channels= self.features[0][0].out_channels
        groups = self.features[0][0].groups
        kernel_size = self.features[0][0].kernel_size
        self.features[0][0].weight = nn.Parameter(torch.Tensor(
                                        out_channels, in_channels // groups, *kernel_size))
        nn.init.kaiming_uniform_(self.features[0][0].weight, a=math.sqrt(5))

        self.init_weights()
        pass

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        outputs = []
        for i, stage in enumerate(stages):
            x = stage(x)
            if i in self.out_levels:
                outputs.append(x)

        if self.num_classes is not None:
            x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
            x = self.classifier(x)
            return x

        return outputs

    def load_state_dict(self, state_dict, **kwargs):
        super().load_state_dict(state_dict, **kwargs)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Parameters
        ----------
        pretrained : str, optional
            Path to pre-trained weights.
            Defaults to None.
        """
        state_dict = load_state_dict_from_url(self.model_urls['mobilenet_v2'])
        model_dict = self.state_dict()
        state_dict_v = [state_dict[k] for k in state_dict]
        for i, k in enumerate(model_dict):
            if model_dict[k].shape == state_dict_v[i].shape:
                model_dict[k] = state_dict_v[i]
        self.load_state_dict(model_dict)

    def make_dilated(self, output_stride):

        if output_stride == 16:
            stage_list = [5, ]
            dilation_list = [2, ]

        elif output_stride == 8:
            stage_list = [4, 5]
            dilation_list = [2, 4]

        else:
            raise ValueError("Output stride should be 16 or 8, got {}.".format(output_stride))

        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )


def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Kostyl for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()
