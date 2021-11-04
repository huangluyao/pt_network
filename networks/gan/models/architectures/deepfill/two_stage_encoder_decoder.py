import torch
import torch.nn as nn
from ...builder import build_module, MODULES
from test_utils.utils.checkpoint import load_checkpoint
from networks.base.cnn.utils import normal_init, constant_init
from torch.nn.modules.batchnorm import _BatchNorm

@MODULES.register_module()
class DeepFillEncoderDecoder(nn.Module):
    def __init__(self,
                 stage1=dict(
                     type='GLEncoderDecoder',
                     encoder=dict(type='DeepFillEncoder'),
                     decoder=dict(type='DeepFillDecoder', in_channels=128),
                     dilation_neck=dict(
                         type='GLDilationNeck',
                         in_channels=128,
                         act_cfg=dict(type='ELU'))),
                 stage2=dict(type='DeepFillRefiner'),
                 return_offset=False):
        super().__init__()
        self.stage1 = build_module(stage1)
        self.stage2 = build_module(stage2)
        self.return_offset = return_offset

    def forward(self, x):
        input_x = x.clone()
        masked_img = input_x[:, :3, ...]
        mask = input_x[:, -1:, ...]
        x = self.stage1(x)

        stage1_res = x.clone()
        stage1_img = stage1_res * mask + masked_img * (1. - mask)
        stage2_input = torch.cat([stage1_img, input_x[:, 3:, ...]], dim=1)
        stage2_res, offset = self.stage2(stage2_input, mask)

        if self.return_offset:
            return stage1_res, stage2_res, offset

        return stage1_res, stage2_res

    # TODO: study the effects of init functions
    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, 0, 0.02)
                elif isinstance(m, (_BatchNorm, nn.InstanceNorm2d)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')
