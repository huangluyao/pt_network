import copy
from networks.base.cnn.components import ConvModule, build_activation_layer
import torch
import torch.nn as nn


class SimpleGatedConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 feat_act_cfg=dict(type='ELU'),
                 gate_act_cfg=dict(type='Sigmoid'),
                 **kwargs
                 ):

        super(SimpleGatedConvModule, self).__init__()

        kwargs_ = copy.deepcopy(kwargs)
        kwargs_['act_cfg'] = None
        self.with_feat_act = feat_act_cfg is not None
        self.with_gate_act = gate_act_cfg is not None

        self.conv = ConvModule(in_channels, out_channels * 2, kernel_size,
                               **kwargs_)

        if self.with_feat_act:
            self.feat_act = build_activation_layer(feat_act_cfg)

        if self.with_gate_act:
            self.gate_act = build_activation_layer(gate_act_cfg)


    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.split(x, x.size(1) // 2, dim=1)
        if self.with_feat_act:
            x = self.feat_act(x)
        if self.with_gate_act:
            gate = self.gate_act(gate)
        x = x * gate
        return x