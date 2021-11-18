import torch
import torch.nn as nn
import torch.nn.functional as F
from ....base.cnn.utils.weight_init import constant_init
from ....base.cnn.components import ConvModule


class SelfAttentionBlock(nn.Module):

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out,
                 norm_cfg, act_cfg, conv_cfg=None):
        super(SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context


# class SelfAttentionBlock(nn.Module):
#     def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels, share_key_query,
#                  query_downsample, key_downsample, key_query_num_convs, value_out_num_convs, key_query_norm,
#                  value_out_norm, matmul_norm, with_out_project, **kwargs):
#         super(SelfAttentionBlock, self).__init__()
#         norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
#         # key project
#         self.key_project = self.buildproject(
#             in_channels=key_in_channels,
#             out_channels=transform_channels,
#             num_convs=key_query_num_convs,
#             use_norm=key_query_norm,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg,
#         )
#         # query project
#         if share_key_query:
#             assert key_in_channels == query_in_channels
#             self.query_project = self.key_project
#         else:
#             self.query_project = self.buildproject(
#                 in_channels=query_in_channels,
#                 out_channels=transform_channels,
#                 num_convs=key_query_num_convs,
#                 use_norm=key_query_norm,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg,
#             )
#         # value project
#         self.value_project = self.buildproject(
#             in_channels=key_in_channels,
#             out_channels=transform_channels if with_out_project else out_channels,
#             num_convs=value_out_num_convs,
#             use_norm=value_out_norm,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg,
#         )
#         # out project
#         self.out_project = None
#         if with_out_project:
#             self.out_project = self.buildproject(
#                 in_channels=transform_channels,
#                 out_channels=out_channels,
#                 num_convs=value_out_num_convs,
#                 use_norm=value_out_norm,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg,
#             )
#         # downsample
#         self.query_downsample = query_downsample
#         self.key_downsample = key_downsample
#         self.matmul_norm = matmul_norm
#         self.transform_channels = transform_channels
#     '''forward'''
#     def forward(self, query_feats, key_feats):
#         batch_size = query_feats.size(0)
#         query = self.query_project(query_feats)
#         if self.query_downsample is not None: query = self.query_downsample(query)
#         query = query.reshape(*query.shape[:2], -1)
#         query = query.permute(0, 2, 1).contiguous()
#         key = self.key_project(key_feats)
#         value = self.value_project(key_feats)
#         if self.key_downsample is not None:
#             key = self.key_downsample(key)
#             value = self.key_downsample(value)
#         key = key.reshape(*key.shape[:2], -1)
#         value = value.reshape(*value.shape[:2], -1)
#         value = value.permute(0, 2, 1).contiguous()
#         sim_map = torch.matmul(query, key)
#         if self.matmul_norm:
#             sim_map = (self.transform_channels ** -0.5) * sim_map
#         sim_map = F.softmax(sim_map, dim=-1)
#         context = torch.matmul(sim_map, value)
#         context = context.permute(0, 2, 1).contiguous()
#         context = context.reshape(batch_size, -1, *query_feats.shape[2:])
#         if self.out_project is not None:
#             context = self.out_project(context)
#         return context
#
#     '''build project'''
#     def buildproject(self, in_channels, out_channels, num_convs, use_norm, norm_cfg, act_cfg):
#         if use_norm:
#             convs = [
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(inplace=True)
#                 )
#             ]
#             for _ in range(num_convs - 1):
#                 convs.append(
#                     nn.Sequential(
#                         nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#                         nn.BatchNorm2d(out_channels),
#                         nn.ReLU(inplace=True)
#                     )
#                 )
#         else:
#             convs = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
#             for _ in range(num_convs - 1):
#                 convs.append(
#                     nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#                 )
#         if len(convs) > 1: return nn.Sequential(*convs)
#         return convs[0]
#
#     def init_weights(self):
#         """Initialize weight of later layer."""
#         if self.out_project is not None:
#             if not isinstance(self.out_project, nn.Conv2d):
#                 constant_init(self.out_project, 0)
