import torch
import torch.nn as nn
import torch.nn.functional as F
from base.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, ConvModule, resize, _BatchNorm)
from ...specific.seg_block import SelfAttentionBlock
from ..builder import HEADS, build_loss


'''features memory'''
class FeaturesMemory(nn.Module):
    def __init__(self, num_classes, feats_channels, transform_channels, out_channels,
                 use_context_within_image=True, num_feats_per_cls=1, use_hard_aggregate=False,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 ):
        super(FeaturesMemory, self).__init__()
        assert num_feats_per_cls > 0, 'num_feats_per_cls should be larger than 0'
        # set attributes
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.transform_channels = transform_channels
        self.out_channels = out_channels
        self.num_feats_per_cls = num_feats_per_cls
        self.use_context_within_image = use_context_within_image
        self.use_hard_aggregate = use_hard_aggregate
        # init memory
        self.memory = nn.Parameter(torch.zeros(num_classes, num_feats_per_cls, feats_channels, dtype=torch.float), requires_grad=False)
        # define self_attention module
        if self.num_feats_per_cls > 1:
            self.self_attentions = nn.ModuleList()
            for _ in range(self.num_feats_per_cls):
                self_attention = SelfAttentionBlock(
                    key_in_channels=feats_channels,
                    query_in_channels=feats_channels,
                    transform_channels=transform_channels,
                    out_channels=feats_channels,
                    share_key_query=False,
                    query_downsample=None,
                    key_downsample=None,
                    key_query_num_convs=2,
                    value_out_num_convs=1,
                    key_query_norm=True,
                    value_out_norm=True,
                    matmul_norm=True,
                    with_out_project=True,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                self.self_attentions.append(self_attention)
            self.fuse_memory_conv = nn.Sequential(
                ConvModule(feats_channels * self.num_feats_per_cls, feats_channels,
                           kernel_size=1, stride=1, padding=0, bias=False,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg)
            )
        else:
            self.self_attention = SelfAttentionBlock(
                key_in_channels=feats_channels,
                query_in_channels=feats_channels,
                transform_channels=transform_channels,
                out_channels=feats_channels,
                share_key_query=False,
                query_downsample=None,
                key_downsample=None,
                key_query_num_convs=2,
                value_out_num_convs=1,
                key_query_norm=True,
                value_out_norm=True,
                matmul_norm=True,
                with_out_project=True,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        # whether need to fuse the contextual information within the input image
        self.bottleneck = nn.Sequential(
            ConvModule(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg,
                       )
        )
        if use_context_within_image:
            self.self_attention_ms = SelfAttentionBlock(
                key_in_channels=feats_channels,
                query_in_channels=feats_channels,
                transform_channels=transform_channels,
                out_channels=feats_channels,
                share_key_query=False,
                query_downsample=None,
                key_downsample=None,
                key_query_num_convs=2,
                value_out_num_convs=1,
                key_query_norm=True,
                value_out_norm=True,
                matmul_norm=True,
                with_out_project=True,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.bottleneck_ms = nn.Sequential(
                ConvModule(feats_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg,
                           )
            )
    '''forward'''
    def forward(self, feats, preds=None, feats_ms=None):
        batch_size, num_channels, h, w = feats.size()
        # extract the history features
        # --(B, num_classes, H, W) --> (B*H*W, num_classes)
        weight_cls = preds.permute(0, 2, 3, 1).contiguous()
        weight_cls = weight_cls.reshape(-1, self.num_classes)
        weight_cls = F.softmax(weight_cls, dim=-1)
        if self.use_hard_aggregate:
            labels = weight_cls.argmax(-1).reshape(-1, 1)
            onehot = torch.zeros_like(weight_cls).scatter_(1, labels.long(), 1)
            weight_cls = onehot
        # --(B*H*W, num_classes) * (num_classes, C) --> (B*H*W, C)
        selected_memory_list = []
        for idx in range(self.num_feats_per_cls):
            memory = self.memory.data[:, idx, :]
            selected_memory = torch.matmul(weight_cls, memory)
            selected_memory_list.append(selected_memory.unsqueeze(1))
        # calculate selected_memory according to the num_feats_per_cls
        if self.num_feats_per_cls > 1:
            relation_selected_memory_list = []
            for idx, selected_memory in enumerate(selected_memory_list):
                # --(B*H*W, C) --> (B, H, W, C)
                selected_memory = selected_memory.view(batch_size, h, w, num_channels)
                # --(B, H, W, C) --> (B, C, H, W)
                selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
                # --append
                relation_selected_memory_list.append(self.self_attentions[idx](feats, selected_memory))
            # --concat
            selected_memory = torch.cat(relation_selected_memory_list, dim=1)
            selected_memory = self.fuse_memory_conv(selected_memory)
        else:
            assert len(selected_memory_list) == 1
            selected_memory = selected_memory_list[0].squeeze(1)
            # --(B*H*W, C) --> (B, H, W, C)
            selected_memory = selected_memory.view(batch_size, h, w, num_channels)
            # --(B, H, W, C) --> (B, C, H, W)
            selected_memory = selected_memory.permute(0, 3, 1, 2).contiguous()
            # --feed into the self attention module
            selected_memory = self.self_attention(feats, selected_memory)
        # return
        memory_output = self.bottleneck(torch.cat([feats, selected_memory], dim=1))
        if self.use_context_within_image:
            feats_ms = self.self_attention_ms(feats, feats_ms)
            memory_output = self.bottleneck_ms(torch.cat([feats_ms, memory_output], dim=1))
        return self.memory.data, memory_output
    '''update'''
    def update(self, features, segmentation, ignore_index=255, strategy='cosine_similarity', momentum_cfg=None, learning_rate=None):
        assert strategy in ['mean', 'cosine_similarity']
        batch_size, num_channels, h, w = features.size()
        momentum = momentum_cfg['base_momentum']
        if momentum_cfg['adjust_by_learning_rate']:
            momentum = momentum_cfg['base_momentum'] / momentum_cfg['base_lr'] * learning_rate
        # use features to update memory
        segmentation = segmentation.long()
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)
        clsids = segmentation.unique()
        for clsid in clsids:
            if clsid == ignore_index: continue
            # --(B, H, W) --> (B*H*W,)
            seg_cls = segmentation.view(-1)
            # --extract the corresponding feats: (K, C)
            feats_cls = features[seg_cls == clsid]
            # --init memory by using extracted features
            need_update = True
            for idx in range(self.num_feats_per_cls):
                if (self.memory[clsid][idx] == 0).sum() == self.feats_channels:
                    self.memory[clsid][idx].data.copy_(feats_cls.mean(0))
                    need_update = False
                    break
            if not need_update: continue
            # --update according to the selected strategy
            if self.num_feats_per_cls == 1:
                if strategy == 'mean':
                    feats_cls = feats_cls.mean(0)
                elif strategy == 'cosine_similarity':
                    similarity = F.cosine_similarity(feats_cls, self.memory[clsid].data.expand_as(feats_cls))
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls = (feats_cls * weight.unsqueeze(-1)).sum(0)
                feats_cls = (1 - momentum) * self.memory[clsid].data + momentum * feats_cls.unsqueeze(0)
                self.memory[clsid].data.copy_(feats_cls)
            else:
                assert strategy in ['cosine_similarity']
                # ----(K, C) * (C, num_feats_per_cls) --> (K, num_feats_per_cls)
                relation = torch.matmul(
                    F.normalize(feats_cls, p=2, dim=1),
                    F.normalize(self.memory[clsid].data.permute(1, 0).contiguous(), p=2, dim=0),
                )
                argmax = relation.argmax(dim=1)
                # ----for saving memory during training
                for idx in range(self.num_feats_per_cls):
                    mask = (argmax == idx)
                    feats_cls_iter = feats_cls[mask]
                    memory_cls_iter = self.memory[clsid].data[idx].unsqueeze(0).expand_as(feats_cls_iter)
                    similarity = F.cosine_similarity(feats_cls_iter, memory_cls_iter)
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls_iter = (feats_cls_iter * weight.unsqueeze(-1)).sum(0)
                    self.memory[clsid].data[idx].copy_(self.memory[clsid].data[idx] * (1 - momentum) + feats_cls_iter * momentum)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, **kwargs):
        super(ASPP, self).__init__()
        align_corners, norm_cfg, act_cfg = kwargs['align_corners'], kwargs['norm_cfg'], kwargs['act_cfg']
        self.align_corners = align_corners
        self.parallel_branches = nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            if dilation == 1:
                branch = nn.Sequential(
                    ConvModule(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False,
                               norm_cfg=norm_cfg,
                               act_cfg=act_cfg)
                )
            else:
                branch = nn.Sequential(
                    ConvModule(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation,
                               dilation=dilation, bias=False,
                               norm_cfg=norm_cfg,
                               act_cfg=act_cfg)
                )
            self.parallel_branches.append(branch)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg)
        )
        self.bottleneck = nn.Sequential(
            ConvModule(out_channels * (len(dilations) + 1), out_channels, kernel_size=3,
                       stride=1, padding=1, bias=False,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
    '''forward'''
    def forward(self, x):
        size = x.size()
        outputs = []
        for branch in self.parallel_branches:
            outputs.append(branch(x))
        global_features = self.global_branch(x)
        global_features = F.interpolate(global_features, size=(size[2], size[3]), mode='bilinear', align_corners=self.align_corners)
        outputs.append(global_features)
        features = torch.cat(outputs, dim=1)
        features = self.bottleneck(features)
        return features


@HEADS.register_module()
class MemoryHead(nn.Module):

    def __init__(self,
                 in_channels,
                 feats_channels,
                 num_classes,
                 transform_channels=256,
                 features_memory_out_channels=512,
                 aspp_cfg=None,
                 memory_update_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 stage1_loss=dict(type='CrossEntropyLoss',use_sigmoid=False,loss_weight=0.4),
                 stage2_loss=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 aux_loss=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
                 train_cfg=None,
                 test_cfg=None
                 ):

        super(MemoryHead, self).__init__()
        self.memory_update_cfg = memory_update_cfg
        self.align_corners = align_corners
        self.num_classes=num_classes
        if aspp_cfg is not None:
            self.context_within_image_module = ASPP(**aspp_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.bottleneck = nn.Sequential(
            ConvModule(in_channels[-1], feats_channels, kernel_size=3, stride=1, padding=1, bias=False,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg))

        self.memory_module = FeaturesMemory(
            num_classes=num_classes,
            feats_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=features_memory_out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        self.decoder_stage1 = nn.Sequential(
            ConvModule(feats_channels, feats_channels, kernel_size=1, stride=1, padding=0, bias=False,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            nn.Conv2d(feats_channels, num_classes, kernel_size=1, stride=1, padding=0),
        )

        self.decoder_stage2 = nn.Sequential(
            ConvModule(features_memory_out_channels, features_memory_out_channels, kernel_size=1, stride=1,
                       padding=0, bias=False,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            nn.Conv2d(features_memory_out_channels, num_classes, kernel_size=1, stride=1, padding=0)
        )

        self.auxiliary_decoder = nn.Sequential(
            ConvModule(in_channels[-2], feats_channels, kernel_size=1, stride=1,
                       padding=0, bias=False,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg),
            nn.Conv2d(feats_channels, num_classes, kernel_size=1, stride=1, padding=0)
        )

        self.stage1_loss = build_loss(stage1_loss)
        self.stage2_loss = build_loss(stage2_loss)
        self.aux_loss = build_loss(aux_loss)

        self.curr_learning_rate = 1e-3
        self.curr_epoch = 0

    def update_step(self, curr_epoch, curr_learning_rate):
        self.curr_epoch = curr_epoch
        self.curr_learning_rate = curr_learning_rate

    def forward(self, x, return_stage1=False, return_memory_input=False):
        if isinstance(x, torch.Tensor):
            x = [x]
        feats_ms = self.context_within_image_module(x[-1]) if hasattr(self, 'context_within_image_module') else None
        memory_input = self.bottleneck(x[-1])

        preds_stage1 = self.decoder_stage1(memory_input)
        stored_memory, memory_output = self.memory_module(memory_input, preds_stage1, feats_ms)
        preds_stage2 = self.decoder_stage2(memory_output)

        if return_stage1 and not return_memory_input:
            return preds_stage2, preds_stage1
        elif return_memory_input and return_stage1:
            return preds_stage2, preds_stage1, memory_input
        elif return_memory_input and not return_stage1:
            return preds_stage2, return_memory_input
        else:
            return preds_stage2

    def forward_train(self, inputs, gt_semantic_seg, **kwargs):
        """Forward function for training.

        Parameters
        ----------
        inputs : list[Tensor]
            List of multi-level img features.
        gt_semantic_seg : Tensor
            Semantic segmentation masks
            used if the architecture supports semantic segmentation task.

        Returns
        -------
        dict[str, Tensor]
            a dictionary of loss components
        """
        preds_stage2, preds_stage1, memory_input  = self.forward(inputs,
                                                    return_stage1=True,
                                                    return_memory_input=True)
        loss = self.losses(preds_stage1, preds_stage2,
                           backbone_outputs=inputs,
                           seg_label=gt_semantic_seg,
                           memory_input=memory_input,
                           )
        return loss

    def forward_infer(self, inputs, **kwargs):
        """Forward function for testing.

        Parameters
        ----------
        inputs : list[Tensor]
            List of multi-level img features.

        Returns
        -------
        Tensor
            Output segmentation map.
        """
        return self.forward(inputs)

    def losses(self, pred_stage1, pred_stage2,
               backbone_outputs, seg_label,
               memory_input
               ):
        """Compute segmentation loss."""
        loss = dict()
        seg_label = seg_label.squeeze(1)
        input_size = seg_label.shape[1:]
        pred_stage2 = resize(input=pred_stage2,
                           size=input_size,
                           mode='bilinear',
                           align_corners=self.align_corners)

        preds_stage1 = resize(input=pred_stage1,
                           size=input_size,
                           mode='bilinear',
                           align_corners=self.align_corners)

        if hasattr(self, 'auxiliary_decoder'):
            backbone_outputs = backbone_outputs[:-1]
            predictions_aux = self.auxiliary_decoder(backbone_outputs[-1])
            predictions_aux = F.interpolate(predictions_aux, size=input_size, mode='bilinear',
                                            align_corners=self.align_corners)
            loss["aux_loss"] = self.aux_loss(predictions_aux, seg_label)

        with torch.no_grad():
            self.memory_module.update(
                features=F.interpolate(memory_input, size=input_size,
                                       mode='bilinear', align_corners=self.align_corners),
                segmentation=seg_label,
                learning_rate=self.curr_learning_rate,
                **self.memory_update_cfg,
            )

        loss["stage1_loss"] = self.stage1_loss(preds_stage1, seg_label)
        loss["stage2_loss"] = self.stage2_loss(pred_stage2, seg_label)

        return loss

    def init_weights(self):
        pass
