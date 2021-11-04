import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder_decoder import EncoderDecoder
from ..builder import SEGMENTORS, build_loss

@SEGMENTORS.register_module()
class MscaleEncoderDecoder(EncoderDecoder):
    """
    Multi-scale attention segmentation model base class
    """
    def __init__(self,
                 *args,
                 n_scale=[0.5, 1.0, 2.0],
                 low_scale = 0.5,
                 mscale_loss=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 **kwargs
                 ):
        super(MscaleEncoderDecoder, self).__init__(*args, **kwargs)
        self.low_scale = low_scale
        self.n_scale= n_scale

        with torch.no_grad():
            in_channel = kwargs['backbone'].get("in_channels")
            x = torch.zeros([2, in_channel, 256, 256])
            p_lo, feats_lo = self._fwd(x)
            scale_in_ch = int(feats_lo.shape[1] * 2)

        self.scale_attn = nn.Sequential(
            nn.Conv2d(scale_in_ch, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.Sigmoid())

        self.criterion = build_loss(mscale_loss)
        self.type = kwargs["decode_head"].get("type")

    def _fwd(self, x):
        x = self.extract_feat(x)
        return self.decode_head(x, return_last_feature=True)

    def nscale_forward(self, inputs, scales):
        """
        Hierarchical attention, primarily used for getting best inference
        results.

        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:

              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint

        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.

        Inputs:
          scales - a list of scales to evaluate
          inputs - dict containing 'images', the input, and 'gts', the ground
                   truth mask

        Output:
          If training, return loss, else return prediction + attention
        """
        x_1x = inputs

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)
        pred = None
        last_feats = None

        for idx, s in enumerate(scales):
            x = F.interpolate(x_1x, scale_factor=s, mode="bilinear",
                              align_corners=False,
                              recompute_scale_factor=True)

            p, feats = self._fwd(x)

            # Generate attention prediction
            if idx > 0:
                assert last_feats is not None
                # downscale feats
                last_feats = scale_as(last_feats, feats)
                cat_feats = torch.cat([feats, last_feats], 1)
                attn = self.scale_attn(cat_feats)
                attn = scale_as(attn, p)

            if pred is None:
                # This is the top scale prediction
                pred = p
            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, p)
                pred = attn * p + (1 - attn) * pred
            else:
                # upscale current
                p = attn * p
                p = scale_as(p, pred)
                attn = scale_as(attn, pred)
                pred = p + (1 - attn) * pred

            last_feats = feats

        return scale_as(pred, inputs)

    def two_scale_forward(self, inputs, ground_truth):

        x_1x = inputs
        x_lo = F.interpolate(x_1x,
                             scale_factor=self.low_scale,
                             mode="bilinear",
                             align_corners=False,
                             recompute_scale_factor = True)

        p_los, feats_lo = self._fwd(x_lo)
        p_1xs, feats_hi = self._fwd(x_1x)

        if isinstance(p_los, list):
            if "STDC" in self.type :
                p_lo = p_los[-2]
                p_1x = p_1xs[-2]
            else:
                p_lo = p_los[-1]
                p_1x = p_1xs[-1]
        else:
            p_lo = p_los
            p_1x = p_1xs

        feats_hi = scale_as(feats_hi, feats_lo)
        cat_feats = torch.cat([feats_lo, feats_hi], 1)
        logit_attn = self.scale_attn(cat_feats)
        logit_attn = scale_as(logit_attn, p_lo)

        p_lo = logit_attn * p_lo
        p_lo = scale_as(p_lo, p_1x)
        logit_attn = scale_as(logit_attn, p_1x)
        joint_pred = p_lo + (1 - logit_attn) * p_1x

        if self.training:
            gt_masks = torch.from_numpy(ground_truth['gt_masks']).to(inputs.device, dtype=inputs.dtype)
            loss = self.decode_head.losses(p_1xs, gt_masks)
            joint_pred = scale_as(joint_pred, inputs)
            loss["mscale_loss"] = self.criterion(joint_pred, gt_masks)
            return loss
        else:
            # FIXME: should add multi-scale values for pred and attn
            return {'pred': joint_pred,
                    'attn_10x': logit_attn}

    def forward_train(self, inputs, ground_truth):

        return self.two_scale_forward(inputs, ground_truth)

    def forward_infer(self, inputs):
        return self.nscale_forward(inputs, self.n_scale)


def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    y_size = y.size(2), y.size(3)

    x_scaled = torch.nn.functional.interpolate(
        x, size=y_size, mode='bilinear',
        align_corners=False)

    return x_scaled