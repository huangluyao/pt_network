import torch


def bbox_overlaps(bboxes1, bboxes2, xyxy=True, mode='iou', is_aligned=False, eps=1e-6):

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]

    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)

    if is_aligned:
        assert rows == cols

    if rows * cols ==0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    if xyxy:
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1])
    else:
        area1 = torch.prod(bboxes1[..., -2:], 1)
        area2 = torch.prod(bboxes2[..., -2:], 1)

    if is_aligned:
        if xyxy:
            lt = torch.max(bboxes1[...,:2], bboxes2[...,:2])
            rb = torch.min(bboxes1[...,2:], bboxes2[...,2:])
        else:
            lt = torch.max(
                (bboxes1[:, :2] - bboxes1[:, 2:] / 2), (bboxes1[:, :2] - bboxes1[:, 2:] / 2)
            )
            rb = torch.min(
                (bboxes2[:, :2] + bboxes2[:, 2:] / 2), (bboxes2[:, :2] + bboxes2[:, 2:] / 2)
            )

        wh = torch.clamp(rb-lt, min=0)
        overlap = wh[...,0] * wh[...,1]

        union = area2 + area1 - overlap

        if mode == "giou":
            enclosed_lb = torch.min(bboxes1[...,:2], bboxes2[...,:2])
            enclosed_rb = torch.max(bboxes1[...,2:], bboxes2[...,2:])
    else:
        if xyxy:
            lt = torch.max(bboxes1[..., :, None, :2],
                           bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
            rb = torch.min(bboxes1[..., :, None, 2:],
                           bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]
        else:
            lt = torch.max(
                (bboxes1[:, None, :2] - bboxes1[:, None, 2:] / 2),
                (bboxes2[:, :2] - bboxes2[:, 2:] / 2),
            )
            rb = torch.min(
                (bboxes1[:, None, :2] + bboxes1[:, None, 2:] / 2),
                (bboxes2[:, :2] + bboxes2[:, 2:] / 2),
            )

        wh = (rb-lt).clamp(min=0, max=None)

        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode == 'iou':
        return ious

    #claculate gious
    encolse_wh = torch.clamp(enclosed_rb - enclosed_lb, min=0)
    enclosed_eara = encolse_wh[..., 0] * encolse_wh[..., 1]
    enclosed_eara = torch.max(enclosed_eara, eps)
    gious = ious - (enclosed_eara - union) / enclosed_eara
    return gious
