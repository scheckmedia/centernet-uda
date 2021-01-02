import torch
import torch.nn as nn
from utils.tensor import _gather_feat, _transpose_and_gather_feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    # keep = (hmax == heat).float()
    keep = 1. - torch.ceil(hmax - heat)
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_detection(heat, wh, reg=None, kps=None, K=100, rotated=False):
    batch, cat, height, width = heat.size()

    # heat = heat.sigmoid_()
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, wh.shape[-1])
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    if not rotated:
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
    else:
        bboxes = torch.cat([
            xs, ys, wh[..., 0:1], wh[..., 1:2], wh[..., 2:3]
        ], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    if kps is not None:
        kps = _transpose_and_gather_feat(kps, inds)
        kps = kps.view(batch, K, kps.size(2) // 2, 2)
        kps[..., 0] += xs
        kps[..., 1] += ys
        return detections, kps

    return detections
