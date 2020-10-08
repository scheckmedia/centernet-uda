import torch
import torch.nn.functional as F
from utils.tensor import _transpose_and_gather_feat, _sigmoid
import numpy as np


class DetectionLoss(torch.nn.Module):
    def __init__(self, hm_weight, wh_weight, off_weight, periodic=False):
        super().__init__()
        self.crit_hm = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_hw = RegL1Loss() if not periodic else PeriodicRegL1Loss()

        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight

    def forward(self, output, batch):
        hm_loss, wh_loss, off_loss = 0.0, 0.0, 0.0

        output['hm'] = _sigmoid(output['hm'])

        hm_loss += self.crit_hm(output['hm'], batch['hm'])
        wh_loss += self.crit_hw(output['wh'], batch['reg_mask'],
                                batch['ind'], batch['wh'])
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                  batch['ind'], batch['reg'])

        loss = self.hm_weight * hm_loss
        loss += self.wh_weight * wh_loss
        loss += self.off_weight * off_loss

        loss_stats = {'centernet_loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats


class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super().__init__()

    def forward(self, out, target):
        return self._neg_loss(out, target)

    def _neg_loss(self, pred, gt):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * \
            neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class RegL1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class PeriodicRegL1Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        pred *= mask
        target *= mask

        pred_wh = pred[..., 0:2]
        pred_angle = pred[..., 2:3]

        target_wh = target[..., 0:2]
        target_angle = target[..., 2:3]

        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred_wh, target_wh, size_average=False)
        loss = loss / (mask.sum() + 1e-4)

        periodic_loss = torch.abs(
            torch.remainder(
                (pred_angle - target_angle) - np.pi / 2,
                np.pi) - np.pi / 2)
        periodic_loss = periodic_loss.sum() / (mask.sum() + 1e-4)

        loss += periodic_loss
        return loss
