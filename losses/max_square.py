import torch
import torch.nn.functional as F


class MaxSquareLoss(torch.nn.Module):
    def forward(self, outputs, batch):
        loss_stats = {}

        x = outputs['hm']
        v = F.softmax(x, dim=1)
        n, c, h, w = v.size()
        loss = -torch.mean(torch.pow(v, 2)) / 2
        loss_stats['max_square_loss'] = loss
        return loss, loss_stats
