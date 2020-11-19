import torch
import torch.nn as nn


class AdventLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        loss_stats = {}
        y_t = torch.FloatTensor(y_pred.size())
        y_t.fill_(y_true)
        y_t = y_t.to(y_pred.get_device())

        advent_loss = self.crit(y_pred, y_t)
        loss_stats['advent_loss'] = advent_loss
        return advent_loss, loss_stats
