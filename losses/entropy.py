import torch
import torch.nn.functional as F


class EntropyLoss(torch.nn.Module):
    def forward(self, outputs, batch):
        loss_stats = {}
        entropy_loss = 0.0

        x = outputs['hm']
        v = F.softmax(x, dim=1)
        n, c, h, w = v.size()
        entropy_loss += -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (
            n * h * w * torch.log2(torch.Tensor([c]).to(x.device))).squeeze()

        loss_stats['entropy_loss'] = entropy_loss
        return entropy_loss, loss_stats
