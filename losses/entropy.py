import torch
import torch.nn.functional as F


class EntropyLoss(torch.nn.Module):
    def __init__(self, eta=None):
        super().__init__()
        self.eta = eta

    def forward(self, outputs, batch):
        loss_stats = {}
        entropy_loss = 0.0

        x = outputs['hm']
        v = F.softmax(x, dim=1)
        n, c, h, w = v.size()
        if self.eta is not None:
            ent = -1.0 * torch.mul(v, torch.log2(v + 1e-30)).sum(dim=1)
            ent /= torch.log2(torch.Tensor([c]).to(x.device))
            ent = ent ** 2.0 + 1e-30
            ent = ent ** self.eta
            entropy_loss = ent.mean()
        else:
            entropy_loss += -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (
                n * h * w * torch.log2(torch.Tensor([c]).to(x.device))).squeeze()

        loss_stats['entropy_loss'] = entropy_loss
        return entropy_loss, loss_stats
