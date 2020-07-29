import torch
import torch.nn.functional as F


class EntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, batch):
        num_stacks = len(outputs)
        loss_stats = {}

        for s in range(num_stacks):
            x = outputs[s]['hm']
            v = F.softmax(x, dim=1)
            n, c, h, w = v.size()
            entropy_loss = -torch.sum(
                torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * torch.log2(torch.Tensor([c]).to(x.device)))

        loss_stats['entropy_loss'] = entropy_loss
        return entropy_loss, loss_stats
