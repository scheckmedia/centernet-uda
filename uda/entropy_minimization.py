from uda.base import Model
from losses.entropy import EntropyLoss


class EntropyMinimization(Model):
    def __init__(self, entropy_weight):
        super().__init__()
        self.entropy_loss = EntropyLoss()
        self.entropy_weight = entropy_weight

    def criterion(self, outputs, batch):
        c_loss, c_stats = self.centernet_loss(outputs["source_domain"], batch)
        e_loss, e_stats = self.entropy_loss(outputs["target_domain"], batch)

        loss = c_loss + e_loss * self.entropy_weight
        stats = {**c_stats, **e_stats}

        return loss, stats

    def step(self, data, is_training=True):
        for k in data:
            data[k] = data[k].to(device=self.device, non_blocking=True)

        if is_training:
            self.optimizer.zero_grad()

        outputs_source_domain = self.backend(data["input"])
        outputs_target_domain = self.backend(data["target_domain_input"])

        outputs = {
            "source_domain": outputs_source_domain,
            "target_domain": outputs_target_domain
        }
        loss, stats = self.criterion(outputs, data)

        if is_training:
            loss.backward()
            self.optimizer.step()

        stats["total_loss"] = loss

        for s in stats:
            stats[s] = stats[s].cpu().detach()

        outputs["stats"] = stats

        return outputs
