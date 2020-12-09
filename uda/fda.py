from uda.base import Model
from utils.image import FDA_source_to_target
from losses.entropy import EntropyLoss


class FDA(Model):
    def __init__(self, entropy_weight, beta, eta=1.5, use_circular=False):
        super().__init__()
        self.entropy_loss = EntropyLoss(eta=eta)
        self.entropy_weight = entropy_weight
        self.beta = beta
        self.eta = eta
        self.use_circular = use_circular

    def step(self, data, is_training=True):
        for k in data:
            data[k] = data[k].to(device=self.device, non_blocking=True)

        if is_training:
            self.optimizer.zero_grad()

        source = data["input"]
        target = data["target_domain_input"]

        mixed = FDA_source_to_target(
            source, target, self.beta, self.use_circular)
        outputs_source_domain = self.backend(mixed)
        outputs_target_domain = self.backend(target)

        outputs = {
            "source_domain": outputs_source_domain,
            "target_domain": outputs_target_domain
        }

        c_loss, c_stats = self.centernet_loss(outputs["source_domain"], data)
        e_loss, e_stats = self.entropy_loss(outputs["target_domain"], data)
        e_loss *= self.entropy_weight

        if is_training:
            c_loss.backward()
            e_loss.backward()
            self.optimizer.step()

        stats = {**c_stats, **e_stats}
        stats["total_loss"] = c_loss + e_loss

        for s in stats:
            stats[s] = stats[s].cpu().detach()

        outputs["stats"] = stats

        return outputs
