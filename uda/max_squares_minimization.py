from uda.base import Model
from losses.max_square import MaxSquareLoss


class MaxSquaresMinimization(Model):
    def __init__(self, max_squares_weight):
        super().__init__()
        self.max_squares_loss = MaxSquareLoss()
        self.max_squares_weight = max_squares_weight

    def criterion(self, outputs, batch):
        s_loss, s_stats = self.centernet_loss(outputs["source_domain"], batch)
        t_loss, t_stats = self.max_squares_loss(
            outputs["target_domain"], batch)
        t_loss *= self.max_squares_weight

        # loss = c_loss + e_loss * self.max_squares_weight
        stats = {**s_stats, **t_stats}

        return s_loss, t_loss, stats

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
        s_loss, t_loss, stats = self.criterion(outputs, data)

        if is_training:
            s_loss.backward()
            t_loss.backward()
            self.optimizer.step()

        stats["total_loss"] = s_loss + t_loss

        for s in stats:
            stats[s] = stats[s].cpu().detach()

        outputs["stats"] = stats

        return outputs
