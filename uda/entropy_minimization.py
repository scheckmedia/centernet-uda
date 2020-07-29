from uda.base import UDA
from losses.entropy import EntropyLoss


class EntropyMinimization(UDA):
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
