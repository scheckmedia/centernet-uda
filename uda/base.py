import torch


class UDA():
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.centernet_loss = None
        self.device = None
        super().__init__()

    def step(self, data, is_training):
        if is_training:
            self.optimizer.zero_grad()

        outputs_source_domain = self.model(data["input"])
        outputs_target_domain = self.model(data["target_domain_input"])

        outputs = {
            "source_domain": outputs_source_domain,
            "target_domain": outputs_target_domain
        }
        loss, stats = self.criterion(outputs, data)

        if is_training:
            loss.backward()
            self.optimizer.step()

        outputs["loss"] = loss
        outputs["stats"] = stats

        return outputs

    def criterion(self, outputs, batch):
        raise NotImplementedError
