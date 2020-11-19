# Here we will perform all the training and implementation of Advent

from uda.base import Model
from losses.advent import AdventLoss
from torch import nn
from torch.optim import Adam
from utils.image import entropy_map
from utils.helper import CustomDataParallel, load_model, save_model
from pathlib import Path
import hydra


class AdversarialEntropyMinimization(Model):
    def __init__(self, adversarial_weight, optimizer=None):
        super().__init__()
        self.adversarial_loss = AdventLoss()
        self.adversarial_weight = adversarial_weight

        self.source_label = 0
        self.target_label = 1

        self.optimizer_settings = optimizer
        self.discriminator = None

    def init_done(self):
        self.discriminator = self.get_fc_discriminator(
            num_classes=self.cfg.model.backend.params.num_classes)
        self.discriminator_scheduler = None

        if self.optimizer_settings is None:
            self.discriminator_optimizer = Adam(
                self.discriminator.parameters())
        else:
            optimizer = hydra.utils.get_class(
                f"torch.optim.{self.optimizer_settings.name}")
            self.discriminator_optimizer = optimizer(
                self.discriminator.parameters(),
                **self.optimizer_settings.params)

            self.discriminator_scheduler = None
            if 'scheduler' in self.optimizer_settings and \
                    self.optimizer_settings.scheduler is not None:
                scheduler = hydra.utils.get_class(
                    f"torch.optim.lr_scheduler.{self.optimizer_settings.scheduler.name}")
                self.discriminator_scheduler = scheduler(
                    **
                    {**{"optimizer": self.discriminator_optimizer},
                     **self.optimizer_settings.scheduler.params})

    # https://github.com/valeoai/ADVENT/blob/516ee50e2ebad65959f89ccf9edebee649e38f3a/advent/model/discriminator.py
    def get_fc_discriminator(self, num_classes, ndf=64):
        model = nn.Sequential(
            # N*num_classes(or channels_img)*64*64
            nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # N*ndf(or features)*32*32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # N*ndf*8*4*4
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
            # N*1*1*1
        )

        return model

    def set_phase(self, is_training=True):
        super().set_phase(is_training)
        if is_training:
            self.discriminator.train()
        else:
            self.discriminator.eval()

    def step(self, data, is_training=True):
        for k in data:
            data[k] = data[k].to(device=self.device, non_blocking=True)

        if is_training:
            self.optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

        for param in self.discriminator.parameters():
            param.requires_grad = False

        outputs_source_domain = self.backend(data["input"])
        outputs_target_domain = self.backend(data["target_domain_input"])

        outputs_target_generator = self.discriminator(
            entropy_map(outputs_target_domain["hm"]))

        outputs = {
            "source_domain": outputs_source_domain,
            "target_domain": outputs_target_domain,
        }

        loss, stats = self.centernet_loss(outputs_source_domain, data)
        if is_training:
            loss.backward()

        # fool generator
        dtf_loss, dtf_stats = self.adversarial_loss(
            outputs_target_generator, self.source_label
        )
        dtf_loss *= self.adversarial_weight

        if is_training:
            dtf_loss.backward()

        # train discriminator
        for param in self.discriminator.parameters():
            param.requires_grad = True

        source = outputs_source_domain["hm"].detach()
        target = outputs_target_domain["hm"].detach()

        outputs_source_generator = self.discriminator(entropy_map(source))
        ds_loss, ds_stats = self.adversarial_loss(
            outputs_source_generator, self.source_label)
        ds_loss /= 2.0

        if is_training:
            ds_loss.backward()

        outputs_target_generator = self.discriminator(entropy_map(target))
        dt_loss, dt_stats = self.adversarial_loss(
            outputs_target_generator, self.target_label)
        dt_loss /= 2.0

        if is_training:
            dt_loss.backward()

        outputs['source_generator'] = outputs_source_generator
        outputs['target_generator'] = outputs_target_domain

        if is_training:
            self.optimizer.step()
            self.discriminator_optimizer.step()

        # compose final loss
        stats["total_loss"] = loss + ds_loss + dt_loss + dtf_loss
        stats["dis_soruce"] = ds_loss
        stats["dis_target"] = dt_loss
        stats["dis_fool"] = dtf_loss

        for s in stats:
            stats[s] = stats[s].cpu().detach()

        outputs["stats"] = stats
        return outputs

    def to(self, device, parallel=False):
        super().to(device, parallel)

        if parallel:
            self.discriminator = CustomDataParallel(self.discriminator)

        self.discriminator.to(device)

    def epoch_end(self):
        super().epoch_end()

        if self.discriminator_scheduler is not None:
            self.discriminator_scheduler.step()

    # todo load and save generator
    def save_model(self, path, epoch, with_optimizer=False):
        super().save_model(path, epoch, with_optimizer)
        if with_optimizer:
            save_model(
                self.discriminator,
                'discriminator.pth',
                epoch,
                self.discriminator_optimizer,
                self.discriminator_scheduler)
        else:
            save_model(self.discriminator, path, epoch)

    def load_model(self, path, resume=False):
        d_weights = str(Path(path).with_name('discriminator.pth'))
        load_model(
            self.discriminator,
            self.discriminator_optimizer,
            self.discriminator_scheduler,
            d_weights,
            resume)
        return super().load_model(path, resume=resume)
