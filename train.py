import logging
import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.helper import AverageMeter
from utils.tensorboard import TensorboardLogger

log = logging.getLogger("uda")
torch.backends.cudnn.benchmark = True


def load_datasets(cfg, down_ratio):
    defaults = {"max_detections": cfg.max_detections,
                "down_ratio": down_ratio,
                "num_classes": cfg.model.backend.params.num_classes,
                "mean": cfg.normalize.mean,
                "std": cfg.normalize.std}

    validation = hydra.utils.get_class(
        f'datasets.{cfg.datasets.validation.name}.Dataset')
    params = {**cfg.datasets.validation.params, **defaults}
    validation = validation(**params)
    validation_loader = DataLoader(
        validation,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True)

    log.info(f"Found {len(validation)} samples in validation dataset")

    training = hydra.utils.get_class(
        f'datasets.{cfg.datasets.training.name}.Dataset')
    params = {**cfg.datasets.training.params, **defaults}
    training = training(**params)
    training_loader = DataLoader(
        training,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True)

    log.info(f"Found {len(training)} samples in training dataset")

    return training_loader, validation_loader


@hydra.main(config_path="configs/defaults.yaml")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    device = torch.device(f'cuda' if cfg.gpu is not None else 'cpu')

    backend = hydra.utils.get_method(
        f'backends.{cfg.model.backend.name}.build')(**cfg.model.backend.params)
    backend.to(device)

    optimizer = hydra.utils.get_class(f"torch.optim.{cfg.optimizer.name}")
    optimizer = optimizer(filter(lambda p: p.requires_grad,
                                 backend.parameters()), **cfg.optimizer.params)

    scheduler = None
    if cfg.optimizer.scheduler is not None:
        scheduler = hydra.utils.get_class(
            f"torch.optim.lr_scheduler.{cfg.optimizer.scheduler.name}")
        scheduler = scheduler(
            **
            {**{"optimizer": optimizer},
             **cfg.optimizer.scheduler.params})

    if cfg.model.uda is not None:
        uda_method = list(cfg.model.uda.keys())[0]
        uda_params = cfg.model.uda[uda_method]
        uda_cls = hydra.utils.get_class(f"uda.{uda_method}")
        uda = uda_cls(**uda_params) if uda_params else uda_cls()
    else:
        uda = hydra.utils.get_class(f"uda.base.Model")()
    uda.cfg = cfg
    uda.device = device
    uda.backend = backend
    uda.optimizer = optimizer
    uda.centernet_loss = hydra.utils.get_class(
        f"losses.{cfg.model.backend.loss.name}")(
        **cfg.model.backend.loss.params)

    uda.scheduler = scheduler

    train_loader, val_loader = load_datasets(
        cfg, down_ratio=backend.down_ratio)
    tensorboard_logger = TensorboardLogger(cfg, val_loader.dataset.classes)

    evaluators = []
    for e in cfg.evaluation:
        e = hydra.utils.get_class(
            f"evaluation.{e}.Evaluator")(**cfg.evaluation[e])
        e.classes = tensorboard_logger.classes
        evaluators.append(e)

    start_epoch = 1
    if cfg.pretrained is not None and cfg.resume is None:
        start_epoch = uda.load_model(cfg.pretrained)
    elif cfg.resume is not None:
        start_epoch = uda.load_model(cfg.resume, True)

    stats = {}
    best = 1e10 if cfg.save_best_metric.mode == 'min' else 1e-10

    for epoch in tqdm(
            range(start_epoch, cfg.epochs + 1),
            initial=start_epoch,
            position=0, desc='Epoch'):
        uda.epoch_start()
        uda.backend.train()
        tag = 'training'
        for step, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                position=1, desc='Training Steps'):
            outputs = uda.step(data)

            for k in outputs["stats"]:
                log_key = f"{tag}/{k}"
                m = stats.get(log_key, AverageMeter(name=k))
                m.update(outputs["stats"][k].item(), data["input"].size(0))
                stats[log_key] = m

        tag = 'validation'
        uda.backend.eval()
        with torch.no_grad():
            for step, data in tqdm(
                    enumerate(val_loader),
                    total=len(val_loader),
                    position=1, desc='Validation Steps'):
                outputs = uda.step(data, is_training=False)

                for k in outputs["stats"]:
                    log_key = f"{tag}/{k}"
                    m = stats.get(log_key, AverageMeter(name=k))
                    m.update(outputs["stats"][k].item(), data["input"].size(0))
                    stats[log_key] = m

                detections = uda.get_detections(outputs, data)
                for e in evaluators:
                    e.add_batch(**detections)

                tensorboard_logger.log_detections(
                    data, detections, epoch, tag='validation')

        for e in evaluators:
            result = e.evaluate()
            stats = {**stats, **result}

        scalars = {}
        for k, s in stats.items():
            if isinstance(s, AverageMeter):
                scalars[k] = s.avg
                s.reset()
            else:
                scalars[k] = s

            tensorboard_logger.log_stat(k, scalars[k], epoch)

        uda.epoch_end()
        tensorboard_logger.reset()
        uda.save_model("model_last.pth", epoch, True)

        if not cfg.save_best_metric.name in scalars:
            log.error(
                f"Metric {cfg.save_best_metric.name} not valid, valid values are {' '.join(scalars.keys())}")
            return

        current = scalars[cfg.save_best_metric.name]
        if (cfg.save_best_metric.mode == 'min' and best > current
                or cfg.save_best_metric.mode == 'max' and best < current):
            uda.save_model("model_best.pth", epoch, True)
            best = current

            log.info(
                f"Save best model with {cfg.save_best_metric.name} of {current:.4f}")


if __name__ == "__main__":
    main()
