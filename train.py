import logging
import os

import hydra
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.helper import AverageMeter
from utils.tensorboard import TensorboardLogger

log = logging.getLogger("uda")
torch.backends.cudnn.benchmark = True


def load_datasets(cfg, down_ratio, rotated_boxes):
    defaults = {"max_detections": cfg.max_detections,
                "down_ratio": down_ratio,
                "rotated_boxes": rotated_boxes,
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

    test_loader = None
    if 'test' in cfg.datasets:
        test = hydra.utils.get_class(
            f'datasets.{cfg.datasets.test.name}.Dataset')
        params = {**cfg.datasets.test.params, **defaults}
        test = test(**params)
        test_loader = DataLoader(
            test,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True)

        log.info(f"Found {len(test)} samples in test dataset")

    return training_loader, validation_loader, test_loader


@hydra.main(config_path="configs/defaults.yaml")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)

    is_multi_gpu = False

    if isinstance(cfg.gpu, ListConfig):
        is_multi_gpu = True
        log.info(f"Use GPUs {str(cfg.gpu).strip('[]')} for training")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)

    device = torch.device(f'cuda' if cfg.gpu is not None else 'cpu')
    log.info(f"Use device {device} for training")

    backend = hydra.utils.get_method(
        f'backends.{cfg.model.backend.name}.build')(**cfg.model.backend.params)

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

    train_loader, val_loader, test_loader = load_datasets(
        cfg, down_ratio=backend.down_ratio, rotated_boxes=backend.rotated_boxes)
    tensorboard_logger = TensorboardLogger(cfg, val_loader.dataset.classes)

    evaluators = []
    for e in cfg.evaluation:
        defaults = {
            'score_threshold': cfg.score_threshold,
            **cfg.evaluation[e]}
        e = hydra.utils.get_class(
            f"evaluation.{e}.Evaluator")(**defaults)
        e.classes = tensorboard_logger.classes
        e.num_workers = cfg.num_workers
        e.use_rotated_boxes = cfg.model.backend.params.rotated_boxes
        evaluators.append(e)

    uda.init_done()

    start_epoch = 1
    if cfg.pretrained is not None and cfg.resume is None:
        start_epoch = uda.load_model(cfg.pretrained)
    elif cfg.resume is not None:
        start_epoch = uda.load_model(cfg.resume, True)

    uda.to(device, is_multi_gpu)

    stats = {}
    best = float("inf") if cfg.save_best_metric.mode == 'min' else -float("inf")

    if not cfg.test_only:
        for epoch in tqdm(
                range(start_epoch, cfg.epochs + 1),
                initial=start_epoch,
                position=0, desc='Epoch'):
            uda.epoch_start()
            uda.set_phase(is_training=True)
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

            if epoch % cfg.eval_at_n_epoch != 0:
                continue

            tag = 'validation'
            uda.set_phase(is_training=False)
            with torch.no_grad():
                for step, data in tqdm(
                        enumerate(val_loader),
                        total=len(val_loader),
                        position=1, desc='Validation Steps'):
                    outputs = uda.step(data, is_training=False)

                    for k in outputs["stats"]:
                        log_key = f"{tag}/{k}"
                        m = stats.get(log_key, AverageMeter(name=k))
                        m.update(
                            outputs["stats"][k].item(),
                            data["input"].size(0))
                        stats[log_key] = m

                    detections = uda.get_detections(outputs, data)
                    detections["image_shape"] = data["input"].shape[1:]
                    for e in evaluators:
                        e.add_batch(**detections)

                    tensorboard_logger.log_detections(
                        data, detections, epoch, tag=tag)

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
            if (cfg.save_best_metric.mode == 'min' and best >
                    current or cfg.save_best_metric.mode == 'max' and best < current):
                uda.save_model("model_best.pth", epoch, True)
                best = current

                log.info(
                    f"Save best model with {cfg.save_best_metric.name} of {current:.4f}")

    if test_loader is not None:
        if cfg.test_only:
            epoch = start_epoch
        tag = 'test'

        uda.set_phase(is_training=False)
        with torch.no_grad():
            for step, data in tqdm(
                    enumerate(test_loader),
                    total=len(test_loader),
                    position=1, desc='Test Steps'):
                outputs = uda.step(data, is_training=False)

                for k in outputs["stats"]:
                    log_key = f"{tag}/{k}"
                    m = stats.get(log_key, AverageMeter(name=k))
                    m.update(
                        outputs["stats"][k].item(),
                        data["input"].size(0))
                    stats[log_key] = m

                detections = uda.get_detections(outputs, data)
                detections["image_shape"] = data["input"].shape[1:]
                for e in evaluators:
                    e.add_batch(**detections)

                tensorboard_logger.log_detections(
                    data, detections, epoch, tag=tag)

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

        tensorboard_logger.reset()


if __name__ == "__main__":
    main()
