import logging

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.helper import AverageMeter
from utils.visualize import Visualizer
from tensorboardX import SummaryWriter

log = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True


def load_datasets(cfg, down_ratio):
    defaults = {"max_detections": cfg.max_detections,
                "down_ratio": down_ratio,
                "num_classes": cfg.model.params.num_classes,
                "mean": cfg.normalize.mean,
                "std": cfg.normalize.std}

    validation = hydra.utils.get_class(
        f'{cfg.datasets.validation.name}.Dataset')
    params = {**cfg.datasets.validation.params, **defaults}
    validation = validation(**params)
    validation_loader = DataLoader(validation, batch_size=cfg.batch_size,
                                   shuffle=False, num_workers=cfg.num_workers)

    log.info(f"Found {len(validation)} samples in validation dataset")

    training = hydra.utils.get_class(f'{cfg.datasets.training.name}.Dataset')
    params = {**cfg.datasets.training.params, **defaults}
    training = training(**params)
    training_loader = DataLoader(training, batch_size=cfg.batch_size,
                                 shuffle=True, num_workers=cfg.num_workers)

    log.info(f"Found {len(training)} samples in training dataset")

    return training_loader, validation_loader


def load_model(cfg, model):
    pass


@hydra.main(config_path="configs/defaults.yaml")
def main(cfg: DictConfig) -> None:
    writer = SummaryWriter('logs')
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda' if len(cfg.gpus) > 0 else 'cpu')

    model = hydra.utils.get_method(
        f'{cfg.model.name}.build')(**cfg.model.params)
    model.to(device)

    optimizer = hydra.utils.get_class(f"torch.optim.{cfg.optimizer.name}")
    optimizer = optimizer(filter(lambda p: p.requires_grad,
                                 model.parameters()), **cfg.optimizer.params)

    uda_method = list(cfg.uda.keys())[0]
    uda_params = cfg.uda[uda_method]
    uda_cls = hydra.utils.get_class(f"uda.{uda_method}")
    uda = uda_cls(**uda_params) if uda_params else uda_cls()
    uda.device = device
    uda.model = model
    uda.optimizer = optimizer
    uda.summary_writer = writer
    uda.centernet_loss = hydra.utils.instantiate(cfg.centernet_loss)
    uda.visualizer = Visualizer(0.3, cfg.normalize.mean, cfg.normalize.std)

    evaluators = []
    for e in cfg.evaluation:
        e = hydra.utils.get_class(
            f"evaluation.{e}.Evaluator")(**cfg.evaluation[e])
        evaluators.append(e)

    train_loader, val_loader = load_datasets(cfg, down_ratio=model.down_ratio)
    uda.classes = val_loader.dataset.classes

    start = 0

    stats = {}
    for epoch in tqdm(range(start, cfg.epochs), position=0, desc='Epoch'):
        running_loss = 0.0
        for step, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                position=1, desc='Training Steps'):
            outputs = uda.step(data)

            for k in outputs["stats"]:
                log_key = f"train/{k}"
                m = stats.get(log_key, AverageMeter(name=k))
                m.update(outputs["stats"][k].item(), data["input"].size(0))
                stats[log_key] = m

            break

        with torch.no_grad():
            for step, data in tqdm(
                    enumerate(val_loader),
                    total=len(val_loader),
                    position=1, desc='Validation Steps'):
                outputs = uda.step(data, is_training=False)

                for k in outputs["stats"]:
                    log_key = f"val/{k}"
                    m = stats.get(log_key, AverageMeter(name=k))
                    m.update(outputs["stats"][k].item(), data["input"].size(0))
                    stats[log_key] = m

                detections = uda.get_detections(outputs, data)
                for e in evaluators:
                    e.add_batch(**detections)

                uda.log_detections(data, detections, epoch)
                break

        uda.log_stats(stats, epoch)

        for e in evaluators:
            result = e.evaluate()


if __name__ == "__main__":
    main()
