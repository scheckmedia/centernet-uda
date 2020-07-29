import hydra
from omegaconf import DictConfig
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True


def load_datasets(cfg, down_ratio):
    defaults = {"max_detections": cfg.max_detections,
                "down_ratio": down_ratio,
                "num_classes": cfg.model.params.num_classes}

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
    uda.centernet_loss = hydra.utils.instantiate(cfg.centernet_loss)

    train_loader, val_loader = load_datasets(cfg, down_ratio=model.down_ratio)

    start = 0

    for epoch in tqdm(range(start, cfg.epochs), position=0, desc='Epoch'):

        running_loss = 0.0
        for step, data in tqdm(enumerate(train_loader), position=1, desc='Training Steps'):
            for k in data:
                data[k] = data[k].to(device=device, non_blocking=True)

            uda.step(data)

        with torch.no_grad():
            for step, data in tqdm(enumerate(val_loader), position=1, desc='Validation Steps'):
                for k in data:
                    data[k] = data[k].to(device=device, non_blocking=True)

                uda.step(data, is_training=False)


if __name__ == "__main__":
    main()
