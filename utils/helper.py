import sys
import hydra
import torch
from omegaconf.listconfig import ListConfig
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# https://github.com/pytorch/examples/blob/8df8e747857261ea481e0b2492413d52bf7cc3a8/imagenet/main.py#L363


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class RedirectOut():
    def __init__(self, out):
        super().__init__()
        self.out = out
        self.original = sys.stdout

    def __enter__(self):
        self.__fd = open(self.out, 'w')
        sys.stdout = self.__fd

    def __exit__(self, type, value, traceback):
        sys.stdout = self.original
        self.__fd.close()


def instantiate_augmenters(augmentation_list):
    augmentation_methods = []
    for augmentation in augmentation_list:
        method = list(augmentation)[0]
        params = dict(augmentation[method])

        if method == 'Sometimes':
            params["then_list"] = instantiate_augmenters(params["then_list"])

        for k, v in params.items():
            if isinstance(v, (list, ListConfig)):
                params[k] = tuple(v)
        m = hydra.utils.get_method(
            f"imgaug.augmenters.{method}")(**params)
        augmentation_methods.append(m)

        log.debug(
            f"Register imgaug.augmenters.{method} as augmentation method")
    return augmentation_methods


# https://discuss.pytorch.org/t/access-att-of-model-wrapped-within-torch-nn-dataparallel-maximum-recursion-depth-exceeded/46975/2
class CustomDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def load_model(model, optimizer, scheduler, path, resume=False):
    path = Path(path)
    if not path.exists():
        log.warning(f"Model path {path} does not exists!")
        return 1

    checkpoint = torch.load(path)
    epoch = checkpoint["epoch"] if resume else 0
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]

    model_state_dict = model.state_dict()

    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                log.warning(
                    f"skip parameter {k} because of shape mismatch")
                state_dict[k] = model_state_dict[k]
        else:
            log.info(f"drop parameter {k}")

    for k in model_state_dict:
        if k not in state_dict:
            log.warning(f"no parameter {k} available")
            state_dict[k] = model_state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    log.info(f"restore pretrained weights")

    if resume and 'optimizer' in checkpoint and optimizer is not None:
        log.info(f"restore optimizer state at epoch {epoch}")
        optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scheduler' in checkpoint and scheduler is not None:
            log.info("restore scheduler state")
            scheduler.load_state_dict(checkpoint['scheduler'])

    return (epoch + 1) if resume else epoch


def save_model(model, path, epoch, optimizer=None, scheduler=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    data = {
        'epoch': epoch,
        'state_dict': state_dict
    }
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()

        if scheduler is not None:
            data["scheduler"] = scheduler.state_dict()

    torch.save(data, path)
