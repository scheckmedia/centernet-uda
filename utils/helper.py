import sys
import hydra
import torch
from omegaconf.listconfig import ListConfig
import logging

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
