import sys

import torch

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
