import logging
import numpy as np
import hydra
from torch.utils import data
log = logging.getLogger(__name__)


class Dataset(data.Dataset):
    def __init__(self, datasets, max_samples=None, **defaults):
        self.max_sampels = max_samples
        self.datasets = {}
        self.num_samples = 0

        for ds in datasets:
            coco = hydra.utils.get_class(
                f'datasets.{ds.name}.Dataset')
            params = {**defaults, **ds.params}
            coco = coco(**params)
            self.num_samples += len(coco)
            self.datasets[self.num_samples] = coco

        self.intervals = np.array(list(self.datasets.keys()))

        log.info(
            f"merged {len(self.datasets)} datasets with a total "
            f"number of {self.num_samples} samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        interval_idx = np.argmax(index < self.intervals)
        interval = self.intervals[interval_idx]
        offset = self.intervals[interval_idx - 1] if interval_idx > 0 else 0
        return self.datasets[interval].__getitem__(index - offset)
