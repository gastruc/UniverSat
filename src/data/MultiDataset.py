from torch.utils.data import Dataset


class MultiDataset(Dataset):
    """Lightweight container bundling several per-dataset modules together.

    Attributes:
        datasets_modules (dict): mapping ``name -> dataset module``.
        datasets (list): ordered list of dataset names.
        collate_fn (dict): mapping ``name -> collate_fn`` of the underlying module.
        scales (dict): mapping ``name -> scale configuration``.
    """

    def __init__(self, datasets, scales):
        self.datasets_modules = datasets
        self.datasets = list(datasets.keys())
        self.scales = scales

    @property
    def collate_fn(self):
        # Lazy + always up-to-date with the underlying modules.
        return {name: self.datasets_modules[name].collate_fn for name in self.datasets}

    def __len__(self):
        return sum(len(self.datasets_modules[name]) for name in self.datasets)
