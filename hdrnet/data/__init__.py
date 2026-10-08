"""Dataset builders."""

from torch.utils.data import DataLoader

from configs.datasets import DATASETS
from hdrnet.data.dataset import DisasterDataset


def make_dataset(name: str, root: str, train: bool, class_aware: bool = False, crop_size: int = 512) -> DisasterDataset:
    spec = DATASETS[name]
    split = spec["splits"]["train" if train else "test"]
    return DisasterDataset(
        root,
        split,
        train=train,
        crop_size=crop_size,
        class_aware=class_aware and train,
        rare_classes=spec["rare_classes"],
    )


def make_loader(
    name: str,
    root: str,
    train: bool,
    batch_size: int,
    workers: int,
    class_aware: bool = False,
    crop_size: int = 512,
) -> DataLoader:
    dataset = make_dataset(name, root, train, class_aware, crop_size)
    return DataLoader(
        dataset,
        batch_size=batch_size if train else 1,
        shuffle=train,
        num_workers=workers,
        drop_last=train,
        pin_memory=True,
    )
