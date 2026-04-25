from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.config import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CUSTOM_IMAGE_SIZE,
    DATA_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    PRETRAINED_IMAGE_SIZE,
    SEED,
    VALIDATION_SIZE,
)


class TransformDataset(Dataset):
    def __init__(self, base_dataset, indices: list[int], transform=None) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform
        self.targets = [int(base_dataset.targets[index]) for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


@dataclass
class CIFAR10Context:
    train_base: datasets.CIFAR10
    test_base: datasets.CIFAR10
    train_indices: list[int]
    val_indices: list[int]
    class_names: list[str]
    loader_cache: dict[str, dict[str, DataLoader]]


def _custom_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(CUSTOM_IMAGE_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _custom_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _pretrained_train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((PRETRAINED_IMAGE_SIZE, PRETRAINED_IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _pretrained_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((PRETRAINED_IMAGE_SIZE, PRETRAINED_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def prepare_cifar10_context(
    validation_size: float = VALIDATION_SIZE,
    seed: int = SEED,
) -> CIFAR10Context:
    train_base = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    test_base = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

    train_indices, val_indices = train_test_split(
        np.arange(len(train_base)),
        test_size=validation_size,
        random_state=seed,
        stratify=train_base.targets,
    )

    return CIFAR10Context(
        train_base=train_base,
        test_base=test_base,
        train_indices=train_indices.tolist(),
        val_indices=val_indices.tolist(),
        class_names=list(train_base.classes),
        loader_cache={},
    )


def build_cifar10_loaders(
    context: CIFAR10Context,
    variant: str,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
) -> dict[str, DataLoader]:
    cache_key = f"{variant}:{batch_size}"
    if cache_key in context.loader_cache:
        return context.loader_cache[cache_key]

    if variant == "custom":
        train_transform = _custom_train_transform()
        eval_transform = _custom_eval_transform()
    elif variant == "pretrained":
        train_transform = _pretrained_train_transform()
        eval_transform = _pretrained_eval_transform()
    else:
        raise ValueError(f"Unsupported CIFAR-10 loader variant: {variant}")

    train_dataset = TransformDataset(context.train_base, context.train_indices, train_transform)
    train_eval_dataset = TransformDataset(
        context.train_base,
        context.train_indices,
        eval_transform,
    )
    val_dataset = TransformDataset(context.train_base, context.val_indices, eval_transform)
    test_indices = list(range(len(context.test_base)))
    test_dataset = TransformDataset(context.test_base, test_indices, eval_transform)

    generator = torch.Generator().manual_seed(SEED)
    pin_memory = torch.cuda.is_available()

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        ),
        "train_eval": DataLoader(
            train_eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    context.loader_cache[cache_key] = loaders
    return loaders
