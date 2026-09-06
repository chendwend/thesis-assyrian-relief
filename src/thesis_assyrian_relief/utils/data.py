from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as transform_functional

from thesis_assyrian_relief.datasets.relief_style_dataset import ReliefStyleDataset


def build_class_to_idx(csv_path: str | Path) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    if "Authority" not in df.columns:
        raise ValueError(f"'Authority' column not found in {csv_path}")

    classes = sorted(df["Authority"].dropna().unique())
    return {cls_name: idx for idx, cls_name in enumerate(classes)}


class ResizeAndPad:
    """Resize the long edge to a square canvas while preserving aspect ratio."""

    def __init__(
        self,
        size: int = 224,
        fill: tuple[int, int, int] = (124, 116, 104),
    ) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        scale = self.size / max(width, height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = transform_functional.resize(
            image,
            [resized_height, resized_width],
            antialias=True,
        )
        horizontal = self.size - resized_width
        vertical = self.size - resized_height
        padding = [
            horizontal // 2,
            vertical // 2,
            horizontal - horizontal // 2,
            vertical - vertical // 2,
        ]
        return transform_functional.pad(resized, padding, fill=self.fill)


def build_resize_transform(resize_mode: str):
    if resize_mode == "stretch":
        return transforms.Resize((224, 224))
    if resize_mode == "pad":
        return ResizeAndPad(size=224)
    raise ValueError(
        f"Unknown resize mode {resize_mode!r}; expected 'stretch' or 'pad'"
    )


def build_eval_transform(resize_mode: str = "stretch"):
    return transforms.Compose([
        build_resize_transform(resize_mode),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_train_transform(resize_mode: str = "stretch"):
    return transforms.Compose([
        build_resize_transform(resize_mode),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_dataset(
    csv_path: str,
    split: str,
    class_to_idx: dict[str, int],
    image_root: str,
    filename_sep: str = "-",
    resize_mode: str = "stretch",
    train: bool = False,
    check_paths: bool = True,
) -> ReliefStyleDataset:
    transform = (
        build_train_transform(resize_mode)
        if train
        else build_eval_transform(resize_mode)
    )

    return ReliefStyleDataset(
        csv_path=csv_path,
        split=split,
        class_to_idx=class_to_idx,
        transform=transform,
        image_root=image_root,
        filename_sep=filename_sep,
        check_paths=check_paths,
    )


def build_dataloader(
    csv_path: str,
    split: str,
    class_to_idx: dict[str, int],
    image_root: str,
    filename_sep: str = "-",
    resize_mode: str = "stretch",
    batch_size: int = 16,
    num_workers: int = 2,
    shuffle: bool = False,
    train: bool = False,
    check_paths: bool = True,
) -> tuple[ReliefStyleDataset, DataLoader]:
    ds = build_dataset(
        csv_path=csv_path,
        split=split,
        class_to_idx=class_to_idx,
        image_root=image_root,
        filename_sep=filename_sep,
        resize_mode=resize_mode,
        train=train,
        check_paths=check_paths,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return ds, loader


def build_class_weights(
    dataset: ReliefStyleDataset,
    class_to_idx: dict[str, int],
) -> torch.Tensor:
    """Build the class weights according to the class distribution in the dataset.

    The class weights are calculated as the inverse of the class frequency.

    The formula for the class weights is:
    weights = sum(counts_by_idx) / (len(counts_by_idx) * counts_by_idx)

    Args:
        dataset: The dataset to build the class weights for.
        class_to_idx: A dictionary mapping class names to labels.

    Returns:
        A tensor of class weights.
    """

    counts = dataset.df["Authority"].value_counts()

    counts_by_idx = np.zeros(len(class_to_idx), dtype=np.float32)
    for cls_name, idx in class_to_idx.items():
        counts_by_idx[idx] = counts[cls_name]

    weights = counts_by_idx.sum() / (len(counts_by_idx) * counts_by_idx)
    return torch.tensor(weights, dtype=torch.float32)
