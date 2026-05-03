from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from thesis_assyrian_relief.datasets.relief_style_dataset import ReliefStyleDataset


def build_class_to_idx(csv_path: str | Path) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    if "Authority" not in df.columns:
        raise ValueError(f"'Authority' column not found in {csv_path}")

    classes = sorted(df["Authority"].dropna().unique())
    return {cls_name: idx for idx, cls_name in enumerate(classes)}


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_train_transform():
    # Keep conservative for now. We can expand augmentations later.
    return transforms.Compose([
        transforms.Resize((224, 224)),
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
    train: bool = False,
    check_paths: bool = True,
) -> ReliefStyleDataset:
    transform = build_train_transform() if train else build_eval_transform()

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