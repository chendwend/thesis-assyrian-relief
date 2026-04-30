from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

from thesis_assyrian_relief.datasets.relief_style_dataset import ReliefStyleDataset


def _make_dummy_image(path: Path, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(128, 128, 128))
    img.save(path)


@pytest.fixture
def class_to_idx() -> dict[str, int]:
    return {
        "Ashurbanipal": 0,
        "Ashurnasirpal II": 1,
        "OTHER": 2,
        "Sargon II": 3,
    }


@pytest.fixture
def transform():
    return transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])


def test_dataset_filters_split_and_maps_labels(tmp_path: Path, class_to_idx: dict[str, int], transform) -> None:
    image_root = tmp_path / "images"
    csv_path = tmp_path / "image_level_dataset.csv"

    rows = [
        {"Relief_ID": "BM 1", "Authority": "Ashurbanipal", "split": "train", "view_index": 1, "suffix": ".jpg"},
        {"Relief_ID": "BM 2", "Authority": "OTHER", "split": "val", "view_index": 1, "suffix": ".jpg"},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    _make_dummy_image(image_root / "BM 1_1.jpg")
    _make_dummy_image(image_root / "BM 2_1.jpg")

    ds = ReliefStyleDataset(
        csv_path=csv_path,
        split="train",
        class_to_idx=class_to_idx,
        transform=transform,
        image_root=image_root,
        filename_sep="_",
        check_paths=True,
    )

    assert len(ds) == 1

    sample = ds[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 16, 16)
    assert sample["authority"] == "Ashurbanipal"
    assert sample["label"] == 0
    assert sample["relief_id"] == "BM 1"


def test_dataset_resolves_filename_from_columns(tmp_path: Path, class_to_idx: dict[str, int], transform) -> None:
    image_root = tmp_path / "images"
    csv_path = tmp_path / "image_level_dataset.csv"

    rows = [
        {"Relief_ID": "AO 19854", "Authority": "OTHER", "split": "train", "view_index": 3, "suffix": "png"},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    expected_path = image_root / "AO 19854_3.png"
    _make_dummy_image(expected_path)

    ds = ReliefStyleDataset(
        csv_path=csv_path,
        split="train",
        class_to_idx=class_to_idx,
        transform=transform,
        image_root=image_root,
        filename_sep="_",
        check_paths=True,
    )

    sample = ds[0]
    assert Path(sample["image_path"]) == expected_path
    assert sample["view_index"] == 3
    assert sample["suffix"] == ".png"


def test_dataset_supports_custom_separator(tmp_path: Path, class_to_idx: dict[str, int], transform) -> None:
    image_root = tmp_path / "images"
    csv_path = tmp_path / "image_level_dataset.csv"

    rows = [
        {"Relief_ID": "MET 32.143.8", "Authority": "Ashurnasirpal II", "split": "train", "view_index": 2, "suffix": ".jpg"},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    expected_path = image_root / "MET 32.143.8-2.jpg"
    _make_dummy_image(expected_path)

    ds = ReliefStyleDataset(
        csv_path=csv_path,
        split="train",
        class_to_idx=class_to_idx,
        transform=transform,
        image_root=image_root,
        filename_sep="-",
        check_paths=True,
    )

    sample = ds[0]
    assert Path(sample["image_path"]) == expected_path
    assert sample["label"] == class_to_idx["Ashurnasirpal II"]


def test_dataset_raises_when_path_missing(tmp_path: Path, class_to_idx: dict[str, int], transform) -> None:
    image_root = tmp_path / "images"
    csv_path = tmp_path / "image_level_dataset.csv"

    rows = [
        {"Relief_ID": "BM 999", "Authority": "Sargon II", "split": "train", "view_index": 1, "suffix": ".jpg"},
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with pytest.raises(FileNotFoundError):
        ReliefStyleDataset(
            csv_path=csv_path,
            split="train",
            class_to_idx=class_to_idx,
            transform=transform,
            image_root=image_root,
            filename_sep="-",
            check_paths=True,
        )