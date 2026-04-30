from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ReliefStyleDataset(Dataset):
    """
    Image-level dataset for relief style learning.

    Preferred CSV columns:
        - Relief_ID
        - Authority
        - split
        - view_index
        - suffix

    Legacy/optional:
        - image_path

    Path resolution strategy:
        1. If image_root is provided, construct filename from:
           <Relief_ID><filename_sep><view_index><suffix>
        2. Otherwise, fall back to image_path column.

    Example:
        Relief_ID = "BM 124773"
        view_index = 2
        suffix = ".jpg"
        filename_sep = "_"

        -> "BM 124773_2.jpg"
    """

    REQUIRED_COLUMNS = {"Relief_ID", "Authority", "split"}

    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        class_to_idx: dict[str, int] | None = None,
        transform: Callable | None = None,
        image_root: str | Path | None = None,
        filename_sep: str = "-",
        check_paths: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.split = split
        self.transform = transform
        self.image_root = Path(image_root) if image_root is not None else None
        self.filename_sep = filename_sep

        self.df = pd.read_csv(self.csv_path).copy()

        missing_cols = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {self.csv_path}: {sorted(missing_cols)}"
            )

        self.df = self.df[self.df["split"] == split].copy().reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split='{split}' in {self.csv_path}")

        if class_to_idx is None:
            classes = sorted(self.df["Authority"].dropna().unique())
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        else:
            self.class_to_idx = dict(class_to_idx)

        unknown_classes = set(self.df["Authority"].unique()) - set(self.class_to_idx.keys())
        if unknown_classes:
            raise ValueError(
                "Found classes in dataset that are missing from class_to_idx: "
                f"{sorted(unknown_classes)}"
            )

        self.df["label"] = self.df["Authority"].map(self.class_to_idx)

        if self.df["label"].isna().any():
            bad_rows = self.df[self.df["label"].isna()][["Relief_ID", "Authority"]].head(10)
            raise ValueError(
                "Some rows could not be mapped to numeric labels. Examples:\n"
                f"{bad_rows.to_string(index=False)}"
            )

        self.df["label"] = self.df["label"].astype(int)

        if check_paths:
            missing_paths = []
            for idx in range(len(self.df)):
                path = self._resolve_image_path(self.df.iloc[idx])
                if not path.is_file():
                    missing_paths.append(
                        {
                            "Relief_ID": self.df.iloc[idx]["Relief_ID"],
                            "resolved_path": str(path),
                        }
                    )

            if missing_paths:
                examples = pd.DataFrame(missing_paths).head(10)
                raise FileNotFoundError(
                    "Some resolved image paths do not exist. Examples:\n"
                    f"{examples.to_string(index=False)}"
                )

    def _normalize_suffix(self, suffix: str) -> str:
        suffix = str(suffix).strip()
        if not suffix:
            raise ValueError("Empty suffix encountered.")
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return suffix

    def _build_filename(self, row: pd.Series) -> str:
        if "view_index" not in row.index:
            raise ValueError(
                "CSV must contain 'view_index' when image_root is used to reconstruct paths."
            )
        if "suffix" not in row.index:
            raise ValueError(
                "CSV must contain 'suffix' when image_root is used to reconstruct paths."
            )

        relief_id = str(row["Relief_ID"]).strip()
        view_index = int(row["view_index"])
        suffix = self._normalize_suffix(row["suffix"])

        return f"{relief_id}{self.filename_sep}{view_index}{suffix}"

    def _resolve_image_path(self, row: pd.Series) -> Path:
        if self.image_root is not None:
            filename = self._build_filename(row)
            return self.image_root / filename

        if "image_path" not in row.index:
            raise ValueError(
                "No image_root was provided, and CSV does not contain 'image_path'."
            )

        return Path(row["image_path"])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image_path = self._resolve_image_path(row)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        sample = {
            "image": image,
            "label": int(row["label"]),
            "authority": row["Authority"],
            "relief_id": row["Relief_ID"],
            "image_path": str(image_path),
        }

        if "view_index" in row.index:
            sample["view_index"] = int(row["view_index"])

        if "suffix" in row.index:
            sample["suffix"] = self._normalize_suffix(row["suffix"])

        return sample