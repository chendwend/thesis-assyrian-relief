from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def _normalize_suffix(suffix: str) -> str:
    suffix = str(suffix).strip()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return suffix


def build_image_filename(
    relief_id: str,
    view_index: int | str,
    suffix: str,
    filename_sep: str = "-",
) -> str:
    suffix = _normalize_suffix(suffix)
    return f"{relief_id}{filename_sep}{view_index}{suffix}"


def build_image_path(
    image_root: str | Path,
    relief_id: str,
    view_index: int | str,
    suffix: str,
    filename_sep: str = "-",
) -> Path:
    image_root = Path(image_root)
    filename = build_image_filename(
        relief_id=relief_id,
        view_index=view_index,
        suffix=suffix,
        filename_sep=filename_sep,
    )
    return image_root / filename


def get_relief_rows(
    image_level_csv: str | Path,
    relief_id: str,
    deduplicate: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(image_level_csv)
    out = df[df["Relief_ID"] == relief_id].copy()

    if out.empty:
        raise ValueError(f"No rows found for Relief_ID={relief_id}")

    # Sort first
    sort_cols = [c for c in ["view_index", "suffix"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    # Deduplicate actual image identity
    if deduplicate:
        dedup_cols = [c for c in ["Relief_ID", "view_index", "suffix"] if c in out.columns]
        if dedup_cols:
            out = out.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    return out


def get_relief_image_paths(
    image_level_csv: str | Path,
    image_root: str | Path,
    relief_id: str,
    filename_sep: str = "-",
    check_paths: bool = True,
    deduplicate: bool = True,
) -> list[Path]:
    rows = get_relief_rows(
        image_level_csv=image_level_csv,
        relief_id=relief_id,
        deduplicate=deduplicate,
    )

    required_cols = {"Relief_ID", "view_index", "suffix"}
    missing = required_cols - set(rows.columns)
    if missing:
        raise ValueError(
            f"image_level_csv is missing required columns for path construction: {sorted(missing)}"
        )

    paths: list[Path] = []
    for _, row in rows.iterrows():
        path = build_image_path(
            image_root=image_root,
            relief_id=row["Relief_ID"],
            view_index=row["view_index"],
            suffix=row["suffix"],
            filename_sep=filename_sep,
        )
        if check_paths and not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        paths.append(path)

    return paths


def plot_relief_views(
    image_level_csv: str | Path,
    image_root: str | Path,
    relief_id: str,
    filename_sep: str = "-",
    title: str | None = None,
    max_cols: int = 3,
    figsize_scale: float = 6.0,
    check_paths: bool = True,
    deduplicate: bool = True,
    show: bool = False,
):
    paths = get_relief_image_paths(
        image_level_csv=image_level_csv,
        image_root=image_root,
        relief_id=relief_id,
        filename_sep=filename_sep,
        check_paths=check_paths,
        deduplicate=deduplicate,
    )

    n = len(paths)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_scale * ncols, figsize_scale * nrows),
        constrained_layout=True,
    )

    if hasattr(axes, "ravel"):
        axes = axes.ravel().tolist()
    else:
        axes = [axes]

    for ax, path in zip(axes, paths):
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(path.name, fontsize=11)
        ax.axis("off")

    for ax in axes[len(paths):]:
        ax.axis("off")

    fig.suptitle(title or f"Relief {relief_id}", fontsize=14)

    if show:
        plt.show()

    return fig


def plot_query_and_retrieved_reliefs(
    image_level_csv: str | Path,
    image_root: str | Path,
    query_relief_id: str,
    retrieved_relief_ids: list[str],
    filename_sep: str = "-",
    figsize_scale: float = 6.0,
    check_paths: bool = True,
    deduplicate: bool = True,
    show: bool = False,
):
    """
    Show the first deduplicated view of the query relief and the first deduplicated
    view of each retrieved relief.
    """
    all_relief_ids = [query_relief_id] + retrieved_relief_ids

    fig, axes = plt.subplots(
        1,
        len(all_relief_ids),
        figsize=(figsize_scale * len(all_relief_ids), figsize_scale),
        constrained_layout=True,
    )
    if len(all_relief_ids) == 1:
        axes = [axes]

    for ax, relief_id in zip(axes, all_relief_ids):
        rows = get_relief_rows(
            image_level_csv=image_level_csv,
            relief_id=relief_id,
            deduplicate=deduplicate,
        )
        first_row = rows.iloc[0]

        img_path = build_image_path(
            image_root=image_root,
            relief_id=first_row["Relief_ID"],
            view_index=first_row["view_index"],
            suffix=first_row["suffix"],
            filename_sep=filename_sep,
        )

        if check_paths and not img_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {img_path}")

        authority = first_row["Authority"]
        img = Image.open(img_path).convert("RGB")

        ax.imshow(img)
        ax.set_title(f"{relief_id}\n{authority}", fontsize=11)
        ax.axis("off")

    if show:
        plt.show()

    return fig