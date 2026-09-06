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


def plot_query_image_retrieval_row(
    retrieval_row: pd.Series,
    image_root: str | Path,
    k: int = 5,
    figsize_scale: float = 4.5,
    show: bool = False,
):
    """
    Plot one query image and its top-k retrieved image neighbors.

    Expected columns:
        query_image_path or query_image_filename
        top1_image_path or top1_image_filename
        ...
    """

    image_root = Path(image_root)

    def resolve_path(prefix: str) -> Path:
        path_col = f"{prefix}_image_path"
        filename_col = f"{prefix}_image_filename"

        if path_col in retrieval_row.index and pd.notna(retrieval_row[path_col]):
            return Path(retrieval_row[path_col])

        if filename_col in retrieval_row.index and pd.notna(retrieval_row[filename_col]):
            return image_root / str(retrieval_row[filename_col])

        raise ValueError(f"Could not resolve image path for prefix={prefix!r}")

    prefixes = ["query"] + [f"top{rank}" for rank in range(1, k + 1)]

    fig, axes = plt.subplots(
        1,
        len(prefixes),
        figsize=(figsize_scale * len(prefixes), figsize_scale),
        constrained_layout=True,
    )

    if len(prefixes) == 1:
        axes = [axes]

    for ax, prefix in zip(axes, prefixes):
        img_path = resolve_path(prefix)

        if not img_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {img_path}")

        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        if prefix == "query":
            title = (
                f"QUERY\n"
                f"{retrieval_row.get('query_image_filename', img_path.name)}\n"
                f"{retrieval_row.get('query_authority', '')}"
            )
        else:
            rank = prefix.replace("top", "")
            title = (
                f"Top {rank}\n"
                f"{retrieval_row.get(f'{prefix}_image_filename', img_path.name)}\n"
                f"{retrieval_row.get(f'{prefix}_authority', '')}\n"
                f"sim={retrieval_row.get(f'{prefix}_similarity', float('nan')):.3f}"
            )

        ax.set_title(title, fontsize=10)

    if show:
        plt.show()

    return fig