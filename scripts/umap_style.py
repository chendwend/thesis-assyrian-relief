from __future__ import annotations

import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd
import torch
from thesis_assyrian_relief.utils.data import build_dataloader

from thesis_assyrian_relief.utils.config import load_yaml_config, ensure_parent_dir
from thesis_assyrian_relief.evaluation.retrieval import (
    aggregate_relief_embeddings,
    extract_embeddings,
)
from thesis_assyrian_relief.evaluation.dependency_group import (
    aggregate_image_embeddings_by_dependency_group,
)
from thesis_assyrian_relief.models.dinov2_probe import DinoStyleProbe
from thesis_assyrian_relief.training.engine import load_checkpoint
from thesis_assyrian_relief.utils.plotting import (
    build_umap_dataframe,
    plot_umap_plotly,
)

import warnings
from numba.core.errors import NumbaWarning
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings(
    "ignore",
    message="n_jobs value .* overridden to 1 by setting random_state.*",
)

warnings.filterwarnings("ignore", category=NumbaWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UMAP visualization for relief style embeddings.")

    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")

    parser.add_argument("--csv-path", type=str, default=None, help="Path to image_level_dataset.csv")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory containing image files")
    parser.add_argument("--filename-sep", type=str, default=None, help="Filename separator between Relief_ID and view_index")
    parser.add_argument(
        "--components-path",
        type=str,
        default=None,
        help="Optional frozen component CSV; when supplied, aggregate and plot dependency groups.",
    )

    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--eval-split", type=str, default=None)

    parser.add_argument(
        "--umap-fit-split",
        type=str,
        default=None,
        help="Dataset split name used to fit UMAP (default: train split from config, see --train-split)",
    )
    parser.add_argument(
        "--umap-plot-splits",
        type=str,
        nargs="*",
        default=None,
        help="Splits to project with the fitted UMAP and include in the plot (default: val). Example: val test",
    )

    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--html-out", type=str, default=None, help="Output HTML for interactive Plotly figure")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional CSV with UMAP coordinates")
    parser.add_argument("--png-out", type=str, default=None, help="Optional static PNG for the thesis")
    parser.add_argument(
        "--highlight-relief-ids",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of Relief_IDs to highlight",
    )

    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    cfg = load_yaml_config(args.config) if args.config is not None else {}

    def pick(cli_value, *path, required: bool = True, default=None):
        if cli_value is not None:
            return cli_value

        cur = cfg
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]

        if cur is not None:
            return cur

        if default is not None:
            return default

        if required:
            joined = ".".join(path)
            raise ValueError(f"Missing required configuration value: CLI arg or config field '{joined}'")

        return None

    resolved = {
        "csv_path": pick(args.csv_path, "data", "csv_path"),
        "image_root": pick(args.image_root, "data", "image_root"),
        "filename_sep": pick(args.filename_sep, "data", "filename_sep", default="-"),
        "components_path": args.components_path,

        "train_split": pick(args.train_split, "splits", "train", default="train"),
        "eval_split": pick(args.eval_split, "splits", "test", default="test"),

        "checkpoint_path": pick(args.checkpoint_path, "outputs", "checkpoint_path"),
        "batch_size": pick(args.batch_size, "train", "batch_size", default=16),
        "num_workers": pick(args.num_workers, "train", "num_workers", default=2),

        "html_out": pick(args.html_out, "outputs", "umap_html_path"),
        "csv_out": pick(args.csv_out, "outputs", "umap_csv_path", required=False),
        "png_out": pick(args.png_out, "outputs", "umap_png_path", required=False),

        "highlight_relief_ids": (
            args.highlight_relief_ids
            if args.highlight_relief_ids is not None
            else cfg.get("umap", {}).get("highlight_relief_ids", [])
        ),
    }

    umap_cfg = cfg.get("umap", {}) or {}
    if args.umap_fit_split is not None:
        resolved["umap_fit_split"] = args.umap_fit_split
    elif isinstance(umap_cfg.get("fit_split"), str):
        resolved["umap_fit_split"] = umap_cfg["fit_split"]
    else:
        resolved["umap_fit_split"] = resolved["train_split"]

    if args.umap_plot_splits is not None:
        resolved["umap_plot_splits"] = list(args.umap_plot_splits)
    elif umap_cfg.get("plot_splits") is not None:
        ps = umap_cfg["plot_splits"]
        resolved["umap_plot_splits"] = ps if isinstance(ps, list) else [ps]
    else:
        resolved["umap_plot_splits"] = ["val"]

    return resolved


def source_from_relief_id(relief_id: str) -> str:
    match = re.match(r"^([A-Za-z]+)", str(relief_id).strip())
    source = match.group(1).upper() if match else "Other"
    return source if source in {"AO", "BM", "MET"} else "Other"


def plot_umap_static(viz_df: pd.DataFrame, out_path: str) -> None:
    data = viz_df.copy()
    source_values = data["relief_ids"] if "relief_ids" in data.columns else data["relief_id"]
    data["source"] = source_values.map(source_from_relief_id)
    authority_colors = {
        "Ashurbanipal": "#3B6FB6",
        "Ashurnasirpal II": "#D17A22",
        "Sargon II": "#2F8F5B",
    }
    source_colors = {
        "BM": "#3B6FB6",
        "AO": "#C44E52",
        "MET": "#55A868",
        "Other": "#8172B2",
    }
    split_markers = {"train": "o", "val": "s", "test": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True, sharey=True)
    for ax, column, colors, title in (
        (axes[0], "authority", authority_colors, "A. Colour by ruler"),
        (axes[1], "source", source_colors, "B. Colour by museum/source prefix"),
    ):
        for split, marker in split_markers.items():
            for category, color in colors.items():
                subset = data[(data["split"] == split) & (data[column] == category)]
                if subset.empty:
                    continue
                ax.scatter(
                    subset["umap_x"],
                    subset["umap_y"],
                    s=30 if split == "train" else 46,
                    marker=marker,
                    c=color,
                    edgecolors="white",
                    linewidths=0.45,
                    alpha=0.78 if split == "train" else 0.95,
                )
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.grid(alpha=0.15, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("UMAP 2")

    authority_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=label)
        for label, color in authority_colors.items()
    ]
    source_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=label)
        for label, color in source_colors.items()
    ]
    split_handles = [
        plt.Line2D(
            [], [], marker=marker, linestyle="", color="#555555", label=split
        )
        for split, marker in split_markers.items()
    ]
    axes[0].legend(
        handles=authority_handles + split_handles,
        fontsize=8,
        frameon=False,
        loc="best",
    )
    axes[1].legend(
        handles=source_handles + split_handles,
        fontsize=8,
        frameon=False,
        loc="best",
    )
    fig.suptitle(
        "Dependency-group DINOv2 probe embeddings (UMAP fitted on training groups)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)

    print("Resolved config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    checkpoint = torch.load(cfg["checkpoint_path"], map_location="cpu")


    if "class_to_idx" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'class_to_idx'.")

    class_to_idx: dict[str, int] = checkpoint["class_to_idx"]
    model_name = checkpoint.get("model_name", "dinov2_vits14")
    emb_dim = checkpoint.get("emb_dim", 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("class_to_idx:", class_to_idx)

    model = DinoStyleProbe(
        num_classes=len(class_to_idx),
        emb_dim=emb_dim,
        model_name=model_name,
    ).to(device)

    load_checkpoint(model, cfg["checkpoint_path"], device=device)

    fit_split = cfg["umap_fit_split"]
    plot_splits_raw: list[str] = cfg["umap_plot_splits"]
    seen_plot: set[str] = set()
    plot_splits: list[str] = []
    for s in plot_splits_raw:
        if s not in seen_plot:
            seen_plot.add(s)
            plot_splits.append(s)

    if not plot_splits:
        raise ValueError(
            "No splits to plot; set --umap-plot-splits or config umap.plot_splits (non-empty)."
        )

    required_splits = {fit_split} | set(plot_splits)

    components_df = (
        pd.read_csv(cfg["components_path"])
        if cfg["components_path"] is not None
        else None
    )
    relief_emb_by_split: dict[str, pd.DataFrame] = {}
    for split in sorted(required_splits):
        print(f"Extracting embeddings for split={split!r}...")
        _, loader = build_dataloader(
            csv_path=cfg["csv_path"],
            split=split,
            class_to_idx=class_to_idx,
            image_root=cfg["image_root"],
            filename_sep=cfg["filename_sep"],
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            shuffle=False,
            train=False,
            check_paths=True,
        )
        img_emb_df = extract_embeddings(model, loader, device)
        if components_df is None:
            relief_emb_by_split[split] = aggregate_relief_embeddings(img_emb_df)
        else:
            relief_emb_by_split[split] = aggregate_image_embeddings_by_dependency_group(
                img_emb_df,
                components_df,
                split=split,
            )

    fit_relief_emb_df = relief_emb_by_split[fit_split]
    project_pairs = [(relief_emb_by_split[s], s) for s in plot_splits]

    viz_df = build_umap_dataframe(
        fit_relief_emb_df=fit_relief_emb_df,
        project_relief_emb_dfs=project_pairs,
    )

    html_out = ensure_parent_dir(cfg["html_out"])

    plot_desc = ", ".join(plot_splits)
    fig = plot_umap_plotly(
        viz_df,
        title=f"Interactive UMAP (fit={fit_split!r}, plotted={plot_desc})",
        highlight_relief_ids=set(cfg["highlight_relief_ids"]),
    )
    fig.write_html(str(html_out), include_plotlyjs="cdn")
    print(f"Saved interactive UMAP HTML to: {html_out}")

    if cfg["csv_out"] is not None:
        csv_out = ensure_parent_dir(cfg["csv_out"])
        viz_df.to_csv(csv_out, index=False)
        print(f"Saved UMAP dataframe CSV to: {csv_out}")

    if cfg["png_out"] is not None:
        png_out = ensure_parent_dir(cfg["png_out"])
        plot_umap_static(viz_df, str(png_out))
        print(f"Saved static UMAP PNG to: {png_out}")


if __name__ == "__main__":
    main()
