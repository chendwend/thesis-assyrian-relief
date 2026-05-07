from __future__ import annotations

import argparse

import torch
from thesis_assyrian_relief.utils.data import build_dataloader

from thesis_assyrian_relief.utils.config import load_yaml_config, ensure_parent_dir
from thesis_assyrian_relief.evaluation.retrieval import (
    aggregate_relief_embeddings,
    extract_embeddings,
)
from thesis_assyrian_relief.models.dinov2_probe import DinoStyleProbe
from thesis_assyrian_relief.training.engine import load_checkpoint
from thesis_assyrian_relief.utils.plotting import (
    build_umap_dataframe,
    plot_umap_plotly,
)

import warnings
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", message=".*xFormers is available.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UMAP visualization for relief style embeddings.")

    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")

    parser.add_argument("--csv-path", type=str, default=None, help="Path to image_level_dataset.csv")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory containing image files")
    parser.add_argument("--filename-sep", type=str, default=None, help="Filename separator between Relief_ID and view_index")

    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--eval-split", type=str, default=None)

    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--html-out", type=str, default=None, help="Output HTML for interactive Plotly figure")
    parser.add_argument("--csv-out", type=str, default=None, help="Optional CSV with UMAP coordinates")
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

        "train_split": pick(args.train_split, "splits", "train", default="train"),
        "eval_split": pick(args.eval_split, "splits", "test", default="test"),

        "checkpoint_path": pick(args.checkpoint_path, "outputs", "checkpoint_path"),
        "batch_size": pick(args.batch_size, "train", "batch_size", default=16),
        "num_workers": pick(args.num_workers, "train", "num_workers", default=2),

        "html_out": pick(args.html_out, "outputs", "umap_html_path"),
        "csv_out": pick(args.csv_out, "outputs", "umap_csv_path", required=False),

        "highlight_relief_ids": (
            args.highlight_relief_ids
            if args.highlight_relief_ids is not None
            else cfg.get("umap", {}).get("highlight_relief_ids", [])
        ),
    }

    return resolved


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

    _, train_loader = build_dataloader(
        csv_path=cfg["csv_path"],
        split=cfg["train_split"],
        class_to_idx=class_to_idx,
        image_root=cfg["image_root"],
        filename_sep=cfg["filename_sep"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
        train=False,
        check_paths=True,
    )

    _, eval_loader = build_dataloader(
        csv_path=cfg["csv_path"],
        split=cfg["eval_split"],
        class_to_idx=class_to_idx,
        image_root=cfg["image_root"],
        filename_sep=cfg["filename_sep"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
        train=False,
        check_paths=True,
    )

    print("Extracting train embeddings...")
    train_img_emb_df = extract_embeddings(model, train_loader, device)
    print("Extracting eval embeddings...")
    eval_img_emb_df = extract_embeddings(model, eval_loader, device)

    train_relief_emb_df = aggregate_relief_embeddings(train_img_emb_df)
    eval_relief_emb_df = aggregate_relief_embeddings(eval_img_emb_df)

    viz_df = build_umap_dataframe(
        relief_emb_dfs=[train_relief_emb_df, eval_relief_emb_df],
        split_names=[cfg["train_split"], cfg["eval_split"]],
    )

    html_out = ensure_parent_dir(cfg["html_out"])

    fig = plot_umap_plotly(
        viz_df,
        title=f"Interactive UMAP of Relief-Level Embeddings ({cfg["train_split"]} + {cfg["eval_split"]})",
        highlight_relief_ids=set(cfg["highlight_relief_ids"]),
    )
    fig.write_html(str(html_out), include_plotlyjs="cdn")
    print(f"Saved interactive UMAP HTML to: {html_out}")

    if cfg["csv_out"] is not None:
        csv_out = ensure_parent_dir(cfg["csv_out"])
        viz_df.to_csv(csv_out, index=False)
        print(f"Saved UMAP dataframe CSV to: {csv_out}")


if __name__ == "__main__":
    main()