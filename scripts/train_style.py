from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from thesis_assyrian_relief.utils.data import (
    build_class_to_idx,
    build_dataloader,
    build_class_weights,
)
from thesis_assyrian_relief.models.dinov2_probe import DinoStyleProbe
from thesis_assyrian_relief.training.engine import fit
from thesis_assyrian_relief.utils.config import load_yaml_config, ensure_parent_dir
import warnings

# Ignore specific warning containing the xFormers message
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", message=".*xFormers is available.*")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv2 style probe for Neo-Assyrian reliefs.")

    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")

    parser.add_argument("--csv-path", type=str, default=None, help="Path to image_level_dataset.csv")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory containing image files")
    parser.add_argument("--filename-sep", type=str, default=None, help="Filename separator between Relief_ID and view_index")

    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--val-split", type=str, default=None)

    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--emb-dim", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)

    parser.add_argument("--checkpoint-path", type=str, default=None, help="Where to save best model checkpoint")
    parser.add_argument("--history-path", type=str, default=None, help="Optional CSV path for training history")

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
        "val_split": pick(args.val_split, "splits", "val", default="val"),

        "model_name": pick(args.model_name, "model", "model_name", default="dinov2_vits14"),
        "emb_dim": pick(args.emb_dim, "model", "emb_dim", default=256),

        "batch_size": pick(args.batch_size, "train", "batch_size", default=16),
        "num_workers": pick(args.num_workers, "train", "num_workers", default=2),
        "num_epochs": pick(args.num_epochs, "train", "num_epochs", default=5),
        "lr": pick(args.lr, "train", "lr", default=1e-3),
        "weight_decay": pick(args.weight_decay, "train", "weight_decay", default=1e-4),

        "checkpoint_path": pick(args.checkpoint_path, "outputs", "checkpoint_path"),
        "history_path": pick(args.history_path, "outputs", "history_path", required=False),
    }

    return resolved

def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)

    checkpoint_path = ensure_parent_dir(cfg["checkpoint_path"])
    history_path = (
        ensure_parent_dir(cfg["history_path"])
        if cfg["history_path"] is not None
        else ensure_parent_dir(checkpoint_path.with_suffix(".history.csv"))
    )

    print("Resolved config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    class_to_idx = build_class_to_idx(cfg["csv_path"])
    print("class_to_idx:", class_to_idx)
    
    print("Building train loader...")
    train_dataset, train_loader = build_dataloader(
        csv_path=cfg["csv_path"],
        split=cfg["train_split"],
        class_to_idx=class_to_idx,
        image_root=cfg["image_root"],
        filename_sep=cfg["filename_sep"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=True,
        train=True,
        check_paths=True,
    )

    val_dataset, val_loader = build_dataloader(
        csv_path=cfg["csv_path"],
        split=cfg["val_split"],
        class_to_idx=class_to_idx,
        image_root=cfg["image_root"],
        filename_sep=cfg["filename_sep"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
        train=False,
        check_paths=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DinoStyleProbe(
        num_classes=len(class_to_idx),
        emb_dim=cfg["emb_dim"],
        model_name=cfg["model_name"],
    ).to(device)

    class_weights = build_class_weights(train_dataset, class_to_idx).to(device)
    print("class_weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameter tensors:", len(trainable_params))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    print("Training model...")
    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=cfg["num_epochs"],
        scheduler=scheduler,
        checkpoint_path=str(checkpoint_path),
        extra_checkpoint_data={
            "class_to_idx": class_to_idx,
            "model_name": cfg["model_name"],
            "emb_dim": cfg["emb_dim"],
        },
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)

    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved history CSV to: {history_path}")


if __name__ == "__main__":
    main()