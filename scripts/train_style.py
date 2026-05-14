from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn as nn

from thesis_assyrian_relief.utils.data import (
    build_class_to_idx,
    build_dataloader,
    build_class_weights,
)
from thesis_assyrian_relief.models.dinov2_probe import DinoStyleProbe
from thesis_assyrian_relief.training.curves import save_training_curves
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
    parser.add_argument(
        "--training-curve-path",
        type=str,
        default=None,
        help="Where to save loss/metric training curves (PNG). Defaults next to checkpoint if omitted.",
    )

    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    cfg = load_yaml_config(args.config) if args.config is not None else {}

    def pick(cli_value, *path, required: bool = True, default=None):
        """
        Resolve a configuration value from the CLI or config file.
        """
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
        "optimizer_name": pick(None,"optimizer","name",required=False,default="adamw",),

        "lr": pick(args.lr,"optimizer","lr",required=False,default=pick(None, "train", "lr", 
        required=False, default=1e-3),
        ),

        "weight_decay": pick(
            args.weight_decay,
            "optimizer",
            "weight_decay",
            required=False,
            default=pick(None, "train", "weight_decay", required=False, default=1e-4),
        ),

        "loss_name": pick(
            None,
            "loss",
            "name",
            required=False,
            default="cross_entropy",
        ),

        "class_weighting": pick(
            None,
            "loss",
            "class_weighting",
            required=False,
            default="balanced",
        ),

        "scheduler_enabled": pick(
            None,
            "scheduler",
            "enabled",
            required=False,
            default=True,
        ),

        "scheduler_name": pick(
            None,
            "scheduler",
            "name",
            required=False,
            default="reduce_on_plateau",
        ),

        "scheduler_mode": pick(
            None,
            "scheduler",
            "mode",
            required=False,
            default="max",
        ),

        "scheduler_factor": pick(
            None,
            "scheduler",
            "factor",
            required=False,
            default=0.5,
        ),

        "scheduler_patience": pick(
            None,
            "scheduler",
            "patience",
            required=False,
            default=2,
        ),
        "scheduler_monitor": pick(
            None,
            "scheduler",
            "monitor",
            required=False,
            default="val_macro_f1",
        ),

        "checkpoint_monitor": pick(
            None,
            "checkpoint",
            "monitor",
            required=False,
            default="val_macro_f1",
        ),

        "checkpoint_mode": pick(
            None,
            "checkpoint",
            "mode",
            required=False,
            default="max",
        ),

        "checkpoint_path": pick(args.checkpoint_path, "outputs", "checkpoint_path"),
        "history_path": pick(args.history_path, "outputs", "history_path", required=False),
        "training_curve_path": pick(
            args.training_curve_path,
            "outputs",
            "training_curve_path",
            required=False,
            default=None,
        ),

        "early_stopping_enabled": pick(
            None,
            "early_stopping",
            "enabled",
            required=False,
            default=False,
        ),

        "early_stopping_monitor": pick(
            None,
            "early_stopping",
            "monitor",
            required=False,
            default="val_macro_f1",
        ),

        "early_stopping_mode": pick(
            None,
            "early_stopping",
            "mode",
            required=False,
            default="max",
        ),

        "early_stopping_patience": pick(
            None,
            "early_stopping",
            "patience",
            required=False,
            default=10,
        ),

        "early_stopping_min_delta": pick(
            None,
            "early_stopping",
            "min_delta",
            required=False,
            default=0.0,
        ),
            }

    return resolved

def build_loss(
    loss_name: str,
    class_weighting: str,
    train_dataset,
    class_to_idx: dict[str, int],
    device: torch.device,
) -> nn.Module:
    if loss_name != "cross_entropy":
        raise ValueError(
            f"Unsupported loss.name={loss_name!r}. "
            "Currently supported: 'cross_entropy'."
        )

    if class_weighting == "balanced":
        class_weights = build_class_weights(train_dataset, class_to_idx).to(device)
        print("class_weights:", class_weights)
        return nn.CrossEntropyLoss(weight=class_weights)

    if class_weighting in {"none", None}:
        print("class_weights: none")
        return nn.CrossEntropyLoss()

    raise ValueError(
        f"Unsupported loss.class_weighting={class_weighting!r}. "
        "Supported: 'balanced', 'none'."
    )


def build_optimizer(
    optimizer_name: str,
    trainable_params,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )

    raise ValueError(
        f"Unsupported optimizer.name={optimizer_name!r}. "
        "Supported: 'adamw', 'adam', 'sgd'."
    )


def build_scheduler(
    scheduler_enabled: bool,
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    mode: str,
    factor: float,
    patience: int,
):
    if not scheduler_enabled:
        print("scheduler: disabled")
        return None

    if scheduler_name == "reduce_on_plateau":
        print(
            "scheduler: ReduceLROnPlateau "
            f"(mode={mode}, factor={factor}, patience={patience})"
        )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
        )

    if scheduler_name == "cosine":
        raise ValueError(
            "scheduler.name='cosine' is not supported yet because it requires "
            "num_epochs/T_max handling. We can add it next."
        )

    raise ValueError(
        f"Unsupported scheduler.name={scheduler_name!r}. "
        "Supported now: 'reduce_on_plateau'."
    )

def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)

    # check if early_stopping.monitor and checkpoint.monitor are the same
    if cfg["early_stopping_monitor"] != cfg["checkpoint_monitor"]:
        raise ValueError(
            "For now, early_stopping.monitor must match checkpoint.monitor. "
            f"Got early_stopping.monitor={cfg['early_stopping_monitor']!r}, "
            f"checkpoint.monitor={cfg['checkpoint_monitor']!r}."
        )


    # check if early_stopping.mode and checkpoint.mode are the same
    if cfg["early_stopping_mode"] != cfg["checkpoint_mode"]:
        raise ValueError(
            "For now, early_stopping.mode must match checkpoint.mode. "
            f"Got early_stopping.mode={cfg['early_stopping_mode']!r}, "
            f"checkpoint.mode={cfg['checkpoint_mode']!r}."
        )

    
    checkpoint_path = ensure_parent_dir(cfg["checkpoint_path"])
    history_path = (
        ensure_parent_dir(cfg["history_path"])
        if cfg["history_path"] is not None
        else ensure_parent_dir(checkpoint_path.with_suffix(".history.csv"))
    )

    training_curve_path = cfg["training_curve_path"]
    if training_curve_path is None:
        training_curve_path = str(checkpoint_path.with_suffix(".curves.png"))
    training_curve_path = ensure_parent_dir(training_curve_path)

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

    criterion = build_loss(
        loss_name=cfg["loss_name"],
        class_weighting=cfg["class_weighting"],
        train_dataset=train_dataset,
        class_to_idx=class_to_idx,
        device=device,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable parameter tensors:", len(trainable_params))

    trainable_total = 0

    # print total number of trainable parameters and their shapes
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"{name:60s} shape={tuple(p.shape)} numel={p.numel():,}")
            trainable_total += p.numel()

    print(f"\nTotal trainable scalar parameters: {trainable_total:,}")

    optimizer = build_optimizer(
        optimizer_name=cfg["optimizer_name"],
        trainable_params=trainable_params,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    print(
        f"optimizer: {cfg['optimizer_name']} "
        f"(lr={cfg['lr']}, weight_decay={cfg['weight_decay']})"
    )

    scheduler = build_scheduler(
        scheduler_enabled=cfg["scheduler_enabled"],
        scheduler_name=cfg["scheduler_name"],
        optimizer=optimizer,
        mode=cfg["scheduler_mode"],
        factor=cfg["scheduler_factor"],
        patience=cfg["scheduler_patience"],
    )

    print("Training model...")
    history, best_epoch = fit(
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
            "optimizer_name": cfg["optimizer_name"],
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "loss_name": cfg["loss_name"],
            "class_weighting": cfg["class_weighting"],
            "scheduler_enabled": cfg["scheduler_enabled"],
            "scheduler_name": cfg["scheduler_name"],
            "scheduler_mode": cfg["scheduler_mode"],
            "scheduler_factor": cfg["scheduler_factor"],
            "scheduler_patience": cfg["scheduler_patience"],
            "checkpoint_monitor": cfg["checkpoint_monitor"],
            "checkpoint_mode": cfg["checkpoint_mode"],
            "scheduler_monitor": cfg["scheduler_monitor"],
            "early_stopping_enabled": cfg["early_stopping_enabled"],
            "early_stopping_monitor": cfg["early_stopping_monitor"],
            "early_stopping_mode": cfg["early_stopping_mode"],
            "early_stopping_patience": cfg["early_stopping_patience"],
            "early_stopping_min_delta": cfg["early_stopping_min_delta"],
        },
        monitor_metric=cfg["checkpoint_monitor"],
        monitor_mode=cfg["checkpoint_mode"],
        early_stopping_enabled=cfg["early_stopping_enabled"],
        early_stopping_patience=cfg["early_stopping_patience"],
        early_stopping_min_delta=cfg["early_stopping_min_delta"],
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)

    save_training_curves(history_df, training_curve_path, best_epoch)

    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved history CSV to: {history_path}")
    print(f"Saved training curves to: {training_curve_path}")


if __name__ == "__main__":
    main()