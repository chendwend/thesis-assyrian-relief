from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from thesis_assyrian_relief.utils.data import build_dataloader

from thesis_assyrian_relief.utils.config import load_yaml_config, ensure_parent_dir
from thesis_assyrian_relief.evaluation.relief_level import evaluate_relief_level
from thesis_assyrian_relief.evaluation.retrieval import (
    aggregate_relief_embeddings,
    build_class_centroids,
    compute_retrieval_metrics,
    extract_embeddings,
    predict_by_nearest_centroid,
)
from thesis_assyrian_relief.models.dinov2_probe import DinoStyleProbe
from thesis_assyrian_relief.training.engine import evaluate, load_checkpoint

import warnings
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")
warnings.filterwarnings("ignore", message=".*xFormers is available.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained style model.")

    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")

    parser.add_argument("--csv-path", type=str, default=None, help="Path to image_level_dataset.csv")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory containing image files")
    parser.add_argument("--filename-sep", type=str, default=None, help="Filename separator between Relief_ID and view_index")

    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--eval-split", type=str, default=None, help="Split to evaluate: val or test")

    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to saved model checkpoint")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--metrics-out", type=str, default=None, help="Optional JSON path for metrics output")
    parser.add_argument("--retrieval-out", type=str, default=None, help="Optional CSV path for retrieval results")

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

        "metrics_out": pick(args.metrics_out, "outputs", "eval_metrics_path", required=False),
        "retrieval_out": pick(args.retrieval_out, "outputs", "eval_retrieval_path", required=False),
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
    num_classes = len(class_to_idx)

    model_name = checkpoint.get("model_name", "dinov2_vits14")
    emb_dim = checkpoint.get("emb_dim", 256)

    print("class_to_idx:", class_to_idx)
    print("model_name:", model_name)
    print("emb_dim:", emb_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DinoStyleProbe(
        num_classes=num_classes,
        emb_dim=emb_dim,
        model_name=model_name,
    ).to(device)

    load_checkpoint(model, cfg["checkpoint_path"], device=device)

    train_ds, train_loader = build_dataloader(
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

    eval_ds, eval_loader = build_dataloader(
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

    criterion = nn.CrossEntropyLoss()

    # 1. image-level evaluation
    image_metrics = evaluate(
        model=model,
        loader=eval_loader,
        criterion=criterion,
        device=device,
    )

    # 2. relief-level evaluation
    relief_metrics, relief_eval_df = evaluate_relief_level(
        model=model,
        loader=eval_loader,
        device=device,
    )

    # 3. embedding extraction
    train_img_emb_df = extract_embeddings(model, train_loader, device)
    eval_img_emb_df = extract_embeddings(model, eval_loader, device)

    train_relief_emb_df = aggregate_relief_embeddings(train_img_emb_df)
    eval_relief_emb_df = aggregate_relief_embeddings(eval_img_emb_df)

    # 4. centroid evaluation
    centroids = build_class_centroids(train_relief_emb_df)
    y_true, y_pred = predict_by_nearest_centroid(eval_relief_emb_df, centroids)

    centroid_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }

    # 5. retrieval evaluation
    retrieval_metrics, retrieval_results_df = compute_retrieval_metrics(
        query_df=eval_relief_emb_df,
        gallery_df=train_relief_emb_df,
        ks=(1, 3, 5),
    )

    all_metrics = {
        "image_level": image_metrics,
        "relief_level": relief_metrics,
        "centroid": centroid_metrics,
        "retrieval": retrieval_metrics,
    }

    print("\n=== Evaluation Summary ===")
    print(json.dumps(all_metrics, indent=2))

    if cfg["metrics_out"] is not None:
        metrics_path = ensure_parent_dir(cfg["metrics_out"])
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Saved metrics JSON to: {metrics_path}")

    if cfg["retrieval_out"] is not None:
        retrieval_path = ensure_parent_dir(cfg["retrieval_out"])
        retrieval_results_df.to_csv(retrieval_path, index=False)
        print(f"Saved retrieval CSV to: {retrieval_path}")

if __name__ == "__main__":
    main()