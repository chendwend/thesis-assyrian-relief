from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from thesis_assyrian_relief.evaluation.dependency_group import (
    aggregate_image_embeddings_by_dependency_group,
)
from thesis_assyrian_relief.evaluation.retrieval import (
    build_class_centroids,
    compute_retrieval_metrics,
    predict_by_nearest_centroid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and centroids by frozen dependency group."
    )
    parser.add_argument("--train-image-embeddings", type=Path, required=True)
    parser.add_argument("--eval-image-embeddings", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-groups-out", type=Path, required=True)
    parser.add_argument("--eval-groups-out", type=Path, required=True)
    parser.add_argument("--retrieval-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    components = pd.read_csv(args.components)
    train_images = pd.read_pickle(args.train_image_embeddings)
    eval_images = pd.read_pickle(args.eval_image_embeddings)
    train_groups = aggregate_image_embeddings_by_dependency_group(
        train_images,
        components,
        split="train",
    )
    eval_groups = aggregate_image_embeddings_by_dependency_group(
        eval_images,
        components,
        split=args.eval_split,
    )

    centroids = build_class_centroids(train_groups)
    y_true, y_pred = predict_by_nearest_centroid(eval_groups, centroids)
    centroid_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    retrieval_metrics, retrieval = compute_retrieval_metrics(
        query_df=eval_groups,
        gallery_df=train_groups,
        ks=(1, 3, 5),
    )
    metrics = {
        "n_train_dependency_groups": int(len(train_groups)),
        "n_eval_dependency_groups": int(len(eval_groups)),
        "nearest_centroid": centroid_metrics,
        "retrieval": retrieval_metrics,
    }

    for path in (
        args.train_groups_out,
        args.eval_groups_out,
        args.retrieval_out,
        args.metrics_out,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    train_groups.to_pickle(args.train_groups_out)
    eval_groups.to_pickle(args.eval_groups_out)
    retrieval.to_csv(args.retrieval_out, index=False)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
