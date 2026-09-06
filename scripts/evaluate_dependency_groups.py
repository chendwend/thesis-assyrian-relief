from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thesis_assyrian_relief.evaluation.dependency_group import (
    aggregate_image_predictions_by_dependency_group,
    compute_dependency_group_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-aggregate exported image logits by frozen dependency group."
    )
    parser.add_argument("--image-predictions", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_predictions = pd.read_csv(args.image_predictions)
    components = pd.read_csv(args.components)
    dependency_predictions = aggregate_image_predictions_by_dependency_group(
        image_predictions,
        components,
        split=args.split,
    )
    metrics = compute_dependency_group_metrics(
        dependency_predictions,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    dependency_predictions.to_csv(args.predictions_out, index=False)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
