"""Summarize repeated grouped-split experiments into CSV and JSON."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (24, 42, 77)


def main() -> None:
    rows = []
    for seed in SEEDS:
        run_dir = ROOT / "outputs" / f"grouped_seed{seed}"
        metrics = json.loads(
            (run_dir / "eval" / "test_metrics.json").read_text(encoding="utf-8")
        )
        history = pd.read_csv(
            run_dir / "checkpoints" / "dinov2_probe.history.csv"
        )
        best_index = int(history["val_macro_f1"].idxmax())
        rows.append(
            {
                "seed": seed,
                "best_epoch": best_index + 1,
                "val_macro_f1": float(history.loc[best_index, "val_macro_f1"]),
                "image_accuracy": metrics["image_level"]["accuracy"],
                "image_macro_f1": metrics["image_level"]["macro_f1"],
                "record_accuracy": metrics["relief_level"]["selected_metrics"][
                    "accuracy"
                ],
                "record_macro_f1": metrics["relief_level"]["selected_metrics"][
                    "macro_f1"
                ],
                "centroid_accuracy": metrics["centroid"]["accuracy"],
                "centroid_macro_f1": metrics["centroid"]["macro_f1"],
                "retrieval_recall_at_1": metrics["retrieval"]["recall@1"],
                "retrieval_recall_at_3": metrics["retrieval"]["recall@3"],
                "retrieval_recall_at_5": metrics["retrieval"]["recall@5"],
            }
        )

    summary = {}
    for key in rows[0]:
        if key in {"seed", "best_epoch"}:
            continue
        values = [float(row[key]) for row in rows]
        summary[key] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }

    output_dir = ROOT / "outputs" / "grouped_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "repeated_seed_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "repeated_seed_summary.json"
    json_path.write_text(
        json.dumps({"runs": rows, "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"runs": rows, "summary": summary}, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
