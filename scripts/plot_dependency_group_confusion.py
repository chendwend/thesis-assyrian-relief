from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thesis_assyrian_relief.evaluation.confusion_plot import (
    save_confusion_matrix_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a confusion matrix from dependency-group predictions."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Test dependency groups")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    save_confusion_matrix_plot(
        predictions["true_label"],
        predictions["pred_label"],
        args.out,
        class_names=["Ashurbanipal", "Ashurnasirpal II", "Sargon II"],
        title=args.title,
    )


if __name__ == "__main__":
    main()
