from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_training_curves(
    history_df: pd.DataFrame,
    out_path: str | Path,
    best_epoch: int | None,
    *,
    train_metric_col: str = "train_macro_f1",
    val_metric_col: str = "val_macro_f1",
) -> None:
    """
    Plot train/val loss and metric vs epoch; mark the epoch where the best checkpoint was saved.

    Expects columns: epoch, train_loss, val_loss, and the metric columns above.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = history_df["epoch"].to_numpy()

    fig, (ax_loss, ax_metric) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)

    ax_loss.plot(epochs, history_df["train_loss"], label="train loss", marker="o", markersize=3)
    ax_loss.plot(epochs, history_df["val_loss"], label="val loss", marker="o", markersize=3)
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)

    ax_metric.plot(epochs, history_df[train_metric_col], label=f"train {train_metric_col}", marker="o", markersize=3)
    ax_metric.plot(epochs, history_df[val_metric_col], label=f"val {val_metric_col}", marker="o", markersize=3)
    ax_metric.set_xlabel("epoch")
    ax_metric.set_ylabel("metric")
    ax_metric.grid(True, alpha=0.3)

    if best_epoch is not None and len(epochs) > 0:
        for ax in (ax_loss, ax_metric):
            ax.axvline(
                best_epoch,
                color="tab:red",
                linestyle="--",
                linewidth=1.2,
                label=f"best ckpt (epoch {best_epoch})",
            )

    ax_loss.legend(loc="best")
    ax_metric.legend(loc="best")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
