from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def collect_predictions_with_logits(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> list[dict]:
    """Collect the predictions and logits for the relief level.

    The predictions and logits are collected from the model for each batch in the loader and returned as a list of dictionaries.

    Args:
        model: The model to collect the predictions and logits from.
        loader: The data loader to collect the predictions and logits from.
        device: The device to collect the predictions and logits from.

    Returns:
        A list of dictionaries containing the predictions and logits for the relief level.
    """
    model.eval()

    rows: list[dict] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        relief_ids = batch["relief_id"]
        authorities = batch["authority"]

        logits, emb = model(images)
        logits = logits.cpu().numpy()
        preds = logits.argmax(axis=1)

        for i in range(len(labels)):
            rows.append(
                {
                    "relief_id": relief_ids[i],
                    "authority": authorities[i],
                    "label": int(labels[i]),
                    "pred": int(preds[i]),
                    "logits": logits[i],
                }
            )

    return rows


def aggregate_logits_by_relief(pred_rows: list[dict]) -> pd.DataFrame:
    """Aggregate the predictions and logits by relief.

    The predictions and logits are aggregated by relief id using the mean of the logits.

    Args:
        pred_rows: A list of dictionaries containing the predictions and logits for the relief level.

    Returns:
        A pandas DataFrame containing the aggregated predictions and logits by relief id.
    """
    pred_df = pd.DataFrame(pred_rows)

    grouped_rows = []
    for relief_id, group in pred_df.groupby("relief_id"):
        logits_stack = np.stack(group["logits"].to_list(), axis=0)
        mean_logits = logits_stack.mean(axis=0)

        true_labels = group["label"].unique()
        true_authorities = group["authority"].unique()

        if len(true_labels) != 1:
            raise ValueError(f"Multiple true labels found for relief_id={relief_id}")
        if len(true_authorities) != 1:
            raise ValueError(f"Multiple authority labels found for relief_id={relief_id}")

        grouped_rows.append(
            {
                "relief_id": relief_id,
                "true_label": int(true_labels[0]),
                "true_authority": true_authorities[0],
                "pred_label": int(mean_logits.argmax()),
                "mean_logits": mean_logits,
                "n_views": len(group),
            }
        )

    return pd.DataFrame(grouped_rows)


def compute_relief_level_metrics(relief_df: pd.DataFrame) -> dict[str, float]:
    y_true = relief_df["true_label"].to_numpy()
    y_pred = relief_df["pred_label"].to_numpy()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_relief_level(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    pred_rows = collect_predictions_with_logits(model, loader, device)
    relief_df = aggregate_logits_by_relief(pred_rows)
    metrics = compute_relief_level_metrics(relief_df)
    return metrics, relief_df