from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

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


def aggregate_logits_by_relief(
    pred_rows: list[dict], method: str = "mean_logits"
) -> pd.DataFrame:
    """Aggregate image-level predictions/logits by relief.

    The predictions and logits are aggregated by relief id using the mean of the logits.

    Supported methods:
        - mean_logits
        - mean_probs
        - mean_log_probs
        - majority_vote
    

    Args:
        pred_rows: A list of dictionaries containing the predictions and logits for the relief level.
        method: The method to aggregate the predictions and logits by relief (default: mean_logits)

    Returns:
        A pandas DataFrame containing the aggregated predictions and logits by relief id.
    """
    allowed_methods = {
        "mean_logits",
        "mean_probs",
        "mean_log_probs",
        "majority_vote",
    }
    if method not in allowed_methods:
        raise ValueError(f"Unknown relief aggregation method: {method}")


    pred_df = pd.DataFrame(pred_rows)

    grouped_rows = []
    for relief_id, group in pred_df.groupby("relief_id"):
        logits_stack = np.stack(group["logits"].to_list(), axis=0)
        probs_stack = softmax_np(logits_stack, axis=1)

        image_preds = group["pred"].to_numpy()

        true_labels = group["label"].unique()
        true_authorities = group["authority"].unique()

        if len(true_labels) != 1:
            raise ValueError(f"Multiple true labels found for relief_id={relief_id}")
        if len(true_authorities) != 1:
            raise ValueError(f"Multiple authority labels found for relief_id={relief_id}")


        mean_logits = logits_stack.mean(axis=0)
        mean_probs = probs_stack.mean(axis=0)
        mean_log_probs = np.log(probs_stack + 1e-12).mean(axis=0)

        if method == "mean_logits":
            scores = mean_logits
            pred_label = int(scores.argmax())
            confidence = float(softmax_np(scores[None, :], axis=1).max())

        elif method == "mean_probs":
            scores = mean_probs
            pred_label = int(scores.argmax())
            confidence = float(scores.max())

        elif method == "mean_log_probs":
            scores = mean_log_probs
            pred_label = int(scores.argmax())
            confidence = float(softmax_np(scores[None, :], axis=1).max())

        elif method == "majority_vote":
            num_classes = logits_stack.shape[1]
            vote_counts = np.bincount(image_preds, minlength=num_classes)
            vote_candidates = np.flatnonzero(vote_counts == vote_counts.max())


            if len(vote_candidates) == 1:
                pred_label = int(vote_candidates[0])
            else:
                # Tie-break using mean probability
                pred_label = int(vote_candidates[np.argmax(mean_probs[vote_candidates])])

            scores = vote_counts
            confidence = float(vote_counts[pred_label] / len(image_preds))


        grouped_rows.append(
            {
                "relief_id": relief_id,
                "true_label": int(true_labels[0]),
                "true_authority": true_authorities[0],
                "pred_label": pred_label,
                "aggregation_method": method,
                "confidence": confidence,
                "mean_logits": mean_logits,
                "mean_probs": mean_probs,
                "mean_log_probs": mean_log_probs,
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
    aggregation_method: str = "mean_logits",
    ) -> tuple[dict[str, float], pd.DataFrame]:

    pred_rows = collect_predictions_with_logits(model, loader, device)
    relief_df = aggregate_logits_by_relief(pred_rows, method=aggregation_method)
    metrics = compute_relief_level_metrics(relief_df)
    metrics["aggregation_method"] = aggregation_method

    return metrics, relief_df


def evaluate_relief_level_all_methods(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    methods: tuple[str, ...] = (
        "mean_logits",
        "mean_probs",
        "mean_log_probs",
        "majority_vote",
    ),
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:

    pred_rows = collect_predictions_with_logits(model, loader, device)
    all_metrics = {}
    all_dfs = []

    for method in methods:
        relief_df = aggregate_logits_by_relief(
            pred_rows,
            method=method,
        )

        metrics = compute_relief_level_metrics(relief_df)
        metrics["aggregation_method"] = method

        all_metrics[method] = metrics
        all_dfs.append(relief_df)

    all_relief_df = pd.concat(all_dfs, ignore_index=True)

    return all_metrics, all_relief_df
