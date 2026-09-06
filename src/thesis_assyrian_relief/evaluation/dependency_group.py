from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def build_filename_to_component_map(
    components_df: pd.DataFrame,
    *,
    split: str,
) -> dict[str, str]:
    """Map each filename in one frozen split to exactly one dependency group."""
    required = {"component_key", "split", "filenames"}
    missing = required - set(components_df.columns)
    if missing:
        raise ValueError(f"Component table is missing columns: {sorted(missing)}")

    filename_to_component: dict[str, str] = {}
    split_components = components_df.loc[components_df["split"] == split]

    for row in split_components.itertuples(index=False):
        for filename in str(row.filenames).split(";"):
            filename = filename.strip()
            if not filename:
                continue
            previous = filename_to_component.get(filename)
            if previous is not None and previous != row.component_key:
                raise ValueError(
                    f"Filename {filename!r} belongs to multiple dependency groups: "
                    f"{previous!r} and {row.component_key!r}"
                )
            filename_to_component[filename] = row.component_key

    if not filename_to_component:
        raise ValueError(f"No filenames found for split {split!r}")
    return filename_to_component


def aggregate_image_predictions_by_dependency_group(
    image_predictions_df: pd.DataFrame,
    components_df: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    """Aggregate exported image logits at the frozen dependency-group level."""
    required = {
        "image_filename",
        "relief_id",
        "true_label",
        "true_authority",
    }
    missing = required - set(image_predictions_df.columns)
    if missing:
        raise ValueError(f"Image prediction table is missing columns: {sorted(missing)}")

    logit_columns = [
        column for column in image_predictions_df.columns if column.startswith("logit_")
    ]
    if not logit_columns:
        raise ValueError("Image prediction table contains no logit columns")

    filename_to_component = build_filename_to_component_map(
        components_df,
        split=split,
    )
    predictions = image_predictions_df.copy()
    predictions["component_key"] = predictions["image_filename"].map(
        filename_to_component
    )

    missing_filenames = sorted(
        predictions.loc[predictions["component_key"].isna(), "image_filename"].unique()
    )
    if missing_filenames:
        preview = ", ".join(missing_filenames[:5])
        raise ValueError(
            f"{len(missing_filenames)} prediction filenames are absent from the "
            f"{split!r} component map: {preview}"
        )

    rows: list[dict] = []
    for component_key, group in predictions.groupby("component_key", sort=True):
        true_labels = group["true_label"].astype(int).unique()
        true_authorities = group["true_authority"].unique()
        if len(true_labels) != 1 or len(true_authorities) != 1:
            raise ValueError(
                f"Dependency group {component_key!r} contains inconsistent labels"
            )

        mean_logits = group[logit_columns].astype(float).mean(axis=0).to_numpy()
        pred_label = int(mean_logits.argmax())
        confidence = float(_softmax(mean_logits)[pred_label])
        relief_ids = sorted(group["relief_id"].astype(str).unique())

        row = {
            "component_key": component_key,
            "true_label": int(true_labels[0]),
            "true_authority": str(true_authorities[0]),
            "pred_label": pred_label,
            "confidence": confidence,
            "n_images": int(len(group)),
            "n_relief_ids": int(len(relief_ids)),
            "relief_ids": ";".join(relief_ids),
        }
        row.update(
            {
                f"mean_{column}": float(value)
                for column, value in zip(logit_columns, mean_logits, strict=True)
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    expected_components = set(
        components_df.loc[components_df["split"] == split, "component_key"]
    )
    observed_components = set(result["component_key"])
    if observed_components != expected_components:
        missing_components = sorted(expected_components - observed_components)
        extra_components = sorted(observed_components - expected_components)
        raise ValueError(
            "Prediction/component coverage mismatch: "
            f"missing={missing_components[:5]}, extra={extra_components[:5]}"
        )
    return result


def compute_dependency_group_metrics(
    predictions_df: pd.DataFrame,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_260_830,
) -> dict[str, float | int | list[float]]:
    y_true = predictions_df["true_label"].astype(int).to_numpy()
    y_pred = predictions_df["pred_label"].astype(int).to_numpy()
    labels = np.unique(y_true)

    metrics: dict[str, float | int | list[float]] = {
        "n_dependency_groups": int(len(predictions_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
    }
    if bootstrap_replicates <= 0:
        return metrics

    rng = np.random.default_rng(bootstrap_seed)
    accuracies = np.empty(bootstrap_replicates, dtype=float)
    macro_f1s = np.empty(bootstrap_replicates, dtype=float)
    for index in range(bootstrap_replicates):
        sample = rng.integers(0, len(predictions_df), size=len(predictions_df))
        sampled_true = y_true[sample]
        sampled_pred = y_pred[sample]
        accuracies[index] = accuracy_score(sampled_true, sampled_pred)
        macro_f1s[index] = f1_score(
            sampled_true,
            sampled_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )

    metrics["accuracy_bootstrap_95_ci"] = [
        float(value) for value in np.quantile(accuracies, [0.025, 0.975])
    ]
    metrics["macro_f1_bootstrap_95_ci"] = [
        float(value) for value in np.quantile(macro_f1s, [0.025, 0.975])
    ]
    return metrics


def aggregate_image_embeddings_by_dependency_group(
    embeddings_df: pd.DataFrame,
    components_df: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    """Average and normalize image embeddings within frozen dependency groups."""
    required = {"image_path", "relief_id", "label", "authority", "embedding"}
    missing = required - set(embeddings_df.columns)
    if missing:
        raise ValueError(f"Embedding table is missing columns: {sorted(missing)}")

    filename_to_component = build_filename_to_component_map(
        components_df,
        split=split,
    )
    embeddings = embeddings_df.copy()
    embeddings["image_filename"] = embeddings["image_path"].map(
        lambda value: Path(str(value)).name
    )
    embeddings["component_key"] = embeddings["image_filename"].map(
        filename_to_component
    )
    missing_filenames = sorted(
        embeddings.loc[embeddings["component_key"].isna(), "image_filename"].unique()
    )
    if missing_filenames:
        preview = ", ".join(missing_filenames[:5])
        raise ValueError(
            f"{len(missing_filenames)} embedding filenames are absent from the "
            f"{split!r} component map: {preview}"
        )

    rows: list[dict] = []
    for component_key, group in embeddings.groupby("component_key", sort=True):
        labels = group["label"].astype(int).unique()
        authorities = group["authority"].unique()
        if len(labels) != 1 or len(authorities) != 1:
            raise ValueError(
                f"Dependency group {component_key!r} contains inconsistent labels"
            )

        mean_embedding = np.stack(group["embedding"].to_list()).mean(axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm == 0:
            raise ValueError(
                f"Zero embedding norm for dependency group {component_key!r}"
            )
        relief_ids = sorted(group["relief_id"].astype(str).unique())
        rows.append(
            {
                "component_key": component_key,
                "relief_id": component_key,
                "label": int(labels[0]),
                "authority": str(authorities[0]),
                "embedding": mean_embedding / norm,
                "n_images": int(len(group)),
                "n_relief_ids": int(len(relief_ids)),
                "relief_ids": ";".join(relief_ids),
            }
        )

    result = pd.DataFrame(rows)
    expected_components = set(
        components_df.loc[components_df["split"] == split, "component_key"]
    )
    observed_components = set(result["component_key"])
    if observed_components != expected_components:
        raise ValueError(
            "Embedding/component coverage mismatch: "
            f"missing={sorted(expected_components - observed_components)[:5]}, "
            f"extra={sorted(observed_components - expected_components)[:5]}"
        )
    return result
