from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_assyrian_relief.evaluation.relief_level import (
    aggregate_logits_by_relief,
    compute_relief_level_metrics,
)


def test_aggregate_logits_by_relief_means_across_views() -> None:
    pred_rows = [
        {
            "relief_id": "BM 1",
            "authority": "Ashurbanipal",
            "label": 0,
            "pred": 0,
            "logits": np.array([3.0, 1.0, 0.0, 0.0], dtype=np.float32),
        },
        {
            "relief_id": "BM 1",
            "authority": "Ashurbanipal",
            "label": 0,
            "pred": 1,
            "logits": np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32),
        },
        {
            "relief_id": "BM 2",
            "authority": "OTHER",
            "label": 2,
            "pred": 2,
            "logits": np.array([0.0, 0.0, 4.0, 0.0], dtype=np.float32),
        },
    ]

    relief_df = aggregate_logits_by_relief(pred_rows)

    assert len(relief_df) == 2
    assert set(relief_df["relief_id"]) == {"BM 1", "BM 2"}

    row_bm1 = relief_df[relief_df["relief_id"] == "BM 1"].iloc[0]
    expected_mean = np.array([2.0, 1.5, 0.0, 0.0], dtype=np.float32)

    assert np.allclose(row_bm1["mean_logits"], expected_mean)
    assert row_bm1["pred_label"] == 0
    assert row_bm1["n_views"] == 2


def test_compute_relief_level_metrics_returns_expected_values() -> None:
    relief_df = pd.DataFrame(
        [
            {"relief_id": "BM 1", "true_label": 0, "pred_label": 0},
            {"relief_id": "BM 2", "true_label": 1, "pred_label": 1},
            {"relief_id": "BM 3", "true_label": 2, "pred_label": 0},
            {"relief_id": "BM 4", "true_label": 3, "pred_label": 3},
        ]
    )

    metrics = compute_relief_level_metrics(relief_df)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_aggregate_logits_by_relief_raises_on_inconsistent_labels() -> None:
    pred_rows = [
        {
            "relief_id": "BM 1",
            "authority": "Ashurbanipal",
            "label": 0,
            "pred": 0,
            "logits": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        },
        {
            "relief_id": "BM 1",
            "authority": "OTHER",
            "label": 2,
            "pred": 2,
            "logits": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        },
    ]

    with pytest.raises(ValueError):
        aggregate_logits_by_relief(pred_rows)


def test_aggregate_logits_by_relief_uses_single_row_per_relief() -> None:
    pred_rows = [
        {
            "relief_id": "AO 1",
            "authority": "Sargon II",
            "label": 3,
            "pred": 3,
            "logits": np.array([0.0, 0.0, 0.0, 5.0], dtype=np.float32),
        },
        {
            "relief_id": "AO 1",
            "authority": "Sargon II",
            "label": 3,
            "pred": 3,
            "logits": np.array([0.0, 0.0, 0.0, 6.0], dtype=np.float32),
        },
        {
            "relief_id": "AO 2",
            "authority": "Ashurnasirpal II",
            "label": 1,
            "pred": 1,
            "logits": np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32),
        },
    ]

    relief_df = aggregate_logits_by_relief(pred_rows)

    assert len(relief_df) == 2
    assert relief_df["relief_id"].nunique() == 2