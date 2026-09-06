from __future__ import annotations

import pandas as pd
import pytest

from thesis_assyrian_relief.evaluation.dependency_group import (
    aggregate_image_embeddings_by_dependency_group,
    aggregate_image_predictions_by_dependency_group,
    compute_dependency_group_metrics,
)


def test_dependency_group_aggregation_combines_linked_relief_ids() -> None:
    images = pd.DataFrame(
        [
            {
                "image_filename": "BM 1-1.jpg",
                "relief_id": "BM 1",
                "true_label": 0,
                "true_authority": "Ashurbanipal",
                "logit_Ashurbanipal": 3.0,
                "logit_Ashurnasirpal_II": 1.0,
                "logit_Sargon_II": 0.0,
            },
            {
                "image_filename": "BM 2-1.jpg",
                "relief_id": "BM 2",
                "true_label": 0,
                "true_authority": "Ashurbanipal",
                "logit_Ashurbanipal": 1.0,
                "logit_Ashurnasirpal_II": 2.0,
                "logit_Sargon_II": 0.0,
            },
        ]
    )
    components = pd.DataFrame(
        [
            {
                "component_key": "component:1",
                "split": "test",
                "filenames": "BM 1-1.jpg;BM 2-1.jpg",
            }
        ]
    )

    result = aggregate_image_predictions_by_dependency_group(
        images,
        components,
        split="test",
    )

    assert len(result) == 1
    assert result.iloc[0]["pred_label"] == 0
    assert result.iloc[0]["n_images"] == 2
    assert result.iloc[0]["n_relief_ids"] == 2
    assert result.iloc[0]["relief_ids"] == "BM 1;BM 2"


def test_dependency_group_aggregation_rejects_unmapped_predictions() -> None:
    images = pd.DataFrame(
        [
            {
                "image_filename": "AO 1-1.jpg",
                "relief_id": "AO 1",
                "true_label": 2,
                "true_authority": "Sargon II",
                "logit_Ashurbanipal": 0.0,
                "logit_Ashurnasirpal_II": 0.0,
                "logit_Sargon_II": 2.0,
            }
        ]
    )
    components = pd.DataFrame(
        [
            {
                "component_key": "component:1",
                "split": "test",
                "filenames": "different.jpg",
            }
        ]
    )

    with pytest.raises(ValueError, match="absent from the 'test' component map"):
        aggregate_image_predictions_by_dependency_group(
            images,
            components,
            split="test",
        )


def test_dependency_group_metrics_report_sample_size_and_intervals() -> None:
    predictions = pd.DataFrame(
        [
            {"true_label": 0, "pred_label": 0},
            {"true_label": 1, "pred_label": 1},
            {"true_label": 2, "pred_label": 1},
        ]
    )

    metrics = compute_dependency_group_metrics(
        predictions,
        bootstrap_replicates=100,
        bootstrap_seed=24,
    )

    assert metrics["n_dependency_groups"] == 3
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert "accuracy_bootstrap_95_ci" in metrics
    assert "macro_f1_bootstrap_95_ci" in metrics


def test_dependency_group_embedding_aggregation_normalizes_group_mean() -> None:
    embeddings = pd.DataFrame(
        [
            {
                "image_path": "/images/BM 1-1.jpg",
                "relief_id": "BM 1",
                "label": 0,
                "authority": "Ashurbanipal",
                "embedding": [2.0, 0.0],
            },
            {
                "image_path": "/images/BM 2-1.jpg",
                "relief_id": "BM 2",
                "label": 0,
                "authority": "Ashurbanipal",
                "embedding": [0.0, 2.0],
            },
        ]
    )
    components = pd.DataFrame(
        [
            {
                "component_key": "component:1",
                "split": "train",
                "filenames": "BM 1-1.jpg;BM 2-1.jpg",
            }
        ]
    )

    result = aggregate_image_embeddings_by_dependency_group(
        embeddings,
        components,
        split="train",
    )

    assert len(result) == 1
    assert result.iloc[0]["embedding"] == pytest.approx([2**-0.5, 2**-0.5])
    assert result.iloc[0]["n_relief_ids"] == 2
