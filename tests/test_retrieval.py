from __future__ import annotations

import numpy as np
import pandas as pd

from thesis_assyrian_relief.evaluation.retrieval import (
    aggregate_relief_embeddings,
    build_class_centroids,
    compute_retrieval_metrics,
    inspect_retrieval_topk,
    predict_by_nearest_centroid,
)


def test_aggregate_relief_embeddings_means_and_normalizes() -> None:
    emb_df = pd.DataFrame(
        [
            {
                "relief_id": "BM 1",
                "authority": "Ashurbanipal",
                "label": 0,
                "image_path": "a.jpg",
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
            },
            {
                "relief_id": "BM 1",
                "authority": "Ashurbanipal",
                "label": 0,
                "image_path": "b.jpg",
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
            },
            {
                "relief_id": "BM 2",
                "authority": "OTHER",
                "label": 2,
                "image_path": "c.jpg",
                "embedding": np.array([0.0, 2.0], dtype=np.float32),
            },
        ]
    )

    relief_df = aggregate_relief_embeddings(emb_df)

    assert len(relief_df) == 2
    row_bm1 = relief_df[relief_df["relief_id"] == "BM 1"].iloc[0]
    row_bm2 = relief_df[relief_df["relief_id"] == "BM 2"].iloc[0]

    assert np.allclose(row_bm1["embedding"], np.array([1.0, 0.0], dtype=np.float32))
    assert np.allclose(row_bm2["embedding"], np.array([0.0, 1.0], dtype=np.float32))
    assert row_bm1["n_views"] == 2
    assert row_bm2["n_views"] == 1


def test_build_class_centroids_returns_normalized_centers() -> None:
    relief_df = pd.DataFrame(
        [
            {
                "relief_id": "A1",
                "authority": "Ashurbanipal",
                "label": 0,
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "n_views": 1,
            },
            {
                "relief_id": "A2",
                "authority": "Ashurbanipal",
                "label": 0,
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "n_views": 1,
            },
            {
                "relief_id": "O1",
                "authority": "OTHER",
                "label": 2,
                "embedding": np.array([0.0, 1.0], dtype=np.float32),
                "n_views": 1,
            },
        ]
    )

    centroids = build_class_centroids(relief_df)

    assert set(centroids.keys()) == {"Ashurbanipal", "OTHER"}
    assert np.allclose(centroids["Ashurbanipal"], np.array([1.0, 0.0], dtype=np.float32))
    assert np.allclose(centroids["OTHER"], np.array([0.0, 1.0], dtype=np.float32))


def test_predict_by_nearest_centroid_behaves_as_expected() -> None:
    relief_df = pd.DataFrame(
        [
            {
                "relief_id": "Q1",
                "authority": "Ashurbanipal",
                "label": 0,
                "embedding": np.array([0.9, 0.1], dtype=np.float32),
                "n_views": 1,
            },
            {
                "relief_id": "Q2",
                "authority": "OTHER",
                "label": 2,
                "embedding": np.array([0.1, 0.9], dtype=np.float32),
                "n_views": 1,
            },
        ]
    )

    centroids = {
        "Ashurbanipal": np.array([1.0, 0.0], dtype=np.float32),
        "OTHER": np.array([0.0, 1.0], dtype=np.float32),
    }

    y_true, y_pred = predict_by_nearest_centroid(relief_df, centroids)

    assert y_true == ["Ashurbanipal", "OTHER"]
    assert y_pred == ["Ashurbanipal", "OTHER"]


def test_compute_retrieval_metrics_on_tiny_controlled_example() -> None:
    gallery_df = pd.DataFrame(
        [
            {
                "relief_id": "G1",
                "authority": "Ashurbanipal",
                "label": 0,
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "n_views": 1,
            },
            {
                "relief_id": "G2",
                "authority": "OTHER",
                "label": 2,
                "embedding": np.array([0.0, 1.0], dtype=np.float32),
                "n_views": 1,
            },
            {
                "relief_id": "G3",
                "authority": "Sargon II",
                "label": 3,
                "embedding": np.array([0.7, 0.7], dtype=np.float32) / np.sqrt(0.98),
                "n_views": 1,
            },
        ]
    )

    query_df = pd.DataFrame(
        [
            {
                "relief_id": "Q1",
                "authority": "Ashurbanipal",
                "label": 0,
                "embedding": np.array([0.95, 0.05], dtype=np.float32),
                "n_views": 1,
            },
            {
                "relief_id": "Q2",
                "authority": "OTHER",
                "label": 2,
                "embedding": np.array([0.05, 0.95], dtype=np.float32),
                "n_views": 1,
            },
        ]
    )

    metrics, results_df = compute_retrieval_metrics(
        query_df=query_df,
        gallery_df=gallery_df,
        ks=(1, 2),
    )

    assert metrics["recall@1"] == 1.0
    assert metrics["recall@2"] == 1.0
    assert len(results_df) == 2
    assert set(results_df.columns) == {
        "query_relief_id",
        "query_authority",
        "top1_relief_id",
        "top1_authority",
        "top1_similarity",
    }


def test_inspect_retrieval_topk_returns_expected_columns() -> None:
    gallery_df = pd.DataFrame(
        [
            {
                "relief_id": "G1",
                "authority": "Ashurbanipal",
                "label": 0,
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "n_views": 1,
            },
            {
                "relief_id": "G2",
                "authority": "OTHER",
                "label": 2,
                "embedding": np.array([0.0, 1.0], dtype=np.float32),
                "n_views": 1,
            },
        ]
    )

    query_df = pd.DataFrame(
        [
            {
                "relief_id": "Q1",
                "authority": "Ashurbanipal",
                "label": 0,
                "embedding": np.array([1.0, 0.0], dtype=np.float32),
                "n_views": 1,
            }
        ]
    )

    out_df = inspect_retrieval_topk(query_df=query_df, gallery_df=gallery_df, k=2)

    assert len(out_df) == 1
    row = out_df.iloc[0]
    assert row["query_relief_id"] == "Q1"
    assert row["query_authority"] == "Ashurbanipal"
    assert bool(row["top1_correct"]) is True
    assert bool(row["top3_hit"]) is True
    assert bool(row["top5_hit"]) is True
    assert len(row["topk_reliefs"]) == 2
    assert len(row["topk_labels"]) == 2
    assert len(row["topk_sims"]) == 2