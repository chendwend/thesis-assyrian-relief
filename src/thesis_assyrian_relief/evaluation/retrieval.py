from __future__ import annotations

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """
    Extract image-level embeddings from a model that returns (logits, emb).
    """
    model.eval()

    rows: list[dict] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        relief_ids = batch["relief_id"]
        authorities = batch["authority"]
        image_paths = batch["image_path"]

        logits, emb = model(images)
        emb = emb.cpu().numpy()

        for i in range(len(labels)):
            rows.append(
                {
                    "relief_id": relief_ids[i],
                    "authority": authorities[i],
                    "label": int(labels[i]),
                    "image_path": image_paths[i],
                    "embedding": emb[i],
                }
            )

    return pd.DataFrame(rows)


def aggregate_relief_embeddings(emb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average image-level embeddings into one normalized embedding per Relief_ID.
    """
    grouped_rows: list[dict] = []

    for relief_id, group in emb_df.groupby("relief_id"):
        emb_stack = np.stack(group["embedding"].to_list(), axis=0)
        mean_emb = emb_stack.mean(axis=0)

        norm = np.linalg.norm(mean_emb)
        if norm == 0:
            raise ValueError(f"Zero embedding norm encountered for relief_id={relief_id}")
        mean_emb = mean_emb / norm

        labels = group["label"].unique()
        authorities = group["authority"].unique()

        if len(labels) != 1:
            raise ValueError(f"Multiple labels found for relief_id={relief_id}")
        if len(authorities) != 1:
            raise ValueError(f"Multiple authorities found for relief_id={relief_id}")

        grouped_rows.append(
            {
                "relief_id": relief_id,
                "label": int(labels[0]),
                "authority": authorities[0],
                "embedding": mean_emb,
                "n_views": len(group),
            }
        )

    return pd.DataFrame(grouped_rows)


def build_class_centroids(relief_emb_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Build one centroid per authority label from relief-level embeddings.
    The centroids are calculated as the mean of the embeddings for each authority label.
    The centroids are L2-normalized.

    Args:
        relief_emb_df: A pandas DataFrame containing the relief-level embeddings.
        

    Returns:
        A dictionary containing the centroids for each authority label as L2-normalized vectors.
    """
    centroids: dict[str, np.ndarray] = {}

    for authority, group in relief_emb_df.groupby("authority"):
        emb_stack = np.stack(group["embedding"].to_list(), axis=0)
        centroid = emb_stack.mean(axis=0)

        norm = np.linalg.norm(centroid)
        if norm == 0:
            raise ValueError(f"Zero centroid norm encountered for authority={authority}")
        centroid = centroid / norm

        centroids[authority] = centroid

    return centroids


def predict_by_nearest_centroid(
    relief_emb_df: pd.DataFrame,
    centroids: dict[str, np.ndarray],
) -> tuple[list[str], list[str]]:
    """
    Predict authority by maximum cosine similarity to class centroids.
    Assumes embeddings and centroids are L2-normalized.
    """
    class_names = list(centroids.keys())

    y_true: list[str] = []
    y_pred: list[str] = []

    for _, row in relief_emb_df.iterrows():
        emb = row["embedding"]
        sims = [float(np.dot(emb, centroids[c])) for c in class_names]
        pred_class = class_names[int(np.argmax(sims))]

        y_true.append(row["authority"])
        y_pred.append(pred_class)

    return y_true, y_pred


def compute_retrieval_metrics(
    query_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    ks: tuple[int, ...] = (1, 3, 5),
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Query each relief in query_df against gallery_df using cosine similarity.
    Relevance is defined as same authority label.
    """
    if len(query_df) == 0:
        raise ValueError("query_df is empty.")
    if len(gallery_df) == 0:
        raise ValueError("gallery_df is empty.")

    gallery_embs = np.stack(gallery_df["embedding"].to_list(), axis=0)
    gallery_labels = gallery_df["authority"].to_list()
    gallery_relief_ids = gallery_df["relief_id"].to_list()

    results: list[dict] = []
    hits = {k: 0 for k in ks}

    for _, row in query_df.iterrows():
        q_emb = row["embedding"]
        q_label = row["authority"]
        q_relief_id = row["relief_id"]

        sims = gallery_embs @ q_emb
        ranked_idx = np.argsort(-sims)

        ranked_labels = [gallery_labels[i] for i in ranked_idx]
        ranked_reliefs = [gallery_relief_ids[i] for i in ranked_idx]
        ranked_sims = [float(sims[i]) for i in ranked_idx]

        for k in ks:
            topk_labels = ranked_labels[:k]
            if q_label in topk_labels:
                hits[k] += 1

        results.append(
            {
                "query_relief_id": q_relief_id,
                "query_authority": q_label,
                "top1_relief_id": ranked_reliefs[0],
                "top1_authority": ranked_labels[0],
                "top1_similarity": ranked_sims[0],
            }
        )

    metrics = {f"recall@{k}": hits[k] / len(query_df) for k in ks}
    results_df = pd.DataFrame(results)

    return metrics, results_df


def inspect_retrieval_topk(
    query_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """
    Return top-k retrieval details per query relief for qualitative inspection.
    """
    if len(query_df) == 0:
        raise ValueError("query_df is empty.")
    if len(gallery_df) == 0:
        raise ValueError("gallery_df is empty.")
    if k <= 0:
        raise ValueError("k must be positive.")

    gallery_embs = np.stack(gallery_df["embedding"].to_list(), axis=0)
    gallery_labels = gallery_df["authority"].to_list()
    gallery_relief_ids = gallery_df["relief_id"].to_list()

    rows: list[dict] = []

    for _, row in query_df.iterrows():
        q_emb = row["embedding"]
        q_label = row["authority"]
        q_relief_id = row["relief_id"]

        sims = gallery_embs @ q_emb
        ranked_idx = np.argsort(-sims)[:k]

        topk_reliefs = [gallery_relief_ids[i] for i in ranked_idx]
        topk_labels = [gallery_labels[i] for i in ranked_idx]
        topk_sims = [float(sims[i]) for i in ranked_idx]

        rows.append(
            {
                "query_relief_id": q_relief_id,
                "query_authority": q_label,
                "topk_reliefs": topk_reliefs,
                "topk_labels": topk_labels,
                "topk_sims": topk_sims,
                "top1_correct": topk_labels[0] == q_label,
                "top3_hit": q_label in topk_labels[: min(3, k)],
                "top5_hit": q_label in topk_labels[: min(5, k)],
            }
        )

    return pd.DataFrame(rows)