from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import umap


def build_umap_dataframe(
    relief_emb_dfs: list[pd.DataFrame],
    split_names: list[str],
    n_neighbors: int = 10,
    min_dist: float = 0.2,
    metric: str = "cosine",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Combine one or more relief-level embedding dataframes and compute a 2D UMAP projection.

    Each relief dataframe is expected to contain:
        - relief_id
        - authority
        - embedding
        - n_views (optional)

    Parameters
    ----------
    relief_emb_dfs:
        List of relief-level embedding dataframes.
    split_names:
        Same length as relief_emb_dfs. Example: ["train", "test"].
    """
    if len(relief_emb_dfs) != len(split_names):
        raise ValueError("relief_emb_dfs and split_names must have the same length.")

    frames = []
    for df, split_name in zip(relief_emb_dfs, split_names):
        cur = df.copy()
        cur["split"] = split_name
        frames.append(cur)

    viz_df = pd.concat(frames, ignore_index=True)

    if viz_df.empty:
        raise ValueError("No embeddings available for UMAP visualization.")

    required_cols = {"relief_id", "authority", "embedding", "split"}
    missing = required_cols - set(viz_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for UMAP dataframe: {sorted(missing)}")

    X = np.stack(viz_df["embedding"].to_list(), axis=0)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    X_2d = reducer.fit_transform(X)

    viz_df = viz_df.copy()
    viz_df["umap_x"] = X_2d[:, 0]
    viz_df["umap_y"] = X_2d[:, 1]

    return viz_df


def plot_umap_matplotlib(
    viz_df: pd.DataFrame,
    annotate_relief_ids: set[str] | None = None,
    title: str = "UMAP of Relief-Level Embeddings",
    figsize: tuple[int, int] = (10, 8),
):
    """
    Static matplotlib UMAP plot.
    """
    required_cols = {"authority", "relief_id", "umap_x", "umap_y"}
    missing = required_cols - set(viz_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for plotting: {sorted(missing)}")

    annotate_relief_ids = annotate_relief_ids or set()

    fig, ax = plt.subplots(figsize=figsize)

    classes = sorted(viz_df["authority"].unique())
    for cls in classes:
        mask = viz_df["authority"] == cls
        ax.scatter(
            viz_df.loc[mask, "umap_x"],
            viz_df.loc[mask, "umap_y"],
            label=cls,
            alpha=0.8,
            s=50,
        )

    for _, row in viz_df.iterrows():
        if row["relief_id"] in annotate_relief_ids:
            ax.annotate(
                row["relief_id"],
                (row["umap_x"], row["umap_y"]),
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True)
    ax.legend()

    return fig, ax


def plot_umap_plotly(
    viz_df: pd.DataFrame,
    title: str = "Interactive UMAP of Relief-Level Embeddings",
    highlight_relief_ids: set[str] | None = None,
):
    """
    Interactive Plotly UMAP plot.

    - Color encodes authority
    - Highlighted reliefs are overlaid as a separate trace with larger markers
    """
    required_cols = {"authority", "relief_id", "split", "umap_x", "umap_y"}
    missing = required_cols - set(viz_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for plotting: {sorted(missing)}")

    plot_df = viz_df.copy()
    highlight_relief_ids = highlight_relief_ids or set()
    plot_df["is_highlighted"] = plot_df["relief_id"].isin(highlight_relief_ids)

    hover_cols = ["relief_id", "authority", "split"]
    if "n_views" in plot_df.columns:
        hover_cols.append("n_views")

    # Base layer: all points, legend only by authority
    fig = px.scatter(
        plot_df[~plot_df["is_highlighted"]],
        x="umap_x",
        y="umap_y",
        color="authority",
        hover_data=hover_cols,
        title=title,
        width=1000,
        height=700,
    )

    fig.update_traces(marker=dict(size=9, opacity=0.8))

    # Overlay highlighted points as separate traces, without adding legend clutter
    highlighted_df = plot_df[plot_df["is_highlighted"]]
    if not highlighted_df.empty:
        for _, row in highlighted_df.iterrows():
            customdata = []
            for col in hover_cols:
                customdata.append(row[col])

            hover_lines = [f"{col}=%{{customdata[{i}]}}" for i, col in enumerate(hover_cols)]
            hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

            fig.add_scatter(
                x=[row["umap_x"]],
                y=[row["umap_y"]],
                mode="markers+text",
                text=[row["relief_id"]],
                textposition="top center",
                customdata=[customdata],
                hovertemplate=hovertemplate,
                marker=dict(
                    size=16,
                    color="black",
                    symbol="diamond",
                    line=dict(color="yellow", width=2),
                ),
                name=f"highlight: {row['relief_id']}",
                showlegend=False,
            )

    return fig