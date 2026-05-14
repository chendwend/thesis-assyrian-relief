from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import umap


def build_umap_dataframe(
    fit_relief_emb_df: pd.DataFrame,
    project_relief_emb_dfs: list[tuple[pd.DataFrame, str]],
    n_neighbors: int = 10,
    min_dist: float = 0.2,
    metric: str = "cosine",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit UMAP on one relief-level embedding set, then project any number of splits into the same space.

    Each relief dataframe is expected to contain:
        - relief_id
        - authority
        - embedding
        - n_views (optional)

    Parameters
    ----------
    fit_relief_emb_df:
        Embeddings used only to fit the UMAP model (not necessarily included in the output).
    project_relief_emb_dfs:
        List of (relief_emb_df, split_name) pairs to transform with the fitted model and concatenate.
    """
    required_emb_cols = {"relief_id", "authority", "embedding"}
    if fit_relief_emb_df.empty:
        raise ValueError("fit_relief_emb_df is empty; cannot fit UMAP.")

    missing_fit = required_emb_cols - set(fit_relief_emb_df.columns)
    if missing_fit:
        raise ValueError(f"fit_relief_emb_df missing columns: {sorted(missing_fit)}")

    if not project_relief_emb_dfs:
        raise ValueError("project_relief_emb_dfs is empty; nothing to project or plot.")

    X_fit = np.stack(fit_relief_emb_df["embedding"].to_list(), axis=0)
    n_fit = X_fit.shape[0]
    n_neighbors_eff = min(n_neighbors, max(2, n_fit - 1))

    reducer = umap.UMAP(
        n_neighbors=n_neighbors_eff,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    reducer.fit(X_fit)

    frames: list[pd.DataFrame] = []
    for df, split_name in project_relief_emb_dfs:
        if df.empty:
            continue
        missing = required_emb_cols - set(df.columns)
        if missing:
            raise ValueError(f"Relief dataframe for split {split_name!r} missing columns: {sorted(missing)}")
        X = np.stack(df["embedding"].to_list(), axis=0)
        X_2d = reducer.transform(X)
        cur = df.copy()
        cur["split"] = split_name
        cur["umap_x"] = X_2d[:, 0]
        cur["umap_y"] = X_2d[:, 1]
        frames.append(cur)

    if not frames:
        raise ValueError("All projected split dataframes were empty.")

    viz_df = pd.concat(frames, ignore_index=True)
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


SPLIT_MARKER_SYMBOL: dict[str, str] = {
    "train": "circle",
    "val": "square",
    "test": "triangle-up",
}


def _marker_symbol_for_split(split: str) -> str:
    return SPLIT_MARKER_SYMBOL.get(split, "circle")


def plot_umap_plotly(
    viz_df: pd.DataFrame,
    title: str = "Interactive UMAP of Relief-Level Embeddings",
    highlight_relief_ids: set[str] | None = None,
):
    """
    Interactive Plotly UMAP plot.

    - Color encodes authority
    - Marker shape encodes data split (train=circle, val=square, test=triangle-up; other splits default to circle)
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

    base_df = plot_df[~plot_df["is_highlighted"]]
    present_splits = list(dict.fromkeys(base_df["split"].astype(str).tolist()))
    symbol_map = {s: _marker_symbol_for_split(s) for s in present_splits}

    fig = px.scatter(
        base_df,
        x="umap_x",
        y="umap_y",
        color="authority",
        color_discrete_map={"Ashurbanipal": "red", "Ashurnarsipal II": "blue", "Sargon II": "green", "Sennacherib": "Purple", "Tiglath-Pileser III": "Yellow"},
        symbol="split",
        symbol_map=symbol_map,
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