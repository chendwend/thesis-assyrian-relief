"""Render submission figures from saved coordinates/maps; never fit or infer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    figures = root.parent / "thesis-overleaf/figures"
    image_root = root.parent / "dataset_v2"
    report = root / "reports/stable_error_dossiers_2026-09-06"
    hashes = {}

    def record(path: Path) -> None:
        hashes[str(path.relative_to(root.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()

    coords = root / "outputs/grouped_seed42/eval/umap_dependency_groups.csv"
    record(coords)
    data = pd.read_csv(coords)
    assert len(data) == 171 and data.component_key.nunique() == 171
    data["source"] = data.relief_ids.map(lambda x: re.match(r"^([A-Za-z]+)", x).group(1))
    data["source"] = data.source.where(data.source.isin(["AO", "BM", "MET"]), "Other")
    authority = {"Ashurbanipal": "#3B6FB6", "Ashurnasirpal II": "#D17A22", "Sargon II": "#2F8F5B"}
    sources = {"BM": "#3B6FB6", "AO": "#C44E52", "MET": "#55A868", "Other": "#8172B2"}
    splits = {"train": "o", "val": "s", "test": "^"}
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9.5})
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 7.1), sharex=True, sharey=True, constrained_layout=True)
    for ax, col, colors, title in [(axes[0], "authority", authority, "A. Monarch attribution"), (axes[1], "source", sources, "B. Museum/source prefix")]:
        for split, marker in splits.items():
            for category, color in colors.items():
                rows = data[(data.split == split) & (data[col] == category)]
                ax.scatter(rows.umap_x, rows.umap_y, s=30 if split == "train" else 46, marker=marker, c=color, edgecolors="white", linewidths=.45, alpha=.78 if split == "train" else .95)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("UMAP 2")
        ax.grid(alpha=.15, linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
        handles = [plt.Line2D([], [], marker="o", linestyle="", color=color, label=label) for label, color in colors.items()]
        handles += [plt.Line2D([], [], marker=marker, linestyle="", color="#555555", label={"train":"Training", "val":"Validation", "test":"Test"}[split]) for split, marker in splits.items()]
        ax.legend(handles=handles, frameon=False, loc="center", ncol=2)
    axes[1].set_xlabel("UMAP 1")
    fig.savefig(figures / "umap_dependency_groups_seed42.png", dpi=300)
    plt.close(fig)

    for stem, positive, negative, output in [
        ("AO_19894-1", "Sargon II", "Ashurnasirpal II", "occlusion_AO19894_margin.png"),
        ("BM_124931-1", "Ashurnasirpal II", "Ashurbanipal", "occlusion_BM124931_margin.png"),
    ]:
        folder = root / f"outputs/grouped_seed42/explainability/occlusion/{stem}"
        map_path = folder / f"{stem}_margin_{positive.replace(' ', '_')}_vs_{negative.replace(' ', '_')}_heatmap.npy"
        meta_path = folder / f"{stem}_metadata.json"
        heatmap = np.load(map_path)
        assert heatmap.shape == (224, 224) and np.isfinite(heatmap).all()
        meta = json.loads(meta_path.read_text())
        photo = image_root / Path(meta["image_path"]).name
        for path in [map_path, meta_path, photo]:
            record(path)
        # Same display resize and 98th-percentile colour limits as explain_occlusion.py.
        source = Image.open(photo).convert("RGB").resize((224, 224))
        vmax = np.percentile(np.abs(heatmap), 98)
        norm = TwoSlopeNorm(vmin=-vmax if vmax > 1e-8 else -1, vcenter=0, vmax=vmax if vmax > 1e-8 else 1)
        fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.95), constrained_layout=True)
        axes[0].imshow(source)
        axes[0].set_title("Input (224 × 224)", fontsize=9.5)
        hm = axes[1].imshow(heatmap, cmap="coolwarm", norm=norm)
        axes[1].set_title("Occlusion effect", fontsize=9.5)
        cb = fig.colorbar(hm, ax=axes[1], fraction=.05, pad=.025)
        cb.ax.tick_params(labelsize=8)
        axes[2].imshow(source)
        axes[2].imshow(heatmap, cmap="coolwarm", norm=norm, alpha=.4)
        axes[2].set_title("Overlay", fontsize=9.5)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{positive} − {negative}\nPrediction: {meta['pred_class']} ({meta['pred_prob']:.3f}); label: {meta['true_class']}", fontsize=9.5)
        fig.savefig(figures / output, dpi=300)
        plt.close(fig)

    ranks = root / "reports/submission_controls_2026-09-06/retrieval_all_ranks.csv"
    record(ranks)
    with ranks.open() as f:
        rows = list(csv.DictReader(f))
    font_path = next(p for p in [Path("C:/Windows/Fonts/arial.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")] if p.exists())
    font = ImageFont.truetype(str(font_path), 32)
    sheet = Image.new("RGB", (1800, 1880), "white")
    draw = ImageDraw.Draw(sheet)
    for row, query in enumerate(["AO 19902", "BM 124867", "BM 124931", "BM 118829"]):
        cells = [(f"Query: {query}", query)]
        for seed in [24, 42, 77]:
            hits = [r for r in rows if r["query_relief_ids"] == query and r["seed"] == str(seed) and r["rank"] == "1"]
            assert len(hits) == 1
            hit = hits[0]
            cells.append((f"Seed {seed}: {hit['neighbour_relief_ids']}\n{hit['neighbour_authority']}  {float(hit['cosine_similarity']):.3f}", hit["neighbour_relief_ids"]))
        for col, (label, accession) in enumerate(cells):
            photo = sorted(image_root.glob(accession + "-*.jpg"))[0]
            record(photo)
            image = Image.open(photo).convert("RGB")
            image.thumbnail((430, 385))
            x, y = col*450, row*470
            sheet.paste(image, (x+(430-image.width)//2, y+78+(385-image.height)//2))
            draw.multiline_text((x+5,y+5), label, font=font, fill="black", spacing=1)
    sheet.save(figures / "stable_error_neighbours.jpg", quality=93)
    sheet.save(report / "nearest_neighbours.jpg", quality=93)
    (report / "figure_input_hashes.json").write_text(json.dumps(hashes, indent=2)+"\n", encoding="utf-8")
    print("Rendered four figures from unchanged saved coordinates, heatmaps, ranks and photographs.")


if __name__ == "__main__":
    main()
