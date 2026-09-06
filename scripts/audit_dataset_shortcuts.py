from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
from scipy.fft import dctn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASSES = ["Ashurbanipal", "Ashurnasirpal II", "Sargon II"]
IMAGE_FEATURES = [
    "log_width",
    "log_height",
    "log_file_bytes",
    "aspect_ratio",
    "luma_mean",
    "luma_std",
    "saturation_mean",
    "edge_mean",
    "border_luma_mean",
    "border_luma_std",
    "border_edge_mean",
    "center_luma_mean",
    "center_luma_std",
] + [f"border_hist_{i}" for i in range(8)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit perceptual duplicates and acquisition/source shortcuts."
    )
    parser.add_argument(
        "--csv-path",
        default="data/splits/image_level_dataset_v2_grouped.csv",
    )
    parser.add_argument("--image-root", default="/home/kostya/projects/dataset_v2")
    parser.add_argument("--out-dir", default="outputs/dataset_audit")
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--dhash-threshold", type=int, default=5)
    parser.add_argument("--contact-sheet-limit", type=int, default=24)
    return parser.parse_args()


def source_from_relief_id(relief_id: str) -> str:
    match = re.match(r"^([A-Za-z]+)", str(relief_id).strip())
    return match.group(1).upper() if match else "UNKNOWN"


def resolve_path(row: pd.Series, image_root: Path) -> Path:
    if "image_path" in row and pd.notna(row["image_path"]):
        path = Path(str(row["image_path"]))
        if path.is_file():
            return path
    suffix = str(row["suffix"])
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return image_root / f"{row['Relief_ID']}-{int(row['view_index'])}{suffix}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bool(bit))
    return value


def perceptual_hashes(image: Image.Image) -> tuple[int, int]:
    gray = ImageOps.grayscale(image)
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    low = dctn(small, type=2, norm="ortho")[:8, :8]
    phash = bits_to_int(low > np.median(low[1:, :]))

    dhash_img = np.asarray(
        gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=float
    )
    dhash = bits_to_int(dhash_img[:, 1:] > dhash_img[:, :-1])
    return phash, dhash


def image_features(path: Path) -> dict[str, float | int | str]:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        phash, dhash = perceptual_hashes(image)
        resized = image.resize((128, 128), Image.Resampling.LANCZOS)

    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    edge = np.asarray(
        Image.fromarray((luma * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES),
        dtype=np.float32,
    ) / 255.0

    border_mask = np.ones((128, 128), dtype=bool)
    border_mask[16:-16, 16:-16] = False
    center_mask = np.zeros((128, 128), dtype=bool)
    center_mask[32:-32, 32:-32] = True
    hist, _ = np.histogram(luma[border_mask], bins=8, range=(0.0, 1.0), density=True)
    hist = hist / max(float(hist.sum()), 1e-12)

    result: dict[str, float | int | str] = {
        "width": width,
        "height": height,
        "file_bytes": path.stat().st_size,
        "log_width": math.log1p(width),
        "log_height": math.log1p(height),
        "log_file_bytes": math.log1p(path.stat().st_size),
        "aspect_ratio": width / height,
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "saturation_mean": float(saturation.mean()),
        "edge_mean": float(edge.mean()),
        "border_luma_mean": float(luma[border_mask].mean()),
        "border_luma_std": float(luma[border_mask].std()),
        "border_edge_mean": float(edge[border_mask].mean()),
        "center_luma_mean": float(luma[center_mask].mean()),
        "center_luma_std": float(luma[center_mask].std()),
        "phash": f"{phash:016x}",
        "dhash": f"{dhash:016x}",
        "sha256": sha256(path),
    }
    result.update({f"border_hist_{i}": float(value) for i, value in enumerate(hist)})
    return result


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def resized_correlation(left: Path, right: Path) -> float:
    def array(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            gray = ImageOps.grayscale(image).resize((64, 64), Image.Resampling.LANCZOS)
        values = np.asarray(gray, dtype=np.float32).ravel()
        return (values - values.mean()) / max(float(values.std()), 1e-6)

    return float(np.mean(array(left) * array(right)))


def find_cross_split_candidates(
    data: pd.DataFrame,
    phash_threshold: int,
    dhash_threshold: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict] = []
    sensitivity = {f"phash_le_{threshold}": 0 for threshold in (2, 4, 6, 8, 10)}

    for left_idx in range(len(data)):
        left = data.iloc[left_idx]
        for right_idx in range(left_idx + 1, len(data)):
            right = data.iloc[right_idx]
            if left["split"] == right["split"]:
                continue
            p_dist = hamming(left["phash"], right["phash"])
            d_dist = hamming(left["dhash"], right["dhash"])
            for threshold in (2, 4, 6, 8, 10):
                if p_dist <= threshold:
                    sensitivity[f"phash_le_{threshold}"] += 1
            exact = left["sha256"] == right["sha256"]
            if not (exact or p_dist <= phash_threshold or d_dist <= dhash_threshold):
                continue
            correlation = resized_correlation(
                Path(left["resolved_path"]), Path(right["resolved_path"])
            )
            credible_automated_match = bool(
                exact
                or p_dist <= phash_threshold
                or (d_dist <= dhash_threshold and correlation >= 0.85)
            )
            rows.append(
                {
                    "left_filename": left["image_filename"],
                    "left_relief_id": left["Relief_ID"],
                    "left_authority": left["Authority"],
                    "left_source": left["source"],
                    "left_split": left["split"],
                    "left_path": left["resolved_path"],
                    "right_filename": right["image_filename"],
                    "right_relief_id": right["Relief_ID"],
                    "right_authority": right["Authority"],
                    "right_source": right["source"],
                    "right_split": right["split"],
                    "right_path": right["resolved_path"],
                    "exact_sha256": exact,
                    "phash_distance": p_dist,
                    "dhash_distance": d_dist,
                    "resized_correlation": correlation,
                    "credible_automated_match": credible_automated_match,
                }
            )

    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["exact_sha256", "phash_distance", "dhash_distance"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    return candidates, sensitivity


def metric_dict(y_true: pd.Series, y_pred: list[str] | np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }


def source_majority_baseline(data: pd.DataFrame) -> dict:
    train = data[data["split"] == "train"]
    global_majority = str(train["Authority"].mode().iat[0])
    lookup = (
        train.groupby("source")["Authority"]
        .agg(lambda values: str(values.value_counts().index[0]))
        .to_dict()
    )
    result = {"training_mapping": lookup, "fallback": global_majority}
    for split in ("val", "test"):
        subset = data[data["split"] == split]
        pred = [lookup.get(source, global_majority) for source in subset["source"]]
        result[split] = metric_dict(subset["Authority"], pred)
    return result


def feature_baseline(data: pd.DataFrame, feature_names: list[str]) -> dict:
    train = data[data["split"] == "train"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=24),
    )
    model.fit(train[feature_names], train["Authority"])
    result = {"features": feature_names}
    for split in ("val", "test"):
        subset = data[data["split"] == split]
        result[split] = metric_dict(
            subset["Authority"], model.predict(subset[feature_names])
        )
    return result


def render_contact_sheet(candidates: pd.DataFrame, out_path: Path, limit: int) -> None:
    if candidates.empty:
        return
    shown = candidates.head(limit)
    fig, axes = plt.subplots(len(shown), 2, figsize=(10, 3.8 * len(shown)))
    if len(shown) == 1:
        axes = np.asarray([axes])
    for row_idx, row in shown.iterrows():
        for col_idx, side in enumerate(("left", "right")):
            with Image.open(row[f"{side}_path"]) as image:
                axes[row_idx, col_idx].imshow(image.convert("RGB"))
            axes[row_idx, col_idx].set_title(
                f"{row[f'{side}_filename']}\n"
                f"{row[f'{side}_split']} | {row[f'{side}_authority']}"
            )
            axes[row_idx, col_idx].axis("off")
        axes[row_idx, 0].set_ylabel(
            f"p={row['phash_distance']}, d={row['dhash_distance']}\n"
            f"corr={row['resized_correlation']:.3f}"
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    data["source"] = data["Relief_ID"].map(source_from_relief_id)
    paths = [resolve_path(row, image_root) for _, row in data.iterrows()]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; examples: {missing[:5]}")
    data["resolved_path"] = [str(path) for path in paths]
    data["image_filename"] = [path.name for path in paths]

    print(f"Extracting audit features for {len(data)} images...")
    features = pd.DataFrame([image_features(path) for path in paths])
    enriched = pd.concat([data.reset_index(drop=True), features], axis=1)
    enriched.to_csv(out_dir / "image_audit_features.csv", index=False)

    candidates, sensitivity = find_cross_split_candidates(
        enriched,
        phash_threshold=args.phash_threshold,
        dhash_threshold=args.dhash_threshold,
    )
    candidates.to_csv(out_dir / "cross_split_near_duplicate_candidates.csv", index=False)
    render_contact_sheet(
        candidates,
        out_dir / "cross_split_near_duplicate_contact_sheet.png",
        args.contact_sheet_limit,
    )

    source_counts = pd.crosstab(enriched["source"], enriched["Authority"])
    source_counts.to_csv(out_dir / "source_by_authority_counts.csv")
    source_nmi = normalized_mutual_info_score(enriched["source"], enriched["Authority"])

    geometry_features = [
        "log_width",
        "log_height",
        "log_file_bytes",
        "aspect_ratio",
    ]
    border_features = [
        "border_luma_mean",
        "border_luma_std",
        "border_edge_mean",
    ] + [f"border_hist_{i}" for i in range(8)]
    results = {
        "manifest": str(csv_path),
        "n_images": int(len(enriched)),
        "class_order": CLASSES,
        "source_authority_normalized_mutual_information": float(source_nmi),
        "source_majority_baseline": source_majority_baseline(enriched),
        "geometry_baseline": feature_baseline(enriched, geometry_features),
        "border_baseline": feature_baseline(enriched, border_features),
        "combined_acquisition_baseline": feature_baseline(enriched, IMAGE_FEATURES),
        "near_duplicate_thresholds": {
            "phash": args.phash_threshold,
            "dhash": args.dhash_threshold,
        },
        "cross_split_candidate_count": int(len(candidates)),
        "cross_split_exact_duplicate_count": int(
            candidates["exact_sha256"].sum() if not candidates.empty else 0
        ),
        "cross_split_credible_automated_match_count": int(
            candidates["credible_automated_match"].sum()
            if not candidates.empty
            else 0
        ),
        "candidate_interpretation": (
            "dHash-only candidates with resized correlation below 0.85 are retained "
            "for visual review but are not counted as credible automated matches"
        ),
        "phash_sensitivity_counts": sensitivity,
    }
    (out_dir / "shortcut_audit_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    print(f"Saved audit outputs to {out_dir}")


if __name__ == "__main__":
    main()
