from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from thesis_assyrian_relief.models.dinov2_probe import DinoStyleProbe
from thesis_assyrian_relief.training.engine import load_checkpoint
from thesis_assyrian_relief.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate class-specific occlusion sensitivity heatmaps."
    )

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=None)

    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--true-class", type=str, default=None)

    parser.add_argument(
        "--target-classes",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Class names to explain. If omitted, explains predicted class "
            "and true class if --true-class is provided."
        ),
    )

    parser.add_argument(
        "--margin-pairs",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Explicit margin pairs to explain, formatted as "
            "'Positive Class::Negative Class'. "
            "Example: 'Ashurbanipal::Ashurnasirpal II'. "
            "If omitted, an automatic margin pair is selected."
        ),
    )

    parser.add_argument("--out-dir", type=str, default=None)

    parser.add_argument(
        "--occlusion-value",
        type=float,
        default=None,
        help=(
            "Value used in normalized tensor space. If omitted, taken from config. "
            "0.0 corresponds approximately to ImageNet mean color after normalization."
        ),
    )

    parser.add_argument(
        "--clamp-negative",
        default=None,
        type=str,
        choices=["true", "false"],
        help="Override config explainability.occlusion.clamp_negative.",
    )


    parser.add_argument("--relief-id", type=str, default=None)
    parser.add_argument("--view-index", type=int, default=None)
    parser.add_argument("--suffix", type=str, default=None)

    parser.add_argument("--occlusion-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)

    return parser.parse_args()


def safe_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = text.strip("_")
    return text


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

def get_diverging_norm(heatmap: np.ndarray):
    abs_max = np.percentile(np.abs(heatmap), 98)
    if abs_max <= 1e-8:
        abs_max = 1.0

    return TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, int], dict[int, str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "class_to_idx" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'class_to_idx'.")

    class_to_idx: dict[str, int] = checkpoint["class_to_idx"]
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    model_name = checkpoint.get("model_name", "dinov2_vits14")
    emb_dim = checkpoint.get("emb_dim", 256)
    num_classes = len(class_to_idx)

    model = DinoStyleProbe(
        num_classes=num_classes,
        emb_dim=emb_dim,
        model_name=model_name,
    ).to(device)

    load_checkpoint(
        model=model,
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    model.eval()

    return model, class_to_idx, idx_to_class


@torch.no_grad()
def predict_single_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    idx_to_class: dict[int, str],
) -> dict:
    logits, emb = model(image_tensor)

    probs = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())

    result = {
        "pred_idx": pred_idx,
        "pred_class": idx_to_class[pred_idx],
        "logits": logits[0].detach().cpu().numpy(),
        "probs": probs[0].detach().cpu().numpy(),
    }

    return result


def heatmap_stats(heatmap: np.ndarray) -> dict:
    hm = heatmap.astype(np.float32)
    return {
        "min": float(np.min(hm)),
        "max": float(np.max(hm)),
        "mean": float(np.mean(hm)),
        "std": float(np.std(hm)),
        "p01": float(np.percentile(hm, 1)),
        "p05": float(np.percentile(hm, 5)),
        "p50": float(np.percentile(hm, 50)),
        "p95": float(np.percentile(hm, 95)),
        "p99": float(np.percentile(hm, 99)),
        "num_positive": int(np.sum(hm > 0)),
        "num_negative": int(np.sum(hm < 0)),
    }


@torch.no_grad()
def occlusion_sensitivity(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_idx: int,
    occlusion_size: int = 32,
    stride: int = 16,
    occlusion_value: float = 0.0,
    clamp_negative: bool = False,
) -> np.ndarray:
    """
    Compute class-specific occlusion sensitivity.

    image_tensor shape:
        [1, C, H, W]

    Importance:
        original_target_logit - occluded_target_logit

    Positive value means:
        occluding this region reduced the target class logit,
        so this region supported the target class.
    """

    if image_tensor.ndim != 4 or image_tensor.size(0) != 1:
        raise ValueError(
            f"Expected image_tensor with shape [1, C, H, W], got {tuple(image_tensor.shape)}"
        )

    _, _, h, w = image_tensor.shape

    original_logits, _ = model(image_tensor)
    original_score = original_logits[0, target_idx].item()

    heatmap_sum = np.zeros((h, w), dtype=np.float32)
    heatmap_count = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride):
        y1 = y
        y2 = min(y + occlusion_size, h)

        for x in range(0, w, stride):
            x1 = x
            x2 = min(x + occlusion_size, w)

            occluded = image_tensor.clone()
            occluded[:, :, y1:y2, x1:x2] = occlusion_value

            occluded_logits, _ = model(occluded)
            occluded_score = occluded_logits[0, target_idx].item()

            importance = original_score - occluded_score

            if clamp_negative:
                importance = max(importance, 0.0)

            heatmap_sum[y1:y2, x1:x2] += importance
            heatmap_count[y1:y2, x1:x2] += 1.0

    heatmap = heatmap_sum / np.maximum(heatmap_count, 1.0)
    return heatmap

@torch.no_grad()
def margin_occlusion_sensitivity(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    positive_idx: int,
    negative_idx: int,
    occlusion_size: int = 32,
    stride: int = 16,
    occlusion_value: float = 0.0,
) -> np.ndarray:
    """
    Compute pairwise margin occlusion sensitivity.

    Margin:
        logit[positive_idx] - logit[negative_idx]

    Importance:
        original_margin - occluded_margin

    Positive value means:
        occluding this region reduced the margin, so this region supports
        the positive class over the negative class.

    Negative value means:
        occluding this region increased the margin, so this region opposes
        the positive-vs-negative decision.
    """

    if image_tensor.ndim != 4 or image_tensor.size(0) != 1:
        raise ValueError(
            f"Expected image_tensor with shape [1, C, H, W], got {tuple(image_tensor.shape)}"
        )

    _, _, h, w = image_tensor.shape

    original_logits, _ = model(image_tensor)
    original_margin = (
        original_logits[0, positive_idx] - original_logits[0, negative_idx]
    ).item()

    heatmap_sum = np.zeros((h, w), dtype=np.float32)
    heatmap_count = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride):
        y1 = y
        y2 = min(y + occlusion_size, h)

        for x in range(0, w, stride):
            x1 = x
            x2 = min(x + occlusion_size, w)

            occluded = image_tensor.clone()
            occluded[:, :, y1:y2, x1:x2] = occlusion_value

            occluded_logits, _ = model(occluded)
            occluded_margin = (
                occluded_logits[0, positive_idx] - occluded_logits[0, negative_idx]
            ).item()

            importance = original_margin - occluded_margin

            heatmap_sum[y1:y2, x1:x2] += importance
            heatmap_count[y1:y2, x1:x2] += 1.0

    heatmap = heatmap_sum / np.maximum(heatmap_count, 1.0)
    return heatmap

def normalize_positive_support(
    heatmap: np.ndarray,
    support_max: float | None = None,
) -> np.ndarray:
    hm = np.maximum(heatmap.astype(np.float32), 0.0)

    if support_max is None:
        support_max = float(np.percentile(hm, 98))

    if support_max <= 1e-8:
        return np.zeros_like(hm, dtype=np.float32)

    return np.clip(hm / support_max, 0.0, 1.0)


def save_evidence_figure(
    resized_pil: Image.Image,
    heatmap: np.ndarray,
    target_class: str,
    pred_class: str,
    pred_prob: float,
    true_class: str | None,
    out_path: Path,
) -> None:
    """
    Save zero-centered evidence map.

    Warm colors: occluding this region decreases target logit,
                 so region supports target class.
    Cool colors: occluding this region increases target logit,
                 so region opposes target class.
    """

    norm = get_diverging_norm(heatmap)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        constrained_layout=True,
    )

    axes[0].imshow(resized_pil)
    axes[0].set_title("Input image\n224×224 model input")
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="coolwarm", norm=norm)
    axes[1].set_title(
        f"Evidence map\nTarget: {target_class}\n"
        "warm=supports, cool=opposes"
    )
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(resized_pil)
    axes[2].imshow(heatmap, cmap="coolwarm", norm=norm, alpha=0.38)
    axes[2].set_title("Evidence overlay")
    axes[2].axis("off")

    title = (
        f"Prediction: {pred_class} ({pred_prob:.3f}) | "
        f"Explained class: {target_class}"
    )
    if true_class is not None:
        title += f" | True: {true_class}"

    fig.suptitle(title, fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_margin_figure(
    resized_pil: Image.Image,
    margin_heatmap: np.ndarray,
    positive_class: str,
    negative_class: str,
    pred_class: str,
    pred_prob: float,
    true_class: str | None,
    out_path: Path,
) -> None:
    """
    Save pairwise decision-margin evidence map.

    Warm colors:
        regions supporting positive_class over negative_class.

    Cool colors:
        regions supporting negative_class over positive_class,
        or opposing the positive-vs-negative decision.
    """

    norm = get_diverging_norm(margin_heatmap)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        constrained_layout=True,
    )

    axes[0].imshow(resized_pil)
    axes[0].set_title("Input image\n224×224 model input")
    axes[0].axis("off")

    im = axes[1].imshow(margin_heatmap, cmap="coolwarm", norm=norm)
    axes[1].set_title(
        "Decision-margin map\n"
        f"{positive_class} vs {negative_class}\n"
        "warm=supports first class"
    )
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(resized_pil)
    axes[2].imshow(margin_heatmap, cmap="coolwarm", norm=norm, alpha=0.40)
    axes[2].set_title("Margin overlay")
    axes[2].axis("off")

    title = (
        f"Prediction: {pred_class} ({pred_prob:.3f}) | "
        f"Margin: {positive_class} - {negative_class}"
    )
    if true_class is not None:
        title += f" | True: {true_class}"

    fig.suptitle(title, fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_support_figure(
    resized_pil: Image.Image,
    heatmap: np.ndarray,
    target_class: str,
    pred_class: str,
    pred_prob: float,
    true_class: str | None,
    out_path: Path,
    support_max: float | None = None,
) -> None:
    """
    Save positive-only support map.

    This hides negative evidence and shows only regions whose occlusion
    reduced the target-class logit.
    """

    support = normalize_positive_support(
        heatmap=heatmap,
        support_max=support_max,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
        constrained_layout=True,
    )

    axes[0].imshow(resized_pil)
    axes[0].set_title("Input image\n224×224 model input")
    axes[0].axis("off")

    im = axes[1].imshow(support, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Positive support map\nTarget: {target_class}")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(resized_pil)
    axes[2].imshow(support, cmap="magma", vmin=0.0, vmax=1.0, alpha=0.42)
    axes[2].set_title("Support overlay")
    axes[2].axis("off")

    title = (
        f"Prediction: {pred_class} ({pred_prob:.3f}) | "
        f"Explained class: {target_class}"
    )
    if true_class is not None:
        title += f" | True: {true_class}"

    fig.suptitle(title, fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_combined_support_panel(
    resized_pil: Image.Image,
    heatmaps_by_class: dict[str, np.ndarray],
    pred_class: str,
    pred_prob: float,
    true_class: str | None,
    out_path: Path,
) -> None:
    """
    Save one panel comparing positive support across target classes
    using a shared scale for this image.
    """

    positive_values = []
    for hm in heatmaps_by_class.values():
        positive_values.append(np.maximum(hm.astype(np.float32), 0.0).ravel())

    if positive_values:
        all_positive = np.concatenate(positive_values)
        support_max = float(np.percentile(all_positive, 98))
    else:
        support_max = 1.0

    if support_max <= 1e-8:
        support_max = 1.0

    target_classes = list(heatmaps_by_class.keys())

    ncols = 1 + len(target_classes)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5 * ncols, 5),
        constrained_layout=True,
    )

    if ncols == 1:
        axes = [axes]

    axes[0].imshow(resized_pil)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    last_im = None

    for ax, target_class in zip(axes[1:], target_classes):
        support = normalize_positive_support(
            heatmap=heatmaps_by_class[target_class],
            support_max=support_max,
        )

        ax.imshow(resized_pil)
        last_im = ax.imshow(
            support,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            alpha=0.42,
        )
        ax.set_title(f"Support for\n{target_class}")
        ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes[1:], fraction=0.025, pad=0.02)

    title = f"Prediction: {pred_class} ({pred_prob:.3f})"
    if true_class is not None:
        title += f" | True: {true_class}"
    title += " | Shared support scale"

    fig.suptitle(title, fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def normalize_suffix(suffix: str) -> str:
    suffix = str(suffix).strip()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return suffix



def get_nested(cfg: dict, path: list[str], default=None):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def pick(cli_value, cfg: dict, path: list[str], default=None):
    if cli_value is not None:
        return cli_value
    value = get_nested(cfg, path, default=default)
    return value


def parse_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value!r}")


def resolve_image_path(args: argparse.Namespace, cfg: dict) -> Path:
    if args.image_path is not None:
        path = Path(args.image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        return path

    if args.relief_id is None or args.view_index is None:
        raise ValueError(
            "Provide either --image-path or both --relief-id and --view-index."
        )

    suffix = args.suffix
    if suffix is None:
        suffix = ".jpg"

    suffix = normalize_suffix(suffix)

    image_root = Path(cfg["data"]["image_root"])
    filename_sep = cfg["data"].get("filename_sep", "-")
    filename = f"{args.relief_id}{filename_sep}{args.view_index}{suffix}"
    path = image_root / filename

    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    return path


def resolve_occlusion_config(args: argparse.Namespace, cfg: dict) -> dict:
    occ_cfg = get_nested(cfg, ["explainability", "occlusion"], default={}) or {}

    clamp_negative = parse_bool(args.clamp_negative)
    if clamp_negative is None:
        clamp_negative = bool(occ_cfg.get("clamp_negative", False))

    return {
        "occlusion_size": int(
            args.occlusion_size
            if args.occlusion_size is not None
            else occ_cfg.get("occlusion_size", 32)
        ),
        "stride": int(
            args.stride
            if args.stride is not None
            else occ_cfg.get("stride", 16)
        ),
        "occlusion_value": float(
            args.occlusion_value
            if args.occlusion_value is not None
            else occ_cfg.get("occlusion_value", 0.0)
        ),
        "clamp_negative": clamp_negative,
    }


def resolve_output_dir(args: argparse.Namespace, cfg: dict, image_path: Path) -> Path:
    if args.out_dir is not None:
        root = Path(args.out_dir)
    else:
        outputs_cfg = cfg.get("outputs", {})
        if "explainability_root" not in outputs_cfg:
            raise ValueError(
                "Missing outputs.explainability_root in config. "
                "Add it to the runtime config or pass --out-dir."
            )
        root = Path(outputs_cfg["explainability_root"])

    image_dir = root / "occlusion" / safe_name(image_path.stem)
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir

def parse_margin_pairs(
    raw_pairs: list[str] | None,
    class_to_idx: dict[str, int],
) -> list[tuple[str, str]]:
    if not raw_pairs:
        return []

    parsed_pairs: list[tuple[str, str]] = []

    for raw in raw_pairs:
        if "::" not in raw:
            raise ValueError(
                f"Invalid margin pair {raw!r}. "
                "Expected format: 'Positive Class::Negative Class'."
            )

        positive_class, negative_class = raw.split("::", maxsplit=1)
        positive_class = positive_class.strip()
        negative_class = negative_class.strip()

        if positive_class not in class_to_idx:
            raise ValueError(
                f"Unknown positive class in margin pair: {positive_class!r}. "
                f"Known classes: {list(class_to_idx.keys())}"
            )

        if negative_class not in class_to_idx:
            raise ValueError(
                f"Unknown negative class in margin pair: {negative_class!r}. "
                f"Known classes: {list(class_to_idx.keys())}"
            )

        if positive_class == negative_class:
            raise ValueError(
                f"Invalid margin pair {raw!r}: positive and negative classes are identical."
            )

        parsed_pairs.append((positive_class, negative_class))

    return parsed_pairs




def main() -> None:
    args = parse_args()

    cfg = load_yaml_config(args.config)

    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path is not None
        else cfg["outputs"]["checkpoint_path"]
    )

    image_path = resolve_image_path(args, cfg)
    out_dir = resolve_output_dir(args, cfg, image_path)
    occ = resolve_occlusion_config(args, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Image:", image_path)
    print("Output directory:", out_dir)
    print("Occlusion config:", occ)

    model, class_to_idx, idx_to_class = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    transform = build_eval_transform()

    original_pil = Image.open(image_path).convert("RGB")
    resized_pil = original_pil.resize((224, 224))

    image_tensor = transform(original_pil).unsqueeze(0).to(device)

    prediction = predict_single_image(
        model=model,
        image_tensor=image_tensor,
        idx_to_class=idx_to_class,
    )

    pred_class = prediction["pred_class"]
    pred_idx = prediction["pred_idx"]
    pred_prob = float(prediction["probs"][pred_idx])

    print("Image:", image_path)
    print("Predicted:", pred_class, f"({pred_prob:.4f})")

    for idx in sorted(idx_to_class):
        cls = idx_to_class[idx]
        prob = float(prediction["probs"][idx])
        logit = float(prediction["logits"][idx])
        print(f"  {cls}: prob={prob:.4f}, logit={logit:.4f}")

    if args.target_classes:
        target_classes = args.target_classes
    else:
        target_classes = [pred_class]
        if args.true_class is not None and args.true_class != pred_class:
            target_classes.append(args.true_class)

    unknown_targets = [c for c in target_classes if c not in class_to_idx]
    if unknown_targets:
        raise ValueError(
            f"Unknown target class/classes: {unknown_targets}. "
            f"Known classes: {list(class_to_idx.keys())}"
        )

    metadata = {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "true_class": args.true_class,
        "pred_class": pred_class,
        "pred_idx": pred_idx,
        "pred_prob": pred_prob,
        "class_to_idx": class_to_idx,
        "probs": {
            idx_to_class[idx]: float(prediction["probs"][idx])
            for idx in sorted(idx_to_class)
        },
        "logits": {
            idx_to_class[idx]: float(prediction["logits"][idx])
            for idx in sorted(idx_to_class)
        },
        "occlusion_size": occ["occlusion_size"],
        "stride": occ["stride"],
        "occlusion_value": occ["occlusion_value"],
        "clamp_negative": occ["clamp_negative"],
        "target_classes": target_classes,
        "requested_margin_pairs": args.margin_pairs,
    }
    metadata.setdefault("heatmap_stats", {})

    image_stem = safe_name(image_path.stem)

    heatmaps_by_class: dict[str, np.ndarray] = {}

    for target_class in target_classes:
        target_idx = class_to_idx[target_class]
        target_safe = safe_name(target_class)

        print(f"\nComputing occlusion heatmap for: {target_class}")

        heatmap = occlusion_sensitivity(
            model=model,
            image_tensor=image_tensor,
            target_idx=target_idx,
            occlusion_size=occ["occlusion_size"],
            stride=occ["stride"],
            occlusion_value=occ["occlusion_value"],
            clamp_negative=occ["clamp_negative"],
        )

        heatmaps_by_class[target_class] = heatmap
        metadata["heatmap_stats"][target_class] = heatmap_stats(heatmap)

        npy_path = out_dir / f"{image_stem}_target_{target_safe}_heatmap.npy"
        evidence_path = out_dir / f"{image_stem}_target_{target_safe}_evidence.png"
        support_path = out_dir / f"{image_stem}_target_{target_safe}_support.png"

        np.save(npy_path, heatmap)

        save_evidence_figure(
            resized_pil=resized_pil,
            heatmap=heatmap,
            target_class=target_class,
            pred_class=pred_class,
            pred_prob=pred_prob,
            true_class=args.true_class,
            out_path=evidence_path,
        )

        save_support_figure(
            resized_pil=resized_pil,
            heatmap=heatmap,
            target_class=target_class,
            pred_class=pred_class,
            pred_prob=pred_prob,
            true_class=args.true_class,
            out_path=support_path,
        )

        print(f"Saved heatmap array: {npy_path}")
        print(f"Saved evidence figure: {evidence_path}")
        print(f"Saved support figure: {support_path}")


    combined_support_path = out_dir / f"{image_stem}_combined_support.png"

    save_combined_support_panel(
        resized_pil=resized_pil,
        heatmaps_by_class=heatmaps_by_class,
        pred_class=pred_class,
        pred_prob=pred_prob,
        true_class=args.true_class,
        out_path=combined_support_path,
    )

    print(f"Saved combined support panel: {combined_support_path}")


    # Pairwise decision-margin explanation.
    # For misclassified examples, the most important margin is:
    # predicted class vs true class.
    metadata.setdefault("margin_heatmap_stats", {})

    margin_pairs: list[tuple[str, str]] = []

    # 1. Explicit user-requested margin pairs.
    margin_pairs.extend(
        parse_margin_pairs(
            raw_pairs=args.margin_pairs,
            class_to_idx=class_to_idx,
        )
    )

    # 2. Automatic fallback pair, only if user did not request explicit pairs.
    if not margin_pairs:
        if args.true_class is not None and args.true_class in class_to_idx:
            if pred_class != args.true_class:
                # Misclassified case: explain predicted class over true class.
                margin_pairs.append((pred_class, args.true_class))
            else:
                # Correct case: compare predicted/true class against strongest alternative.
                probs = prediction["probs"]
                alternative_indices = [
                    idx for idx in sorted(idx_to_class)
                    if idx_to_class[idx] != pred_class
                ]
                if alternative_indices:
                    best_alt_idx = max(
                        alternative_indices,
                        key=lambda idx: float(probs[idx]),
                    )
                    margin_pairs.append((pred_class, idx_to_class[best_alt_idx]))

    # Deduplicate while preserving order.
    seen_pairs = set()
    unique_margin_pairs: list[tuple[str, str]] = []
    for pair in margin_pairs:
        if pair not in seen_pairs:
            unique_margin_pairs.append(pair)
            seen_pairs.add(pair)


    metadata["resolved_margin_pairs"] = [
        {
            "positive_class": positive_class,
            "negative_class": negative_class,
        }
        for positive_class, negative_class in unique_margin_pairs
    ]

    for positive_class, negative_class in unique_margin_pairs:
        positive_idx = class_to_idx[positive_class]
        negative_idx = class_to_idx[negative_class]

        positive_safe = safe_name(positive_class)
        negative_safe = safe_name(negative_class)

        print(
            f"\nComputing margin occlusion heatmap: "
            f"{positive_class} vs {negative_class}"
        )

        margin_heatmap = margin_occlusion_sensitivity(
            model=model,
            image_tensor=image_tensor,
            positive_idx=positive_idx,
            negative_idx=negative_idx,
            occlusion_size=occ["occlusion_size"],
            stride=occ["stride"],
            occlusion_value=occ["occlusion_value"],
        )

        margin_key = f"{positive_class}__minus__{negative_class}"
        metadata["margin_heatmap_stats"][margin_key] = heatmap_stats(margin_heatmap)

        margin_npy_path = (
            out_dir
            / f"{image_stem}_margin_{positive_safe}_vs_{negative_safe}_heatmap.npy"
        )
        margin_png_path = (
            out_dir
            / f"{image_stem}_margin_{positive_safe}_vs_{negative_safe}.png"
        )

        np.save(margin_npy_path, margin_heatmap)

        save_margin_figure(
            resized_pil=resized_pil,
            margin_heatmap=margin_heatmap,
            positive_class=positive_class,
            negative_class=negative_class,
            pred_class=pred_class,
            pred_prob=pred_prob,
            true_class=args.true_class,
            out_path=margin_png_path,
        )

        print(f"Saved margin heatmap array: {margin_npy_path}")
        print(f"Saved margin figure: {margin_png_path}")


    metadata_path = out_dir / f"{image_stem}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata: {metadata_path}")


if __name__ == "__main__":
    main()