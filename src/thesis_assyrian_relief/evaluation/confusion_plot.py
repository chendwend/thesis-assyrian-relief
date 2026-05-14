from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_confusion_matrix_plot(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    out_path: str | Path,
    *,
    class_names: Sequence[str],
    title: str | None = None,
) -> None:
    """
    Plot and save a confusion matrix (counts). Rows are true class, columns are predicted.

    ``class_names`` must be ordered by class index (0 .. n-1).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = np.arange(len(class_names), dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    n = len(class_names)
    fig_w = min(22, max(6.0, 0.45 * n))
    fig_h = min(20, max(5.0, 0.4 * n))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d", xticks_rotation=45, im_kw={"vmin": 0})

    if title:
        ax.set_title(title)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
