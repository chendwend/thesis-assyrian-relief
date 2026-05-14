from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()

    running_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        logits, emb = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    metrics = compute_classification_metrics(all_targets, all_preds)
    metrics["loss"] = running_loss / len(loader.dataset)

    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    running_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits, emb = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    metrics = compute_classification_metrics(all_targets, all_preds)
    metrics["loss"] = running_loss / len(loader.dataset)

    return metrics


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    scheduler: Any | None = None,
    checkpoint_path: str | None = None,
    extra_checkpoint_data: dict[str, Any] | None = None,
    monitor_metric: str = "val_macro_f1",
    monitor_mode: str = "max",
    early_stopping_enabled: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0,
) -> tuple[list[dict[str, float]], int | None]:

    if monitor_mode not in {"max", "min"}:
        raise ValueError(f"monitor_mode must be 'max' or 'min', got {monitor_mode!r}")

    best_score = float("-inf") if monitor_mode == "max" else float("inf")
    best_epoch: int | None = None
    history: list[dict[str, float]] = []

    early_stopping_best_score = best_score
    early_stopping_counter = 0

    if early_stopping_enabled:
        print(
            "early stopping: enabled "
            f"(monitor={monitor_metric}, mode={monitor_mode}, "
            f"patience={early_stopping_patience}, "
            f"min_delta={early_stopping_min_delta})"
        )
    else:
        print("early stopping: disabled")


    def is_better(
        current: float,
        best: float,
        mode: str,
        min_delta: float = 0.0,
    ) -> bool:
        if mode == "max":
            return current > best + min_delta

        if mode == "min":
            return current < best - min_delta

        raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            # supcon_criterion=supcon_criterion,
            # lambda_ce = LAMBDA_CE,
            # lambda_supcon = LAMBDA_SUPCON,
            # ce_criterion = CE_CRITERION,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion, # ce_criterion = ce_criterion
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            # train_ce_loss = train_metrics["ce_loss"],
            # train_supcon_loss = train_metrics["supcon_loss"],
            "train_acc": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"lr={row['lr']:.2e} | "
            f"train_loss={row['train_loss']:.4f} "
            #  f"(ce={row['train_ce_loss']:.4f}, supcon={row['train_supcon_loss']:.4f}) "
            f"train_acc={row['train_acc']:.4f} "
            f"train_f1={row['train_macro_f1']:.4f} | "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f} "
            f"val_f1={row['val_macro_f1']:.4f}"
        )

        if monitor_metric not in row:
            raise KeyError(
                f"monitor_metric={monitor_metric!r} not found in history row. "
                f"Available metrics: {list(row.keys())}"
            )

        monitor_value = row[monitor_metric]

        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(monitor_value)
            else:
                scheduler.step()

        if is_better(
            current=monitor_value,
            best=best_score,
            mode=monitor_mode,
            min_delta=0.0,
        ):
            best_score = monitor_value
            best_epoch = epoch

            if checkpoint_path is not None:
                payload: dict[str, Any] = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best_score,
                    "monitor_metric": monitor_metric,
                    "monitor_mode": monitor_mode,
                    "early_stopping_enabled": early_stopping_enabled,
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                    "history": history,
                }

                if extra_checkpoint_data is not None:
                    payload.update(extra_checkpoint_data)

                torch.save(payload, checkpoint_path)
                print(
                    f"  Saved new best model to {checkpoint_path} "
                    f"({monitor_metric}={monitor_value:.4f})"
                )

        if early_stopping_enabled:
            if is_better(
                current=monitor_value,
                best=early_stopping_best_score,
                mode=monitor_mode,
                min_delta=early_stopping_min_delta,
            ):
                early_stopping_best_score = monitor_value
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(
                    f"  Early stopping counter: "
                    f"{early_stopping_counter}/{early_stopping_patience}"
                )

            if early_stopping_counter >= early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch: {best_epoch}, "
                    f"best {monitor_metric}: {best_score:.4f}"
                )
                break

    return history, best_epoch


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint