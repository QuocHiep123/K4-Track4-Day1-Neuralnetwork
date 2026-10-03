"""plots.py — Vẽ biểu đồ thí nghiệm và so sánh nhóm.

Mỗi thí nghiệm một ảnh figures/<exp_id>.png gồm ít nhất 3 ô:
  (1) train_loss và val_loss theo epoch
  (2) val_acc và val_macro_f1 theo epoch
  (3) grad_norm theo epoch (đo TRƯỚC khi clip)
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô."""
    cfg = result["cfg"]
    hist = result["history"]
    summary = result.get("summary", {})
    exp_id = cfg.get("exp_id", "exp")

    epochs = hist["epoch"]
    if not epochs:
        return

    best_epoch = summary.get("best_epoch", 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Loss
    axes[0].plot(epochs, hist["train_loss"], label="Train Loss", marker="o", markersize=3, color="#1f77b4")
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss", marker="s", markersize=3, color="#ff7f0e")
    axes[0].axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep {best_epoch}")
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # 2. Accuracy & Macro-F1
    axes[1].plot(epochs, hist["val_acc"], label="Val Acc", marker="^", markersize=3, color="#2ca02c")
    if "val_macro_f1" in hist:
        axes[1].plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", marker="d", markersize=3, color="#d62728")
    axes[1].axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep {best_epoch}")
    axes[1].set_title("Val Metric vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    # 3. Gradient Norm
    axes[2].plot(epochs, hist["grad_norm"], label="Grad Norm (pre-clip)", marker=".", markersize=3, color="#9467bd")
    axes[2].axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep {best_epoch}")
    axes[2].set_title("Gradient Norm vs Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("L2 Norm")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend()

    title_str = (
        f"[{exp_id}] {cfg.get('optimizer')} (lr={cfg.get('lr')}) | "
        f"loss={cfg.get('loss')} | init={cfg.get('init')} | clip={cfg.get('clip_norm')}"
    )
    fig.suptitle(title_str, fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục."""
    if not results:
        return

    metric_names = {
        "val_loss": "Validation Loss",
        "train_loss": "Train Loss",
        "val_acc": "Validation Accuracy",
        "val_macro_f1": "Validation Macro-F1",
        "grad_norm": "Gradient Norm (pre-clip)",
    }
    label_y = metric_names.get(metric, metric)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for res in results:
        exp_id = res["cfg"].get("exp_id", "exp")
        hist = res["history"]
        if metric in hist and len(hist[metric]) > 0:
            ax.plot(hist["epoch"], hist[metric], label=exp_id, marker="o", markersize=3)

    ax.set_title(title if title else f"So sánh {label_y}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label_y)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()

    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=150, bbox_inches="tight")
    plt.close(fig)
