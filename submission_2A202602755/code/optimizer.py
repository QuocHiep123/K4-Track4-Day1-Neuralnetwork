"""optimizer.py — Bộ tối ưu, scheduler và cắt gradient.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).
File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer."""
    if name not in OPTIMIZERS:
        raise ValueError(f"Bộ tối ưu '{name}' không nằm trong {OPTIMIZERS}")

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, ví dụ cosine."""
    if name is None or name == "" or name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    raise ValueError(f"Scheduler không hỗ trợ: {name}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Khi dùng mixed precision FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    param_list = list(params)
    param_list = [p for p in param_list if p.grad is not None]
    if not param_list:
        return 0.0

    if max_norm is None:
        total_norm = torch.nn.utils.clip_grad_norm_(param_list, max_norm=float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(param_list, max_norm=float(max_norm))

    return float(total_norm)
