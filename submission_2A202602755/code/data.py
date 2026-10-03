"""data.py — Nạp dữ liệu, tách validation, chuẩn hoá và cấp phát batch.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.
Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    p = Path(processed_dir)
    train_path = p / "train.npz"
    eval_path = p / "eval.npz"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file npz tại '{processed_dir}'. Hãy chạy 'python scripts/split_data.py' trước."
        )

    tr = np.load(train_path)
    ev = np.load(eval_path)

    X_train_full = tr["X"]
    y_train_full = tr["y"]
    X_eval = ev["X"]
    y_eval = ev["y"]
    eval_row_id = ev["row_id"]

    assert X_train_full.shape == (464809, 54) and X_train_full.dtype == np.float32, (
        f"X_train_full shape/dtype không đúng: {X_train_full.shape}, {X_train_full.dtype}"
    )
    assert y_train_full.shape == (464809,) and y_train_full.dtype == np.int64, (
        f"y_train_full shape/dtype không đúng: {y_train_full.shape}, {y_train_full.dtype}"
    )
    assert X_eval.shape == (116203, 54) and X_eval.dtype == np.float32, (
        f"X_eval shape/dtype không đúng: {X_eval.shape}, {X_eval.dtype}"
    )
    assert y_eval.shape == (116203,) and y_eval.dtype == np.int64, (
        f"y_eval shape/dtype không đúng: {y_eval.shape}, {y_eval.dtype}"
    )
    assert len(eval_row_id) == 116203, "eval_row_id phải có đúng 116,203 phần tử"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Lý do: Không được tính trên eval hay toàn bộ dữ liệu để tránh data leakage.
    """
    numeric_part = X_tr[:, :N_NUMERIC]
    mean = numeric_part.mean(axis=0)
    std = numeric_part.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_scaled = X.copy()
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / std
    return X_scaled


def prepare_data(device: str | torch.device = "cpu", val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước nạp, tách val, chuẩn hoá và đưa lên device."""
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)

    mean, std = fit_standardizer(X_tr)
    X_tr_norm = apply_standardizer(X_tr, mean, std)
    X_val_norm = apply_standardizer(X_val, mean, std)
    X_eval_norm = apply_standardizer(X_eval, mean, std)

    dev = torch.device(device)
    X_tr_t = torch.tensor(X_tr_norm, dtype=torch.float32, device=dev)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64, device=dev)
    X_val_t = torch.tensor(X_val_norm, dtype=torch.float32, device=dev)
    y_val_t = torch.tensor(y_val, dtype=torch.int64, device=dev)
    X_eval_t = torch.tensor(X_eval_norm, dtype=torch.float32, device=dev)
    y_eval_t = torch.tensor(y_eval, dtype=torch.int64, device=dev)

    # Thống kê lớp đa số trên val
    val_counts = np.bincount(y_val, minlength=7)
    majority_acc = float(val_counts.max() / len(y_val))

    print(f"Data prepared on device '{device}':")
    print(f"  Train: X={X_tr_t.shape}, y={y_tr_t.shape}")
    print(f"  Val  : X={X_val_t.shape}, y={y_val_t.shape}")
    print(f"  Eval : X={X_eval_t.shape}, y={y_eval_t.shape}")
    print(f"  Majority class accuracy on val = {majority_acc:.4f} (lớp {val_counts.argmax()})")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader."""
    N = len(X)
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)

    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
