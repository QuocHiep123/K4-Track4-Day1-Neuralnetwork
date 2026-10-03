"""train.py — Vòng lặp huấn luyện, đánh giá và lưu kết quả.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1",
    group="baseline",
    description="Baseline M-base (SGD+momentum 0.9, He init, CE)",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Chọn bằng val
    weight_decay=0.0,
    momentum=0.9,
    batch=512,
    epochs=20,
    hidden=(256, 128),
    dropout=0.0,
    init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    Đúng định nghĩa trong scripts/evaluate.py.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds = []
    N = len(X)
    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds, dim=0)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_one_hot = F.one_hot(y, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, y_one_hot)
    else:
        raise ValueError(f"Hàm mất mát '{loss_name}' không hợp lệ (chỉ hỗ trợ 'ce' hoặc 'mse')")


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1, cm) ở chế độ eval() và no_grad."""
    model.eval()
    total_loss = 0.0
    correct = 0
    N = len(X)
    cm = np.zeros((7, 7), dtype=np.int64)

    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)

        if loss_name == "ce":
            batch_loss = F.cross_entropy(logits, yb, reduction="sum").item()
        elif loss_name == "mse":
            y_oh = F.one_hot(yb, num_classes=logits.shape[1]).float()
            batch_loss = F.mse_loss(logits, y_oh, reduction="sum").item()
        else:
            raise ValueError(f"Loss không hỗ trợ: {loss_name}")

        total_loss += batch_loss
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())

        y_true_np = yb.cpu().numpy()
        y_pred_np = pred.cpu().numpy()
        np.add.at(cm, (y_true_np, y_pred_np), 1)

    macro_f1 = macro_f1_from_confusion(cm)
    return {
        "loss": float(total_loss / N),
        "acc": float(correct / N),
        "macro_f1": float(macro_f1),
        "cm": cm,
    }


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = dict(cfg)
    if cfg.get("lr") is None:
        raise ValueError(f"Cấu hình thí nghiệm {cfg.get('exp_id')} thiếu 'lr'!")

    set_seed(cfg["seed"])
    device = data["X_tr"].device

    # Khởi tạo mô hình
    hidden = tuple(cfg["hidden"])
    model = MLP(
        hidden=hidden,
        dropout=cfg.get("dropout", 0.0),
        init=cfg.get("init", "he"),
    ).to(device)

    assert count_params(model) == EXPECTED_PARAMS[hidden], (
        f"Số tham số {count_params(model)} không khớp {EXPECTED_PARAMS[hidden]}"
    )

    optimizer = build_optimizer(
        cfg["optimizer"],
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
        momentum=cfg.get("momentum", 0.9),
    )

    # Thiết lập mixed precision
    precision = cfg.get("precision", "fp32")
    use_amp = (precision in ("fp16", "bf16")) and (device.type == "cuda")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler(device="cuda") if (precision == "fp16" and device.type == "cuda") else None

    # Bước 0: Đo loss ban đầu trước khi cập nhật
    step0_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg.get("loss", "ce"))
    step0_loss = step0_res["loss"]

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = None
    diverged = False

    # Tập con train cố định 50,000 mẫu để đo train_loss nhanh ở cuối mỗi epoch
    eval_sub = min(50000, len(data["X_tr"]))
    X_tr_sub = data["X_tr"][:eval_sub]
    y_tr_sub = data["y_tr"][:eval_sub]

    gen = torch.Generator(device="cpu" if device.type != "cuda" else "cuda")
    gen.manual_seed(cfg["seed"])

    epochs = cfg.get("epochs", 20)
    batch_size = cfg.get("batch", 512)
    clip_norm = cfg.get("clip_norm", None)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        step_grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size, generator=gen, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg.get("loss", "ce"))
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg.get("loss", "ce"))

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            step_grad_norms.append(gn)

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        if diverged:
            print(f"[{cfg.get('exp_id')}] DIVERGED tại epoch {epoch} (loss = NaN/inf)!")
            break

        # Đánh giá cuối epoch
        eval_tr = evaluate(model, X_tr_sub, y_tr_sub, loss_name=cfg.get("loss", "ce"))
        eval_val = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg.get("loss", "ce"))
        mean_gn = float(np.mean(step_grad_norms)) if step_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(eval_tr["loss"])
        history["val_loss"].append(eval_val["loss"])
        history["val_acc"].append(eval_val["acc"])
        history["val_macro_f1"].append(eval_val["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(epoch_time)

        # Lưu best state
        if eval_val["loss"] < best_val_loss:
            best_val_loss = eval_val["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Bộ nhớ đỉnh
    peak_mem_MB = 0.0
    if device.type == "cuda":
        peak_mem_MB = float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024))

    if not diverged and len(history["val_loss"]) > 0:
        best_idx = best_epoch - 1
        summary = {
            "step0_loss": float(step0_loss),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "val_acc": float(history["val_acc"][best_idx]),
            "val_macro_f1": float(history["val_macro_f1"][best_idx]),
            "time_per_epoch_s": float(np.mean(history["epoch_time_s"])),
            "peak_mem_MB": round(peak_mem_MB, 2),
            "diverged": False,
        }
    else:
        summary = {
            "step0_loss": float(step0_loss),
            "best_val_loss": float("nan"),
            "best_epoch": 0,
            "final_train_loss": float("nan"),
            "final_val_loss": float("nan"),
            "val_acc": 0.0,
            "val_macro_f1": 0.0,
            "time_per_epoch_s": 0.0,
            "peak_mem_MB": round(peak_mem_MB, 2),
            "diverged": True,
        }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    df = pd.DataFrame({
        "row_id": np.asarray(row_id, dtype=np.int64),
        "pred": np.asarray(preds, dtype=np.int64),
    })
    df.to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=cfg.get("dropout", 0.0), init=cfg.get("init", "he")).to(device)
    model.load_state_dict(result["best_state"])

    preds = predict(model, data["X_eval"])
    preds_np = preds.cpu().numpy()
    write_predictions(data["eval_row_id"], preds_np, pred_path)
    print(f"Đã lưu dự đoán eval vào {pred_path} ({len(preds_np)} dòng)")
