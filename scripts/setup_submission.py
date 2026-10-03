"""setup_submission.py — Tạo thư mục submission_<MSSV> và sao chép toàn bộ code chuẩn."""
import os
import shutil
from pathlib import Path

MSSV = "2A202602755"
SUB_DIR = Path(f"submission_{MSSV}")

# Tạo cấu trúc thư mục
(SUB_DIR / "code").mkdir(parents=True, exist_ok=True)
(SUB_DIR / "figures").mkdir(parents=True, exist_ok=True)
(SUB_DIR / "results").mkdir(parents=True, exist_ok=True)

# Copy toàn bộ file code từ code/ sang submission_<MSSV>/code/
code_files = [
    "data.py", "model.py", "optimizer.py", "train.py",
    "plots.py", "results_table.py", "lab.ipynb"
]

for cf in code_files:
    src = Path("code") / cf
    dst = SUB_DIR / "code" / cf
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"Warning: {src} not found!")

print(f"Submission folder '{SUB_DIR}' initialized successfully!")
