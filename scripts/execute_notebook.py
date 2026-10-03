"""execute_notebook.py — Chạy toàn bộ cell của lab.ipynb và nhúng output thực thi trực tiếp vào file .ipynb."""
import sys
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def execute_and_save(nb_path: str):
    p = Path(nb_path).resolve()
    print(f"Đang thực thi {p}...")
    with open(p, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(p.parent)}})

    with open(p, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Đã thực thi và lưu đầy đủ output vào: {p}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "submission_2A202602755/code/lab.ipynb"
    execute_and_save(target)
