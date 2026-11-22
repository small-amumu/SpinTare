"""Compare HF tokens (no special tokens) with llama-tokenize --ids output of the same text.

Usage: python tokcheck.py <hf dir> <text> <engine ids file>; exit 0 iff the id lists are equal.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from transformers import AutoTokenizer


def main() -> int:
    hf, txt, eng = sys.argv[1:4]
    tk = AutoTokenizer.from_pretrained(hf)
    a = tk.encode(Path(txt).read_text(encoding="utf-8"), add_special_tokens=False)
    b = ast.literal_eval(Path(eng).read_text().strip().splitlines()[-1])
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    name = Path(hf).name
    print(f"{name}: hf {len(a)} engine {len(b)} first mismatch {n} equal {a == b}")
    if n is not None:
        lo = max(0, n - 2)
        print("  hf    ", a[lo:n + 4], tk.convert_ids_to_tokens(a[lo:n + 4]))
        print("  engine", b[lo:n + 4], tk.convert_ids_to_tokens(b[lo:n + 4]))
    return 0 if a == b else 1


if __name__ == "__main__":
    raise SystemExit(main())
