"""CH4Z gate F0: re-create a SmoothAttention-folded checkpoint from a recorded lambda
(REGISTRATION_CH4Z.md).

Uses sa_fold.write_folded() from the unchanged sa_fold.py (md5 2fad4d61, next to this file), which
writes the folded directory and verifies it (every target tensor = the fold of the source and
changed, every other tensor bit-identical). Then every output shard's md5 must equal the
out_files_md5 of the fold's gate JSON (the original fold run), and the rewritten / identical counts
must equal its "write" record. Exit 0 only then. usage: python sa_apply.py --model HF_DIR --out
OUT_DIR --lambda X.lambda.pt --gate-json X.gate.json --result OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sa_fold  # noqa: E402

SA_FOLD_MD5 = "2fad4d612321110e061d3fc40db1ed79"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lambda", dest="lam", required=True)
    ap.add_argument("--gate-json", required=True)
    ap.add_argument("--result", required=True)
    a = ap.parse_args()
    here = Path(sa_fold.__file__)
    if sa_fold.md5(here) != SA_FOLD_MD5:
        raise SystemExit(f"sa_fold.py md5 {sa_fold.md5(here)} != {SA_FOLD_MD5}")
    gate = json.loads(Path(a.gate_json).read_text(encoding="utf-8"))
    if not gate["gate"]["pass"]:
        raise SystemExit("the recorded fold did not pass its gate")
    d = torch.load(a.lam, map_location="cpu", weights_only=True)
    lam = d["lambda"]
    # The released lambda files drop the run machine's model path, so the file md5 differs from the
    # recorded one; the lambda tensor itself must match the recorded tensor md5.
    tensor_md5 = hashlib.md5(lam.contiguous().numpy().tobytes()).hexdigest()
    if "lambda_tensor_md5" in gate and tensor_md5 != gate["lambda_tensor_md5"]:
        raise SystemExit("lambda tensor md5 differs from the gate record")
    if "lambda_tensor_md5" not in gate and sa_fold.md5(Path(a.lam)) != gate["lambda_md5"]:
        raise SystemExit("lambda file md5 differs from the gate record")
    cfg = AutoConfig.from_pretrained(a.model)
    q3 = cfg.model_type == "qwen3"
    if q3 != bool(gate["qwen3_shared_lambda"]) or q3 != bool(d["shared_over_kv_heads"]):
        raise SystemExit(f"architecture {cfg.model_type} does not match the recorded fold")
    if list(lam.shape) != gate["lambda_shape"]:
        raise SystemExit(f"lambda shape {list(lam.shape)} != {gate['lambda_shape']}")
    w = sa_fold.write_folded(Path(a.model), Path(a.out), lam, cfg, q3)
    md5s = {f.name: sa_fold.md5(f) for f in sorted(Path(a.out).glob("*.safetensors"))}
    ok = md5s == gate["out_files_md5"] and w == gate["write"]
    res = {
        "model": a.model,
        "out": a.out,
        "lambda_md5": gate["lambda_md5"],
        "write": w,
        "recorded_write": gate["write"],
        "out_files_md5": md5s,
        "recorded_out_files_md5": gate["out_files_md5"],
        "identical_to_record": ok,
        "torch": torch.__version__,
    }
    Path(a.result).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(
        "SAAPPLY",
        "ok" if ok else "MISMATCH",
        json.dumps({"write": w, "md5_equal": md5s == gate["out_files_md5"]}),
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
