# SPDX-License-Identifier: Apache-2.0
"""PF1 trace analysis: GPU kernel time by category from torch-profiler traces (one per arm).

For every trace (*.json or *.json.gz under the given directory, file name carries the arm as pf1_<arm>...):
GPU kernel events (cat "kernel") are summed by name into categories (attention, kv_write, gemm, norm_act_rope,
sampling, memcpy_memset, other); GPU busy time is the union of kernel intervals; the wall span is the first to the
last kernel. Output: one table per arm (ms and share) plus the top 15 kernels, and a JSON summary.
Usage: python pf1_trace.py <trace_dir> <out.json>
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CATS = [
    (
        "kv_write",
        r"reshape_and_cache|kv_cache_update|cache_update|_flag_exempt|concat_and_cache|write_kv|"
        r"per_token_head_.*(quant|write|store)|_quant_.*cache",
    ),
    ("attention", r"attention|attn|flash|fmha|_fwd_kernel|paged|unified_attention|mu_addback"),
    ("gemm", r"gemm|cutlass|xmma|cublas|sm\d\d_|ampere_|hopper_|Kernel2|matmul|splitK|gemv"),
    ("norm_act_rope", r"rms_norm|layernorm|layer_norm|silu|act_and_mul|gelu|rotary|rope"),
    ("sampling", r"sampl|topk|top_k|argmax|softmax|gumbel|penalt|logit"),
    ("memcpy_memset", r"memcpy|memset|Memcpy|Memset"),
]
RX = [(c, re.compile(p, re.I)) for c, p in CATS]


def category(name: str) -> str:
    for c, rx in RX:
        if rx.search(name):
            return c
    return "other"


def load(p: Path) -> list[dict]:
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def analyze(p: Path) -> dict:
    ev = [
        e
        for e in load(p)
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    by_name: dict[str, float] = defaultdict(float)
    by_cat: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)
    iv = []
    for e in ev:
        name = e.get("name", "")
        dur = float(e.get("dur", 0.0))
        cat = "memcpy_memset" if e.get("cat") != "kernel" else category(name)
        by_name[name] += dur
        by_cat[cat] += dur
        cnt[cat] += 1
        iv.append((float(e["ts"]), float(e["ts"]) + dur))
    iv.sort()
    busy, cur_s, cur_e = 0.0, None, None
    for s, e_ in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e_
        else:
            cur_e = max(cur_e, e_)
    if cur_e is not None:
        busy += cur_e - cur_s
    span = (iv[-1][1] - iv[0][0]) if iv else 0.0
    tot = sum(by_cat.values())
    top = sorted(by_name.items(), key=lambda x: -x[1])[:15]
    return dict(
        trace=str(p),
        kernels=len(ev),
        span_ms=span / 1e3,
        busy_ms=busy / 1e3,
        busy_frac=busy / span if span else 0.0,
        kernel_ms=tot / 1e3,
        by_cat_ms={k: v / 1e3 for k, v in sorted(by_cat.items(), key=lambda x: -x[1])},
        by_cat_count=dict(cnt),
        top=[(n[:120], d / 1e3) for n, d in top],
    )


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = {}
    for p in sorted(list(root.rglob("*.json")) + list(root.rglob("*.json.gz"))):
        m = re.search(r"pf1_([a-z0-9_]+?)(?:_dp|\.|$)", p.name)
        arm = m.group(1) if m else p.stem
        if arm in res or "pf1_" not in p.name:
            continue
        r = analyze(p)
        res[arm] = r
        print(
            f"== {arm}: span {r['span_ms']:.0f} ms, GPU busy {r['busy_frac']:.1%}, kernels {r['kernels']}"
        )
        for c, ms in r["by_cat_ms"].items():
            print(f"   {c:14s} {ms:9.1f} ms  {ms / r['kernel_ms']:6.1%}")
        for n, ms in r["top"][:8]:
            print(f"     {ms:8.1f} ms  {n}")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0 if res else 2


if __name__ == "__main__":
    raise SystemExit(main())
