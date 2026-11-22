"""T2 (ext_compare/REGISTRATION_T2_runmean.md): turboQuantDC's running-mean key centring in the LC1
harness.

Imports lc1_long.py unchanged (its directory is given with --lc1-dir) and only adds arms. The
registered run used the original lc1_long.py (md5 LC1_MD5); the repository copy next to this file
differs only in where it imports the ggml quantizers from (spintare/ggml_ref.py) and was checked to
reproduce LC1 bit for bit, so it is accepted too:
  RMP_f    prefill as one chunk: m = mean over [0, L) of the test text's own post-RoPE keys
  RM512_f  chunks of 512 positions: chunk [s, e) uses m_e = mean over [0, e)
  RMT_f    token by token: position t uses m_t = mean over [0, t]
for f in nr / ro: reconstruct(k - m) + bf16(m) (subtract the float32 mean, add back its bfloat16
rounding, as turboquantdc/generation_layers.py _CompressedLayer.update stores
`chunk_mean.to(key_states.dtype)`). Arms run: X, A_nr, A_ro, D_nr, D_ro and the six above.
Everything else (texts, calibration, buckets, exact pass, rho readings, checks) is lc1_long.real.

Usage: python t2_runmean.py real <out_json> <model_dir> <short> --lc1-dir DIR --order hf|gguf --data
DIR [--L 32768]
       python t2_runmean.py agg <dir with t2_<short>.json> --logs DIR --lc1-res DIR --lc1-dir DIR
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import statistics as st
import sys
from pathlib import Path

import torch

LC1_MD5 = "a010e27d745f383cb58e057b3d8efe4b"
LC1_SEEN: dict[str, str] = {}
FMTS = ("nr", "ro")
RM = ("RMP", "RM512", "RMT")
ARMS = ("X",) + tuple(f"{t}_{f}" for f in FMTS for t in ("A", "D") + RM)
NEED_CENTRING_MIN = 2.0
NO_CENTRING = ("llama31_8b", "llama3b", "mistral7b", "granite8b")
LC = None  # the lc1_long module
_orig_arm_keys = None


def running_means(y64: torch.Tensor, arm: str) -> torch.Tensor:
    """y64 (KVH, n, d) float64 post-RoPE keys -> the mean each position is centred with (KVH, n,
    d)."""
    kvh, n, d = y64.shape
    if arm == "RMP":
        return y64.mean(1, keepdim=True).expand(kvh, n, d)
    cs = y64.cumsum(1)  # cs[:, t] = sum over [0, t]
    if arm == "RMT":
        cnt = torch.arange(1, n + 1, dtype=torch.float64, device=y64.device)
        return cs / cnt[None, :, None]
    e = torch.clamp(
        (torch.arange(n, device=y64.device) // 512 + 1) * 512, max=n
    )  # end of each position's chunk
    return cs[:, e - 1] / e.to(torch.float64)[None, :, None]


def arm_keys(y32: torch.Tensor, li: int, arm: str) -> torch.Tensor:
    if arm == "X" or arm.split("_")[0] in ("A", "D"):
        return _orig_arm_keys(y32, li, arm)
    treat, fmt = arm.split("_")
    assert treat in RM, arm
    m = running_means(y32.double(), treat)
    m32 = m.float()
    back = m32.to(torch.bfloat16).double()  # the stored snapshot, bfloat16
    return (LC.reconstruct(y32 - m32, LC.FMT[fmt]).double() + back).float()


def load_lc1(d: str):
    global LC, _orig_arm_keys
    p = Path(d) / "lc1_long.py"
    got = hashlib.md5(p.read_bytes()).hexdigest()
    LC1_SEEN["md5"] = got
    if got != LC1_MD5 and p.resolve().parent != Path(__file__).resolve().parent:
        raise SystemExit(f"lc1_long.py md5 {got} != {LC1_MD5} and it is not the repository copy")
    sys.path.insert(0, d)
    LC = importlib.import_module("lc1_long")
    _orig_arm_keys = LC.arm_keys
    LC.arm_keys = arm_keys  # lc_attn looks the name up in its module
    LC.ARMS = ARMS
    return LC


def real(a) -> int:
    load_lc1(a.lc1_dir)
    rc = LC.real(a)
    if rc == 0:
        d = json.loads(Path(a.out).read_text())
        d["meta"]["t2_script_md5"] = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
        d["meta"]["lc1_md5"] = LC1_SEEN["md5"]
        tmp = Path(str(a.out) + ".tmp2")
        tmp.write_text(json.dumps(d, allow_nan=False))
        tmp.replace(a.out)
    return rc


def agg(a) -> int:
    need = lambda n: max(1, math.ceil(0.75 * n))  # noqa: E731
    models, excluded = {}, {}
    for f in sorted(Path(a.dir).glob("t2_*.json")):
        if f.name.endswith(".smoke.json"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        m = d["meta"]
        short = m["short"]
        rc = Path(a.logs) / f"t2_{short}.rc"
        if short not in LC1_EXPECTED or m["dtype"].startswith("float32"):
            excluded[short] = "unplanned model or float32 weights"
            continue
        if (
            not d.get("complete")
            or len(d["rows"]) != 2
            or not rc.exists()
            or rc.read_text().strip() != "0"
        ):
            excluded[short] = "incomplete or exit code not 0"
            continue
        nb = len(m["buckets"])
        kl = {
            arm: [st.mean(r["arms"][arm]["kl"][b] for r in d["rows"]) for b in range(nb)]
            for arm in m["arms"]
        }
        lc1 = json.loads((Path(a.lc1_res) / f"lc1_{short}.json").read_text(encoding="utf-8"))
        assert lc1["meta"]["buckets"] == m["buckets"], short
        kl1 = {
            arm: [st.mean(r["arms"][arm]["kl"][b] for r in lc1["rows"]) for b in range(nb)]
            for arm in lc1["meta"]["arms"]
        }
        out: dict = {
            "in_range": (m["max_position_embeddings"] or 0) >= m["L"],
            "kl": kl,
            "D_vs_LC1": {
                f_: [kl[f"D_{f_}"][b] / kl1[f"D_{f_}"][b] for b in range(nb)] for f_ in FMTS
            },
            "per_text": {},
        }
        out["D_reproducible"] = all(abs(x - 1) <= 0.02 for f_ in FMTS for x in out["D_vs_LC1"][f_])
        for f_ in FMTS:
            out[f"needs_centring_{f_}"] = kl[f"A_{f_}"][0] >= NEED_CENTRING_MIN * kl[f"D_{f_}"][0]
            for x in RM:
                h = [kl[f"{x}_{f_}"][b] / kl[f"D_{f_}"][b] for b in range(nb)]
                out[f"h_{x}_{f_}"] = h
                out[f"{x}_over_A_{f_}"] = [kl[f"{x}_{f_}"][b] / kl[f"A_{f_}"][b] for b in range(nb)]
            out[f"RMT_over_B2lc1_{f_}"] = [
                kl[f"RMT_{f_}"][b] / kl1[f"B2_{f_}"][b] for b in range(nb)
            ]
            out[f"RMP_over_BLlc1_{f_}"] = [
                kl[f"RMP_{f_}"][b] / kl1[f"BL_{f_}"][b] for b in range(nb)
            ]
            out[f"RMT_over_Elc1_{f_}"] = [kl[f"RMT_{f_}"][b] / kl1[f"E_{f_}"][b] for b in range(nb)]
        for r in d["rows"]:
            a_ = r["arms"]
            out["per_text"][r["text"]] = {
                f"h_{x}_{f_}_last": a_[f"{x}_{f_}"]["kl"][-1] / a_[f"D_{f_}"]["kl"][-1]
                for x in RM
                for f_ in FMTS
            }
        if short in NO_CENTRING:
            for f_ in FMTS:
                out[f"T2_4_harms_{f_}"] = sum(v > 1.1 for v in out[f"RMT_over_A_{f_}"]) >= 3
        models[short] = out
    res: dict = {
        "models": models,
        "excluded": excluded,
        "missing": [x for x in LC1_EXPECTED if x not in models and x not in excluded],
        "verdicts": {},
    }
    for scope in ("all", "in_range"):
        ms = {k: v for k, v in models.items() if scope == "all" or v["in_range"]}
        v: dict = {}
        for f_ in FMTS:
            sub = {k: x for k, x in ms.items() if x[f"needs_centring_{f_}"]}
            n = len(sub)
            for x in RM:
                key = f"{x}_{f_}"
                if n < 3:
                    v[key] = {"models": n, "verdict": "UNDECIDABLE"}
                    continue
                t1 = sum(y[f"h_{x}_{f_}"][-1] >= 1.5 for y in sub.values())
                t2 = sum(y[f"h_{x}_{f_}"][-1] >= 1.5 * y[f"h_{x}_{f_}"][0] for y in sub.values())
                v[key] = {
                    "models": n,
                    "names": sorted(sub),
                    "T2_1_n": t1,
                    "T2_1_holds": t1 >= need(n),
                    "T2_2_n": t2,
                    "T2_2_holds": t2 >= need(n),
                }
        res["verdicts"][scope] = v
    json.dump(res, sys.stdout, indent=1)
    return 0


LC1_EXPECTED = (
    "llama31_8b",
    "mistral7b",
    "olmo7b",
    "q25_1p5b",
    "q3_8b",
    "q25_7b",
    "granite8b",
    "llama3b",
    "yi6b",
    "ds7b",
    "r1q_7b",
    "q3_0p6b",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("real")
    r.add_argument("out", type=Path)
    r.add_argument("model_dir")
    r.add_argument("short")
    r.add_argument("--lc1-dir", required=True)
    r.add_argument("--order", required=True, choices=["hf", "gguf"])
    r.add_argument("--data", required=True)
    r.add_argument("--L", type=int, default=32768)
    r.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    r.add_argument("--device", default="cuda")
    r.add_argument("--smoke", action="store_true")
    g = sub.add_parser("agg")
    g.add_argument("dir")
    g.add_argument("--logs", required=True)
    g.add_argument("--lc1-res", required=True)
    g.add_argument("--lc1-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "real":
        return real(a)
    load_lc1(a.lc1_dir)
    return agg(a)


if __name__ == "__main__":
    raise SystemExit(main())
