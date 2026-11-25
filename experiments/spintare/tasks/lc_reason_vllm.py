# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Long chain-of-thought accuracy on a vLLM OpenAI-compatible server, same data / prompt / sampling
/ scoring as lc_reason.py (R1 template, temperature 0.6, top_p 0.95, seed 1000 + item, max 8192 new
tokens).

The prompt is tokenized with the HF tokenizer (no extra special tokens) and sent as token ids to
/v1/completions, so the BOS is not doubled. Generated tokens are counted from usage; the loop flag
re-tokenizes the output text. Usage: python lc_reason_vllm.py --hf <model_dir> --label fp8 --port
8098 --data gsm8k --start 400 --n 200 --par 16 Output:
$KVT_REASON_ROOT/results/{tag}_{data}_{label}_s{start}.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from lc_reason import SUFFIX, last_boxed, load, norm, phrase_loop

ROOT = Path(
    os.environ.get("KVT_REASON_ROOT", str(Path(os.environ.get("KVREL_WORK", "work")) / "reason"))
)


def post(port: int, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=7200).read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--data", default="gsm8k")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--par", type=int, default=16)
    ap.add_argument("--tag", default="r1q7v")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.hf)
    items = load(a.data, a.start + a.n)[a.start :]

    def one(k: int) -> dict:
        it, idx = items[k], a.start + k
        prompt = "<｜begin▁of▁sentence｜><｜User｜>" + it["q"] + SUFFIX + "<｜Assistant｜><think>\n"
        ids = tok(prompt, add_special_tokens=False).input_ids
        r = post(
            a.port,
            {
                "model": "m",
                "prompt": ids,
                "max_tokens": a.max_new,
                "temperature": 0.6,
                "top_p": 0.95,
                "seed": 1000 + idx,
            },
        )
        ch = r["choices"][0]
        text = ch["text"]
        n_gen = r.get("usage", {}).get("completion_tokens", 0)
        pred = last_boxed(text)
        ok = pred is not None and norm(pred) == norm(it["a"])
        toks = tok(text, add_special_tokens=False).input_ids
        return {
            "i": idx,
            "ok": ok,
            "pred": pred,
            "gold": it["a"],
            "n_gen": n_gen,
            "truncated": ch.get("finish_reason") == "length" or n_gen >= a.max_new,
            "loop": phrase_loop(toks),
            "tail": text[-300:],
        }

    t0 = time.time()
    with ThreadPoolExecutor(a.par) as ex:
        recs = list(ex.map(one, range(len(items))))
    n = len(recs)
    s = {
        "acc": 100 * sum(r["ok"] for r in recs) / n,
        "loop": 100 * sum(r["loop"] for r in recs) / n,
        "trunc": 100 * sum(r["truncated"] for r in recs) / n,
        "mean_gen": sum(r["n_gen"] for r in recs) / n,
        "n": n,
        "sec": time.time() - t0,
    }
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"label": a.label, "summary": s, "records": recs},
        open(out / f"{a.tag}_{a.data}_{a.label}_s{a.start}.json", "w"),
        ensure_ascii=False,
    )
    print(
        "DONE",
        a.tag,
        a.data,
        a.label,
        " ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()),
        flush=True,
    )


if __name__ == "__main__":
    main()
