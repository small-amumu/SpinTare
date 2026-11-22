"""Needle-in-a-haystack probe as a detection baseline (after the NIAH probe of SpectrumKV), on
llama.cpp with a real quantized K cache; compared with the one-pass offset statistics for cost and
accuracy.

Haystack: LongBench hotpotqa contexts joined and cut to --ctx tokens; needle: "The special magic
number is NNNNNNN." at --n depths spread evenly over [0.05, 0.95]; greedy decoding of 24 tokens, a
hit when the output contains the number. Client only (llama-server on --port); templates as
lc_lb.py. Usage: python lc_probe.py --model qwen25 --ctk q4_0 --port 8097 --n 20 --ctx 4000 Output:
$KVREL_WORK/results/probe/{model}/probe_{ctk}_{ctx}.json, last line "DONE ... acc X wall Ys"
"""

from __future__ import annotations
import os

import argparse
import json
import random
import time
from pathlib import Path

import lb
from lc_lb import HEAD, TAIL, post, tok

Q = "\n\nWhat is the special magic number mentioned in the text above? Answer with the number only."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctk", required=True)
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=4000)
    ap.add_argument("--hf", default=None)
    a = ap.parse_args()
    if a.model in HEAD:
        h, t = HEAD[a.model], TAIL[a.model]
    else:
        from transformers import AutoTokenizer

        ht = AutoTokenizer.from_pretrained(a.hf)
        msgs = [{"role": "user", "content": "@@SPLIT@@"}]
        try:
            s_ = ht.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            s_ = ht.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        h, t = s_.split("@@SPLIT@@")
    t0 = time.time()
    head, tail, q = tok(a.port, h, True), tok(a.port, t, True), tok(a.port, Q, False)
    rows = [json.loads(l) for l in open(lb.DATA / "hotpotqa.jsonl", encoding="utf-8")][:40]
    hay = tok(a.port, "\n".join(r["context"] for r in rows), False)
    rng = random.Random(0)
    recs = []
    for i in range(a.n):
        num = str(rng.randint(1_000_000, 9_999_999))
        needle = tok(a.port, f" The special magic number is {num}. ", False)
        body_len = a.ctx - len(head) - len(tail) - len(q) - len(needle)
        depth = 0.05 + 0.9 * i / max(a.n - 1, 1)
        cut = int(body_len * depth)
        ids = head + hay[:cut] + needle + hay[cut:body_len] + q + tail
        res = post(
            a.port,
            "/completion",
            {"prompt": ids, "n_predict": 24, "temperature": 0, "top_k": 1, "cache_prompt": False},
        )
        recs.append(
            {
                "i": i,
                "depth": round(depth, 3),
                "num": num,
                "pred": res["content"],
                "hit": num in res["content"],
            }
        )
    acc = sum(r["hit"] for r in recs) / len(recs)
    wall = time.time() - t0
    out = Path(os.environ.get("KVREL_WORK", "work")) / "results" / "probe" / a.model
    out.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"ctk": a.ctk, "ctx": a.ctx, "acc": acc, "wall_s": wall, "records": recs},
        open(out / f"probe_{a.ctk}_{a.ctx}.json", "w"),
        ensure_ascii=False,
        indent=1,
    )
    print("DONE", a.model, "probe", a.ctk, a.ctx, f"acc {acc:.2f}", f"wall {wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
