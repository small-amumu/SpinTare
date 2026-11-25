# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""CH4X copy of lc_reason.py (ad33ed04) with --start (global item index for the seed and the
record). CH4X A5: + --journal (each finished record is appended to a JSONL journal as it completes;
on start the items already in the journal are skipped, so a run cut by a server crash resumes;
in-flight items rerun with the same seed). The output file holds all records (in item order);
summary["sec"] is this attempt's time, summary["resumed"] the count of records taken from the
journal. Long chain-of-thought reasoning on llama.cpp with a real quantized K cache: accuracy,
looping rate and generation length. Model: DeepSeek-R1-Distill-Qwen-7B (Qwen2 architecture, k_proj
bias). Data: GSM8K test and MATH-500, items start..start+n. Prompt as recommended for DeepSeek-R1:
the user message ends with "Please reason step by step, and put your final answer within \\boxed{}."
and the assistant turn starts with "<think>\\n"; temperature 0.6, top_p 0.95, a fixed seed per item
(the same for every K-cache arm). Client only: llama-server must run with -np P parallel slots; this
script sends P concurrent requests. Usage: python lc_reason.py --ctk q4_0 --port 8092 --data gsm8k
--n 200 --max_new 8192 --par 4
       [--journal FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(
    os.environ.get("KVT_REASON_ROOT", os.path.join(os.environ.get("KVREL_WORK", "work"), "reason"))
)
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=7200).read())


def load(name, n):
    import pyarrow.parquet as pq  # the dataset is parquet

    if name == "gsm8k":
        f = sorted((ROOT / "gsm8k" / "main").glob("test-*.parquet"))[0]
        rows = pq.read_table(f).to_pylist()[:n]
        return [
            {"q": r["question"], "a": r["answer"].split("####")[-1].strip().replace(",", "")}
            for r in rows
        ]
    f = sorted((ROOT / "math500").rglob("*.jsonl"))
    rows = [json.loads(l) for l in open(f[0])][:n]
    return [{"q": r["problem"], "a": r["answer"]} for r in rows]


def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(s) and d:
        d += {"{": 1, "}": -1}.get(s[j], 0)
        j += 1
    return s[i + 7 : j - 1] if d == 0 else None


def norm(x):
    x = (
        x.replace("\\left", "")
        .replace("\\right", "")
        .replace("\\!", "")
        .replace("dfrac", "frac")
        .replace("tfrac", "frac")
    )
    x = (
        re.sub(r"\\text\{([^}]*)\}", r"\1", x)
        .replace("^\\circ", "")
        .replace("^{\\circ}", "")
        .replace("\\%", "")
        .replace("$", "")
    )
    x = re.sub(r"\s+", "", x).rstrip(".")
    try:
        return str(float(x.replace(",", ""))).rstrip("0").rstrip(".")
    except ValueError:
        return x


def phrase_loop(
    tokens, max_period=64, min_reps=4
):  # long reasoning: period up to 64 tokens, >= 4 repeats
    for per in range(1, max_period + 1):
        need = per * min_reps
        if len(tokens) < need:
            break
        tail = tokens[-need:]
        if all(tail[i] == tail[i % per] for i in range(need)):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctk", required=True)
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--data", default="gsm8k")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--tag", default="r1q7")
    ap.add_argument("--start", type=int, default=0)  # CH4X: item range start..start+n-1
    ap.add_argument("--journal", default=None)  # CH4X A5: per-record JSONL
    a = ap.parse_args()
    if a.n < 1 or a.start < 0:
        raise SystemExit("need --n >= 1 and --start >= 0")
    bind = {
        "ctk": a.ctk,
        "data": a.data,
        "max_new": a.max_new,
        "tag": a.tag,
    }  # journal belongs to this cell
    items = load(a.data, a.start + a.n)[a.start :]
    if len(items) != a.n:
        raise SystemExit(f"dataset has {len(items)} items in range, want {a.n}")
    jpath = Path(a.journal) if a.journal else None  # no journal: run every item
    done: dict[int, dict] = {}
    if jpath is not None and jpath.exists():
        for line in jpath.read_bytes().splitlines():
            try:
                j = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):  # a torn last line from a kill
                continue
            if j.get("bind") != bind:
                raise SystemExit(f"journal line bound to {j.get('bind')}, this run is {bind}")
            if "rec" not in j:  # a "started" line
                continue
            r = j["rec"]
            if not (a.start <= r["i"] < a.start + a.n) or r["gold"] != items[r["i"] - a.start]["a"]:
                raise SystemExit(f"journal record {r['i']} does not belong to this run")
            done[r["i"]] = r
        if jpath.stat().st_size and not jpath.read_bytes().endswith(
            b"\n"
        ):  # terminate a torn last line
            with open(jpath, "a", encoding="utf-8") as jf:
                jf.write("\n")
    todo = [k for k in range(a.n) if a.start + k not in done]
    print(f"journal {jpath}: {len(done)} done, {len(todo)} to run", flush=True)
    jlock = threading.Lock()
    pid = os.getpid()

    def jwrite(obj: dict) -> None:
        if jpath is None:
            return
        with jlock:  # CH4X A5: durable per line
            with open(jpath, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(obj, ensure_ascii=True) + "\n")
                jf.flush()
                os.fsync(jf.fileno())

    def one(k):
        it = items[k]
        jwrite(
            {"bind": bind, "started": a.start + k, "pid": pid}
        )  # names the in-flight set at a crash
        prompt = "<｜begin▁of▁sentence｜><｜User｜>" + it["q"] + SUFFIX + "<｜Assistant｜><think>\n"
        r = post(
            a.port,
            "/completion",
            {
                "prompt": prompt,
                "n_predict": a.max_new,
                "temperature": 0.6,
                "top_p": 0.95,
                "seed": 1000 + a.start + k,
                "cache_prompt": False,
                "return_tokens": True,
            },
        )
        toks = r.get("tokens") or []
        pred = last_boxed(r["content"])
        ok = pred is not None and norm(pred) == norm(it["a"])
        rec = {
            "i": a.start + k,
            "ok": ok,
            "pred": pred,
            "gold": it["a"],
            "n_gen": r.get("tokens_predicted", len(toks)),
            "truncated": r.get("stop_type") == "limit"
            or r.get("truncated", False)
            or (r.get("tokens_predicted", 0) >= a.max_new),
            "loop": phrase_loop(toks),
            "tail": r["content"][-300:],
        }
        jwrite({"bind": bind, "rec": rec, "pid": pid})
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(a.par) as ex:
        new = list(ex.map(one, todo))
    for r in new:
        done[r["i"]] = r
    recs = [done[a.start + k] for k in range(a.n)]
    n = len(recs)
    s = {
        "acc": 100 * sum(r["ok"] for r in recs) / n,
        "loop": 100 * sum(r["loop"] for r in recs) / n,
        "trunc": 100 * sum(r["truncated"] for r in recs) / n,
        "mean_gen": sum(r["n_gen"] for r in recs) / n,
        "n": n,
        "sec": time.time() - t0,
        "resumed": n - len(new),
    }
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"ctk": a.ctk, "summary": s, "records": recs},
        open(out / f"{a.tag}_{a.data}_{a.ctk}_s{a.start}n{a.n}.json", "w"),
        ensure_ascii=False,
    )
    print(
        "DONE",
        a.tag,
        a.data,
        a.ctk,
        f"s{a.start}n{a.n}",
        " ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()),
        flush=True,
    )


if __name__ == "__main__":
    main()
