# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""LongBench on llama.cpp with a real quantized K cache; prompts, truncation and scoring as lb.py.

Client only: llama-server must already run on --port (e.g. -fa on -ctk <K type> -ctv q8_0).
Usage: python lc_lb_ho.py --model qwen25 --ctk q4_0 --port 8091 --n 50
Output: $KVREL_WORK/results/lc/{model}/{task}_lc_{ctk}.json and summary_lc_{ctk}.json
"""

from __future__ import annotations
import os

import argparse
import json
import time
import urllib.request
from pathlib import Path

import lb

QWEN25_SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
HEAD = {
    "qwen25": f"<|im_start|>system\n{QWEN25_SYS}<|im_end|>\n<|im_start|>user\n",
    "qwen3": "<|im_start|>user\n",
    "llama31": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
}
TAIL = {
    "qwen25": "<|im_end|>\n<|im_start|>assistant\n",
    "qwen3": "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "llama31": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
}


def post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def tok(port: int, text: str, special: bool) -> list[int]:
    return post(
        port, "/tokenize", {"content": text, "add_special": False, "parse_special": special}
    )["tokens"]


def phrase_loop(
    tokens: list[int], max_period: int = 8, min_reps: int = 3
) -> bool:  # phrase-loop detector of the harness
    for per in range(1, max_period + 1):
        need = per * min_reps
        if len(tokens) < need:
            break
        tail = tokens[-need:]
        if all(tail[i] == tail[i % per] for i in range(need)):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctk", required=True)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tasks", default="hotpotqa,2wikimqa,musique,passage_retrieval_en")
    ap.add_argument("--max_len", type=int, default=31500)
    ap.add_argument("--hf", default=None)  # head / tail cut from the HF chat template
    ap.add_argument("--start", type=int, default=0)  # item range start..start+n (default 0)
    ap.add_argument(
        "--sampling", default=None
    )  # JSON overriding greedy, e.g. {"repeat_penalty": 1.3}
    a = ap.parse_args()
    if a.model in HEAD:
        h, t = HEAD[a.model], TAIL[a.model]
    else:
        from transformers import AutoTokenizer

        ht = AutoTokenizer.from_pretrained(a.hf, trust_remote_code=True)
        msgs = [{"role": "user", "content": "@@SPLIT@@"}]
        try:
            s_ = ht.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            s_ = ht.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        h, t = s_.split("@@SPLIT@@")
    head, tail = tok(a.port, h, True), tok(a.port, t, True)
    out = Path(os.environ.get("KVREL_WORK", "work")) / "results" / "lc" / a.model
    out.mkdir(parents=True, exist_ok=True)
    summary, t0 = {}, time.time()
    for task in a.tasks.split(","):
        rows = [json.loads(l) for l in open(lb.DATA / f"{task}.jsonl", encoding="utf-8")][
            a.start : a.start + a.n
        ]
        recs = []
        for i, r in enumerate(rows):
            body = tok(
                a.port, lb.PROMPT[task].format(context=r["context"], input=r["input"]), False
            )
            budget = a.max_len - len(head) - len(tail)
            trunc = len(body) > budget
            if trunc:
                body = body[: budget // 2] + body[-(budget // 2) :]
            ids = head + body + tail
            req = {
                "prompt": ids,
                "n_predict": lb.MAXGEN[task],
                "temperature": 0,
                "top_k": 1,
                "cache_prompt": False,
                "return_tokens": True,
            }
            if a.sampling:
                req.update(json.loads(a.sampling))
            res = post(a.port, "/completion", req)
            pred = res["content"]
            gen = res.get("tokens") or tok(a.port, pred, False)
            recs.append(
                {
                    "i": a.start + i,
                    "len": len(ids),
                    "trunc": trunc,
                    "pred": pred,
                    "answers": r["answers"],
                    "score": lb.score(task, pred, r["answers"]),
                    "loop": phrase_loop(gen),
                    "tps_decode": (res.get("timings") or {}).get("predicted_per_second"),
                    "tps_prefill": (res.get("timings") or {}).get("prompt_per_second"),
                }
            )
        avg = 100 * sum(x["score"] for x in recs) / len(recs)
        summary[task] = {"score": avg, "n": len(recs), "loops": sum(x["loop"] for x in recs)}
        json.dump(
            {"ctk": a.ctk, "records": recs},
            open(out / f"{task}_lc_{a.ctk}.json", "w"),
            ensure_ascii=False,
        )
        print(
            a.model,
            a.ctk,
            task,
            f"{avg:.2f}",
            "loops",
            summary[task]["loops"],
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    summary["avg"] = sum(v["score"] for v in summary.values()) / len(summary)
    json.dump(
        {"ctk": a.ctk, "summary": summary}, open(out / f"summary_lc_{a.ctk}.json", "w"), indent=1
    )
    print(
        "DONE",
        a.model,
        "lc",
        a.ctk,
        f"avg {summary['avg']:.2f}",
        "loops",
        sum(v["loops"] for k, v in summary.items() if k != "avg"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
