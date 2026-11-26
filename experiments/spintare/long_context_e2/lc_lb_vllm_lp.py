# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""E2 variant of lc_lb_vllm.py: also stores generated token ids and top-2 log-probabilities per
token.

LongBench client for a vLLM OpenAI-compatible server, same prompts / truncation / scoring as
lc_lb.py (census protocol).

Tokenization and the chat template come from the HF tokenizer (--hf); prompts are sent as token ids
to /v1/completions, greedy (temperature 0), so results are comparable with the llama.cpp client up
to tokenizer details. Usage: python lc_lb_vllm.py --model <tag> --hf <model_dir> --label <cfg>
--port 8099 --n 50 --max_len 8000 Output: $KVREL_WORK/results/lc_vllm/{model}/{task}_lc_{label}.json
and summary_lc_{label}.json
"""

from __future__ import annotations
import os

import argparse
import json
import time
import urllib.request
from pathlib import Path

import lb
from lc_lb import phrase_loop


def post(port: int, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--served", default="m")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--tasks", default="hotpotqa,2wikimqa,musique,passage_retrieval_en")
    ap.add_argument("--max_len", type=int, default=8000)
    a = ap.parse_args()
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
    enc = lambda x: ht(x, add_special_tokens=False).input_ids
    head, tail = enc(h), enc(t)
    out = Path(os.environ.get("KVREL_WORK", "work")) / "results" / "lc_vllm" / a.model
    out.mkdir(parents=True, exist_ok=True)
    summary, t0 = {}, time.time()
    for task in a.tasks.split(","):
        rows = [json.loads(l) for l in open(lb.DATA / f"{task}.jsonl", encoding="utf-8")][
            a.start : a.start + a.n
        ]
        recs = []
        for i, r in enumerate(rows):
            body = enc(lb.PROMPT[task].format(context=r["context"], input=r["input"]))
            budget = a.max_len - len(head) - len(tail)
            trunc = len(body) > budget
            if trunc:
                body = body[: budget // 2] + body[-(budget // 2) :]
            ids = head + body + tail
            res = post(
                a.port,
                {
                    "model": a.served,
                    "prompt": ids,
                    "max_tokens": lb.MAXGEN[task],
                    "temperature": 0,
                    "logprobs": 2,
                    "return_tokens_as_token_ids": True,
                },
            )
            pred = res["choices"][0]["text"]
            lpd = res["choices"][0]["logprobs"]
            gen = [int(s.split(":")[1]) for s in lpd["tokens"]]
            top2 = [sorted(d.values(), reverse=True)[:2] for d in lpd["top_logprobs"]]
            recs.append(
                {
                    "i": a.start + i,
                    "len": len(ids),
                    "trunc": trunc,
                    "pred": pred,
                    "answers": r["answers"],
                    "score": lb.score(task, pred, r["answers"]),
                    "loop": phrase_loop(enc(pred)),
                    "gen": gen,
                    "top2": top2,
                }
            )
        avg = 100 * sum(x["score"] for x in recs) / len(recs)
        summary[task] = {"score": avg, "n": len(recs), "loops": sum(x["loop"] for x in recs)}
        json.dump(
            {"label": a.label, "records": recs},
            open(out / f"{task}_lc_{a.label}.json", "w"),
            ensure_ascii=False,
        )
        print(
            a.model,
            a.label,
            task,
            f"{avg:.2f}",
            "loops",
            summary[task]["loops"],
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    summary["avg"] = sum(v["score"] for v in summary.values()) / len(summary)
    json.dump(
        {"label": a.label, "summary": summary},
        open(out / f"summary_lc_{a.label}.json", "w"),
        indent=1,
    )
    print(
        "DONE",
        a.model,
        "vllm",
        a.label,
        f"avg {summary['avg']:.2f}",
        "loops",
        sum(v["loops"] for k, v in summary.items() if k != "avg"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
