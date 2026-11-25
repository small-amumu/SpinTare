# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""LongBench task definitions shared by the llama.cpp and vLLM clients (lc_lb.py, lc_lb_vllm.py).

Prompt templates, generation lengths and scoring follow the official repository (THUDM/LongBench,
pred.py / metrics.py):
  - qa_f1: token F1 after answer normalisation (lower case, no punctuation, no articles, collapsed
    whitespace), maximum over the reference answers;
  - retrieval: the "Paragraph N" named in the output against the reference, share of matches.
Over-long inputs are truncated in the middle, as in the official code. `main` drives an in-process
simulated-quantization harness that is not part of this repository; the clients use the task
definitions only, and `selftest` checks the metrics.

Usage:
  python lb.py selftest
"""

from __future__ import annotations
import os

import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

DATA = Path(os.environ.get("KVREL_WORK", "work")) / "data" / "data"
OUT = Path(os.environ.get("KVREL_WORK", "work")) / "results"

PROMPT = {
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
}
MAXGEN = {
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "passage_retrieval_en": 32,
    "multifieldqa_en": 64,
    "qasper": 128,
}
METRIC = {
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "musique": "f1",
    "passage_retrieval_en": "retrieval",
    "multifieldqa_en": "f1",
    "qasper": "f1",
}


# ---------------- metrics (LongBench metrics.py) ----------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gt: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gt).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def retrieval(pred: str, gt: str) -> float:
    m = re.findall(r"Paragraph (\d+)", gt)
    if not m:
        return 0.0
    nums = re.findall(r"\d+", pred)
    if not nums:
        return 0.0
    return sum(1 for n in nums if n == m[0]) / len(nums)


def score(task: str, pred: str, answers: list[str]) -> float:
    if task in ("hotpotqa", "2wikimqa", "musique"):
        pred = pred.lstrip("\n").split("\n")[0]
    fn = f1 if METRIC[task] == "f1" else retrieval
    return max(fn(pred, a) for a in answers)


def selftest() -> int:
    ok = True
    ok &= abs(f1("The Eiffel Tower!", "eiffel tower") - 1.0) < 1e-9
    ok &= abs(f1("Paris France", "Paris") - 2 / 3) < 1e-9
    ok &= retrieval("Paragraph 12", "Paragraph 12") == 1.0
    ok &= retrieval("Paragraph 3", "Paragraph 12") == 0.0
    ok &= abs(retrieval("Paragraph 12 or 3", "Paragraph 12") - 0.5) < 1e-9
    ok &= score("hotpotqa", "\nYes\nmore", ["yes"]) == 1.0
    print("SELFTEST", "OK" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--policy", default="bf16")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--tasks", default="hotpotqa,2wikimqa,musique,passage_retrieval_en")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)  # item range start..start+n
    ap.add_argument("--max_len", type=int, default=31500)
    a = ap.parse_args()

    import torch
    from kvt import evals
    from kvt.cli import _load_calib, _spec
    from kvt.policy import KVPolicy
    from kvt.runtime import load_model

    spec = _spec(a.model)
    model, st = load_model(spec)
    store = _load_calib(spec, a.calib)
    tok = evals.Tok(spec)
    head, tail = evals.chat_wrap(spec, tok)
    tail_ids = tok.encode(tail)
    out_dir = OUT / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    t0 = time.time()
    for task in a.tasks.split(","):
        rows = [json.loads(l) for l in open(DATA / f"{task}.jsonl", encoding="utf-8")][
            a.start : a.start + a.n
        ]
        recs = []
        for i, r in enumerate(rows):
            body = tok.encode(PROMPT[task].format(context=r["context"], input=r["input"]))
            budget = a.max_len - len(head) - len(tail_ids)
            trunc = len(body) > budget
            if trunc:
                half = budget // 2
                body = body[:half] + body[-half:]
            ids = head + body + tail_ids
            st.policy = KVPolicy.parse(a.policy, spec, store) if a.policy != "bf16" else None
            res = evals.generate(model, ids, tok.eos, MAXGEN[task])
            pred = tok.decode([t for t in res.tokens if t not in tok.eos])
            s = score(task, pred, r["answers"])
            recs.append(
                {
                    "i": a.start + i,
                    "len": len(ids),
                    "trunc": trunc,
                    "pred": pred,
                    "answers": r["answers"],
                    "score": s,
                    "loop": evals.phrase_loop(res.tokens),
                }
            )
            torch.cuda.empty_cache()
        avg = 100 * sum(x["score"] for x in recs) / len(recs)
        summary[task] = {
            "score": avg,
            "n": len(recs),
            "loops": sum(x["loop"] for x in recs),
            "mean_len": sum(x["len"] for x in recs) / len(recs),
        }
        json.dump(
            {"policy": a.policy, "records": recs},
            open(out_dir / f"{task}_{a.tag}.json", "w"),
            ensure_ascii=False,
        )
        print(
            spec.name,
            a.tag,
            task,
            f"{avg:.2f}",
            "loops",
            summary[task]["loops"],
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    summary["avg"] = sum(v["score"] for v in summary.values()) / len(summary)
    json.dump(
        {"policy": a.policy, "summary": summary},
        open(out_dir / f"summary_{a.tag}.json", "w"),
        indent=1,
    )
    print("DONE", spec.name, a.tag, f"avg {summary['avg']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
