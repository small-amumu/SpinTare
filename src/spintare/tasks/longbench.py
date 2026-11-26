# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Small-sample task evaluation: LongBench subtasks on llama.cpp's real quantised K cache
(llama-server client).

Prompt templates, middle truncation and scoring follow the LongBench repository (THUDM/LongBench,
pred.py / metrics.py):
  - qa_f1: token F1 after answer normalisation (lowercase, strip punctuation and articles, collapse
    whitespace), maximum over the reference answers;
  - retrieval: the "Paragraph N" number in the output compared with the reference, as a matching
    fraction.
Deviations from the official evaluation (both runs of a comparison are scored the same way, so the
paired difference is much less affected (though not guaranteed to be), but absolute scores are not
comparable with the leaderboard):
  - for hotpotqa, 2wikimqa and musique only the first line of the output is scored;
  - the middle truncation keeps the whole prompt, chat template included, within --max_len (so it
    fits the server context), instead of truncating the bare prompt before templating;
  - the built-in llama31 chat template has no system header; to use the model's own, pass
    --model <another name> --hf <model dir>.
Greedy decoding; inputs longer than --max_len are truncated in the middle. Requires a running
llama-server (e.g. -c 32768 -fa on -ctk <K type> -ctv q8_0); this script is only the client.
Data: the LongBench jsonl files (see setup/get_longbench.sh), one <task>.jsonl per task in --data.
Usage: python -m spintare longbench --model qwen25 --ctk q4_0 --port 8091 --n 50 --data DATA_DIR
       --out OUT_DIR
       python -m spintare longbench selftest
Output: <out>/<model>/<task>_lc_<tag>.json and summary_lc_<tag>.json (tag = --tag or --ctk).
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

PROMPT = {
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
        "question based on the given passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
        "question based on the given passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not "
        "output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the "
        "question based on the given passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which "
        "paragraph the abstract is from.\n\n{context}\n\nThe following is an "
        "abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is "
        'from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: '
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following "
        "question based on the above text, only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as "
        "you can, using a single phrase or sentence if possible. If the question cannot be "
        'answered based on the information in the article, write "unanswerable". If the question '
        'is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any '
        "explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as "
        "concisely as you can, using a single phrase or sentence if possible. If the question "
        'cannot be answered based on the information in the article, write "unanswerable". If the '
        'question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any '
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
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
    if task in ("hotpotqa", "2wikimqa", "musique"):  # deviation: first line only (see top)
        pred = pred.lstrip("\n").split("\n")[0]
    fn = f1 if METRIC[task] == "f1" else retrieval
    return max(fn(pred, a) for a in answers)


def phrase_loop(tokens: list[int], max_period: int = 8, min_reps: int = 3) -> bool:
    """True when the output ends in a period-p repetition (p <= max_period) repeated at least
    min_reps times."""
    for per in range(1, max_period + 1):
        need = per * min_reps
        if len(tokens) < need:
            break
        tail = tokens[-need:]
        if all(tail[i] == tail[i % per] for i in range(need)):
            return True
    return False


def selftest() -> int:
    ok = True
    ok &= abs(f1("The Eiffel Tower!", "eiffel tower") - 1.0) < 1e-9
    ok &= abs(f1("Paris France", "Paris") - 2 / 3) < 1e-9
    ok &= retrieval("Paragraph 12", "Paragraph 12") == 1.0
    ok &= retrieval("Paragraph 3", "Paragraph 12") == 0.0
    ok &= abs(retrieval("Paragraph 12 or 3", "Paragraph 12") - 0.5) < 1e-9
    ok &= score("hotpotqa", "\nYes\nmore", ["yes"]) == 1.0
    ok &= phrase_loop([1, 2, 1, 2, 1, 2]) and not phrase_loop([1, 2, 3, 4, 5, 6])
    print("SELFTEST", "OK" if ok else "FAIL")
    return 0 if ok else 1


LOCAL = urllib.request.build_opener(
    urllib.request.ProxyHandler({})
)  # never route loopback through a proxy


def post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with LOCAL.open(req, timeout=3600) as r:
        return json.loads(r.read())


def tok(port: int, text: str, special: bool) -> list[int]:
    return post(
        port, "/tokenize", {"content": text, "add_special": False, "parse_special": special}
    )["tokens"]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(prog="spintare longbench")
    ap.add_argument("--model", required=True, help="qwen25, qwen3, llama31, or any name with --hf")
    ap.add_argument(
        "--ctk", required=True, help="K cache type the server runs with (recorded in the output)"
    )
    ap.add_argument("--tag", default=None, help="output tag (default: --ctk)")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--tasks", default="hotpotqa,2wikimqa,musique,passage_retrieval_en")
    ap.add_argument("--max_len", type=int, default=31500)
    ap.add_argument(
        "--hf", default=None, help="HF model dir: take the chat head / tail from its template"
    )
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    tag = a.tag or a.ctk
    if a.model not in HEAD and not a.hf:
        ap.error(f"--model {a.model} has no built-in chat template: pass --hf <model dir>")
    if a.model in HEAD and a.hf:
        print(
            f"note: --model {a.model} uses its built-in chat template; --hf is ignored",
            file=sys.stderr,
        )
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
    head, tail = tok(a.port, h, True), tok(a.port, t, True)
    out = a.out / a.model
    out.mkdir(parents=True, exist_ok=True)
    summary, t0 = {}, time.time()
    for task in a.tasks.split(","):
        with open(a.data / f"{task}.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()][a.start : a.start + a.n]
        recs = []
        for i, r in enumerate(rows):
            body = tok(a.port, PROMPT[task].format(context=r["context"], input=r["input"]), False)
            budget = a.max_len - len(head) - len(tail)
            if budget < 2:
                sys.exit(f"--max_len {a.max_len} leaves no room for the prompt body")
            trunc = len(body) > budget
            if trunc:
                body = body[: budget // 2] + body[-(budget // 2) :]
            ids = head + body + tail
            req = {
                "prompt": ids,
                "n_predict": MAXGEN[task],
                "temperature": 0,
                "cache_prompt": False,
                "return_tokens": True,
            }
            res = post(a.port, "/completion", req)
            pred = res["content"]
            gen = res.get("tokens") or tok(a.port, pred, False)
            recs.append(
                {
                    "id": r.get("_id"),
                    "i": a.start + i,
                    "len": len(ids),
                    "trunc": trunc,
                    "pred": pred,
                    "answers": r["answers"],
                    "score": score(task, pred, r["answers"]),
                    "loop": phrase_loop(gen),
                }
            )
        if not recs:
            sys.exit(f"{task}: no questions in range --start {a.start} --n {a.n}")
        avg = 100 * sum(x["score"] for x in recs) / len(recs)
        summary[task] = {"score": avg, "n": len(recs), "loops": sum(x["loop"] for x in recs)}
        with open(out / f"{task}_lc_{tag}.json", "w", encoding="utf-8") as f:
            json.dump({"ctk": a.ctk, "records": recs}, f, ensure_ascii=False)
        print(
            a.model,
            tag,
            task,
            f"{avg:.2f}",
            "loops",
            summary[task]["loops"],
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    summary["avg"] = sum(v["score"] for v in summary.values()) / len(summary)
    with open(out / f"summary_lc_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"ctk": a.ctk, "summary": summary}, f, indent=1)
    print("DONE", a.model, tag, f"avg {summary['avg']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
