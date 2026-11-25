"""RU section: RULER (NVIDIA/RULER, arXiv 2404.06654) on llama.cpp with real K-cache quantization.

Data: jsonl produced by RULER's scripts/data/prepare.py (--model_template_type base) per model
tokenizer, laid out as
  <data_root>/<length>/<task>/validation.jsonl, fields: index, input, outputs, length,
  answer_prefix.
Prompt: chat template (head/tail from lc_lb.py or the HF tokenizer) around `input`; RULER's
answer_prefix is appended after the assistant header, as RULER does for chat models. Greedy
/completion, cache_prompt false, loop detection as lc_lb. Scoring: RULER eval/synthetic/constants.py
-- string_match_all (niah, vt, cwe), string_match_part (qa). Usage: python lc_ruler.py --model
qwen25 --ctk q4_0 --port 8093 --data_root ~/ruler_data/qwen25_7b --par 4 Output:
$KVREL_WORK/results/ruler/{model}/{ctk}/{length}_{task}.json and summary_{ctk}.json
"""

from __future__ import annotations
import os

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lc_lb import HEAD, TAIL, phrase_loop, post, tok

TASKS = ["niah_single_2", "niah_multikey_2", "niah_multivalue", "vt", "cwe", "qa_1"]
GEN = {
    "niah_single_2": 128,
    "niah_multikey_2": 128,
    "niah_multivalue": 128,
    "vt": 30,
    "cwe": 120,
    "qa_1": 32,
    "niah_multikey_3": 128,
    "niah_multiquery": 128,
}  # the four reported tasks; RULER niah value length


def string_match_all(pred: str, ref: list[str]) -> float:
    return sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)


def string_match_part(pred: str, ref: list[str]) -> float:
    return max(1.0 if r.lower() in pred.lower() else 0.0 for r in ref)


def chat_parts(model: str, hf: str | None) -> tuple[str, str]:
    if model in HEAD:
        return HEAD[model], TAIL[model]
    from transformers import AutoTokenizer

    ht = AutoTokenizer.from_pretrained(hf)
    msgs = [{"role": "user", "content": "@@SPLIT@@"}]
    try:
        s = ht.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        s = ht.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    h, t = s.split("@@SPLIT@@")
    return h, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctk", required=True)
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--lengths", default="4096,8192,16384,32768")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--par", type=int, default=1)
    ap.add_argument("--hf", default=None)
    a = ap.parse_args()
    h, t = chat_parts(a.model, a.hf)
    head, tail = tok(a.port, h, True), tok(a.port, t, True)
    out = Path(os.environ.get("KVREL_WORK", "work")) / "results" / "ruler" / a.model / a.ctk
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    t0 = time.time()
    for L in a.lengths.split(","):
        for task in a.tasks.split(","):
            f = Path(a.data_root).expanduser() / L / task / "validation.jsonl"
            rows = [json.loads(x) for x in open(f, encoding="utf-8")][: a.n]

            def one(r: dict) -> dict:
                ids = (
                    head
                    + tok(a.port, r["input"], False)
                    + tail
                    + tok(a.port, r.get("answer_prefix", ""), False)
                )
                res = post(
                    a.port,
                    "/completion",
                    {
                        "prompt": ids,
                        "n_predict": GEN[task],
                        "temperature": 0,
                        "top_k": 1,
                        "cache_prompt": False,
                        "return_tokens": True,
                    },
                )
                pred = res["content"]
                gen = res.get("tokens") or tok(a.port, pred, False)
                fn = string_match_part if task.startswith("qa") else string_match_all
                return {
                    "index": r["index"],
                    "len": len(ids),
                    "pred": pred,
                    "outputs": r["outputs"],
                    "score": fn(pred, r["outputs"]),
                    "loop": phrase_loop(gen),
                }

            with ThreadPoolExecutor(a.par) as ex:
                recs = list(ex.map(one, rows))
            sc = 100 * sum(x["score"] for x in recs) / len(recs)
            key = f"{L}/{task}"
            summary[key] = {
                "score": round(sc, 2),
                "n": len(recs),
                "loops": sum(x["loop"] for x in recs),
                "max_len": max(x["len"] for x in recs),
            }
            json.dump(
                {"ctk": a.ctk, "records": recs},
                open(out / f"{L}_{task}.json", "w"),
                ensure_ascii=False,
            )
            print(
                a.model,
                a.ctk,
                key,
                f"{sc:.2f}",
                "loops",
                summary[key]["loops"],
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
    per_len = {}
    for L in a.lengths.split(","):
        v = [summary[f"{L}/{x}"]["score"] for x in a.tasks.split(",")]
        per_len[L] = round(sum(v) / len(v), 2)
    json.dump(
        {"ctk": a.ctk, "per_length": per_len, "summary": summary},
        open(out / f"summary_{a.ctk}.json", "w"),
        indent=1,
    )
    loops = sum(v["loops"] for v in summary.values())
    print(
        "DONE",
        a.model,
        "ruler",
        a.ctk,
        " ".join(f"{k}:{v}" for k, v in per_len.items()),
        "loops",
        loops,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
