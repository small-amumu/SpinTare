"""RULER client for a vLLM OpenAI-compatible server: the vLLM port of lc_ruler.py (60782e3a,
llama.cpp E2 / CH4Y).

Same as lc_ruler.py:
- data: RULER scripts/data/prepare.py jsonl, <data_root>/<length>/<task>/validation.jsonl (fields index, input,
  outputs, answer_prefix); the first n rows;
- prompt: chat template head / tail cut from the HF tokenizer (Qwen3: enable_thinking=False) around `input`, then
  RULER's answer_prefix after the assistant header;
- generation lengths (GEN; niah_multikey_3 / niah_multiquery 128 as lc_ruler.py), greedy, one request at a time;
- scoring: string_match_all (niah, vt, cwe), string_match_part (qa); loop flag phrase_loop on the generated tokens;
- output layout: <out>/<length>_<task>.json {"ctk": label, "records": [...]} and summary_<label>.json, so the CH4X
  ruler_check.py (7b88e390) validates it unchanged.
vLLM-specific:
- token ids from the HF tokenizer (llama.cpp tokenized on its server); prompts are sent as token ids;
- repetition_penalty 1.0 and the other penalties 0 are sent explicitly, so the model's generation_config defaults
  (Qwen2.5: 1.05) do not apply (llama.cpp greedy has no penalty);
- max_tokens = min(GEN, max_model_len - prompt length): llama.cpp stops at a full context slot the same way;
- every record keeps prompt_tokens reported by the server (must equal len), finish_reason, max_tokens and the sum
  of the generated tokens' logprobs (gen_logprob; used by the postmean smoke to see that the cached
  K changed);
- resumable (issue log 113): each record is appended to <length>_<task>.partial.jsonl and fsynced before the next
  request; a rerun after a server crash reuses finished task files and partial records and runs only
  the missing items; "ITEM <task> <index>" is printed before each request, so the item in flight at
  a crash is in the log.
Usage: python lc_ruler_vllm.py --model <tag> --hf <dir> --label <arm> --port 8121 --data_root <dir>
--length 32768
       --tasks a,b --n 50 --max_model_len 32768
Output: $KVREL_WORK/results/ruler_vllm/{model}/{label}/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

GEN = {
    "niah_single_2": 128,
    "niah_multikey_2": 128,
    "niah_multivalue": 128,
    "niah_multikey_3": 128,
    "niah_multiquery": 128,
    "vt": 30,
    "cwe": 120,
    "qa_1": 32,
}


def string_match_all(pred: str, ref: list[str]) -> float:
    return sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)


def string_match_part(pred: str, ref: list[str]) -> float:
    return max(1.0 if r.lower() in pred.lower() else 0.0 for r in ref)


def phrase_loop(
    tokens: list[int], max_period: int = 8, min_reps: int = 3
) -> bool:  # as lc_lb.phrase_loop
    for per in range(1, max_period + 1):
        need = per * min_reps
        if len(tokens) < need:
            break
        tail = tokens[-need:]
        if all(tail[i] == tail[i % per] for i in range(need)):
            return True
    return False


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file, fsync, rename: a crash leaves the old file or the new one, never a cut
    one."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fp:
        fp.write(text)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


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
    ap.add_argument("--port", type=int, default=8121)
    ap.add_argument("--served", default="m")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--length", default="32768")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max_model_len", type=int, required=True)
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

    def enc(x: str) -> list[int]:
        return ht(x, add_special_tokens=False).input_ids

    head, tail = enc(h), enc(t)
    out = Path(os.environ.get("KVREL_WORK", "work")) / "results" / "ruler_vllm" / a.model / a.label
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    t0 = time.time()
    L = a.length
    for task in a.tasks.split(","):
        f = Path(a.data_root).expanduser() / L / task / "validation.jsonl"
        raw = f.read_bytes()
        rows = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()][: a.n]
        if len(rows) != a.n:
            raise SystemExit(f"{f}: {len(rows)} rows, want {a.n}")
        fn = string_match_part if task.startswith("qa") else string_match_all
        final = out / f"{L}_{task}.json"
        part = out / f"{L}_{task}.partial.jsonl"
        if (
            final.exists()
        ):  # finished in an earlier attempt of this cell (resume after a server crash)
            recs = json.loads(final.read_text(encoding="utf-8"))["records"]
            if [x["index"] for x in recs] != [r["index"] for r in rows]:
                raise SystemExit(f"{final}: index list differs from the data file")
            rows_todo: list[dict] = []
            part.unlink(
                missing_ok=True
            )  # left over if a crash came between the final write and the unlink
            print("RESUME", task, "complete file reused", flush=True)
        else:
            done: dict = {}
            if part.exists():
                lines = [x for x in part.read_text(encoding="utf-8").splitlines() if x.strip()]
                for k, line in enumerate(lines):
                    try:
                        x = json.loads(line)
                    except ValueError:
                        if (
                            k == len(lines) - 1
                        ):  # a last line cut by a crash mid-write: that item runs again
                            break
                        raise
                    done[x["index"]] = x
                write_atomic(
                    part, "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in done.values())
                )
                # (rewritten clean: a cut last line is dropped before appending again)
                print("RESUME", task, len(done), "records from the partial file", flush=True)
            rows_todo = [r for r in rows if r["index"] not in done]
        for r in rows_todo:
            print("ITEM", task, r["index"], flush=True)  # the item in flight, if the server dies
            ids = head + enc(r["input"]) + tail + enc(r.get("answer_prefix", ""))
            mt = min(GEN[task], a.max_model_len - len(ids))
            if mt < 1:
                raise SystemExit(
                    f"{task} index {r['index']}: prompt {len(ids)} leaves no room in {a.max_model_len}"
                )
            res = post(
                a.port,
                {
                    "model": a.served,
                    "prompt": ids,
                    "max_tokens": mt,
                    "temperature": 0,
                    "repetition_penalty": 1.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "return_token_ids": True,
                    "logprobs": 1,
                },
            )
            ch = res["choices"][0]
            pred = ch["text"]
            tok_src = "server" if ch.get("token_ids") is not None else "client"
            gen = ch["token_ids"] if tok_src == "server" else enc(pred)
            lps = (ch.get("logprobs") or {}).get("token_logprobs") or []
            pt = res.get("usage", {}).get("prompt_tokens")
            if pt != len(ids):
                raise SystemExit(
                    f"{task} index {r['index']}: server prompt_tokens {pt} != {len(ids)}"
                )
            rec = {
                "index": r["index"],
                "len": len(ids),
                "prompt_tokens": pt,
                "max_tokens": mt,
                "finish_reason": ch.get("finish_reason"),
                "pred": pred,
                "outputs": r["outputs"],
                "score": fn(pred, r["outputs"]),
                "loop": phrase_loop(gen),
                "n_gen": len(gen),
                "gen_logprob": sum(x for x in lps if x is not None),
                "tok_src": tok_src,
            }
            with open(
                part, "a", encoding="utf-8"
            ) as fp:  # one record per line, on disk before the next request
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fp.flush()
                os.fsync(fp.fileno())
            done[r["index"]] = rec
        if not final.exists():
            recs = [done[r["index"]] for r in rows]
            write_atomic(final, json.dumps({"ctk": a.label, "records": recs}, ensure_ascii=False))
            part.unlink(missing_ok=True)
        sc = 100 * sum(x["score"] for x in recs) / len(recs)
        key = f"{L}/{task}"
        summary[key] = {
            "score": round(sc, 2),
            "n": len(recs),
            "loops": sum(x["loop"] for x in recs),
            "max_len": max(x["len"] for x in recs),
            "capped": sum(x["max_tokens"] < GEN[task] for x in recs),
            "data_md5": hashlib.md5(raw).hexdigest(),
        }
        print(
            a.model,
            a.label,
            key,
            f"{sc:.2f}",
            "loops",
            summary[key]["loops"],
            "capped",
            summary[key]["capped"],
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    per_len = {
        L: round(sum(v["score"] for v in summary.values()) / len(summary), 2)
    }  # as lc_ruler.py
    write_atomic(
        out / f"summary_{a.label}.json",
        json.dumps({"ctk": a.label, "per_length": per_len, "summary": summary}, indent=1),
    )
    print(
        "DONE",
        a.model,
        "ruler_vllm",
        a.label,
        " ".join(f"{k.split('/')[1]}:{v['score']}" for k, v in summary.items()),
        "loops",
        sum(v["loops"] for v in summary.values()),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
