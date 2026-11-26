# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""E3: token-level KL(bf16 || candidate) from vLLM's own full-vocabulary logits (PREDICTIONS.md
E2/E3 design).

One process (VLLM_ENABLE_V1_MULTIPROCESSING=0); prompt_logprobs=1 forces logits at every prompt
position and LogitsProcessor.forward is wrapped to keep their log-softmax. max_num_batched_tokens
512 (chunked prefill), so later positions attend to keys read back from the cache. Text: the first 4
x 2048 tokens of the calibration text, one prompt per chunk; scored rows predict tokens 1024..2047
of each chunk. Usage: python e3_kld.py --model <dir> --kv <dtype> --label <name> --refdir <dir>
[--ref] [--graph] --util <frac>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

import torch  # noqa: E402

CTX, FIRST, CHUNKS = 2048, 1024, 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kv", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--refdir", required=True)
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--util", type=float, required=True)
    ap.add_argument(
        "--text", default=str(Path(os.environ.get("KVREL_WORK", "work")) / "data/diag_text.txt")
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--graph", action="store_true", help="CUDA graphs and torch.compile (default: eager)"
    )
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import vllm.model_executor.layers.logits_processor as lpm

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(open(a.text, encoding="utf-8").read(), add_special_tokens=False).input_ids
    assert len(ids) >= CTX * CHUNKS, f"text has {len(ids)} tokens"
    bos = tok.bos_token_id if tok("a").input_ids[:1] == [tok.bos_token_id] else None
    captured: list[torch.Tensor] = []
    orig = lpm.LogitsProcessor.forward

    def forward(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if out is not None and out.dim() == 2 and out.shape[0] > 1:
            captured.append(out.float().log_softmax(-1).cpu())
        return out

    lpm.LogitsProcessor.forward = forward
    kw = dict(
        model=a.model,
        kv_cache_dtype=a.kv,
        enforce_eager=not a.graph,
        max_model_len=CTX + 64,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        enable_prefix_caching=False,
        gpu_memory_utilization=a.util,
        seed=0,
    )
    try:
        llm = LLM(attention_backend="TRITON_ATTN", **kw)
    except TypeError:
        llm = LLM(**kw)
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)
    refdir = Path(a.refdir)
    refdir.mkdir(parents=True, exist_ok=True)
    kls, same = [], []
    t0 = time.time()
    for c in range(CHUNKS):
        chunk = ids[c * CTX : (c + 1) * CTX]
        if bos is not None:
            chunk = [bos] + chunk[1:]
        captured.clear()
        llm.generate([TokensPrompt(prompt_token_ids=chunk)], sp, use_tqdm=False)
        lp = torch.cat(captured, 0)
        # row r predicts token r + 1; vLLM computes the last row too (it predicts the generated
        # token)
        assert lp.shape[0] in (CTX - 1, CTX), (
            f"captured {lp.shape[0]} rows, expected {CTX - 1} or {CTX}"
        )
        lp = lp[FIRST - 1 : CTX - 1]
        if a.ref:
            torch.save(lp, refdir / f"ref_c{c}.pt")
            continue
        ref = torch.load(refdir / f"ref_c{c}.pt")
        kl = (ref.exp() * (ref - lp)).sum(-1)
        kls.append(kl)
        same.append((ref.argmax(-1) == lp.argmax(-1)).float())
    res = {
        "label": a.label,
        "kv": a.kv,
        "model": a.model,
        "pm": os.environ.get("VLLM_KV_POSTMEAN"),
        "secs": round(time.time() - t0, 1),
    }
    if not a.ref:
        k = torch.cat(kls)
        res.update(
            D=float(k.mean()),
            D_chunks=[float(x.mean()) for x in kls],
            top1_same=float(torch.cat(same).mean()),
            n=int(k.numel()),
        )
    json.dump(res, open(a.out, "w"), indent=1)
    print("E3", json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
