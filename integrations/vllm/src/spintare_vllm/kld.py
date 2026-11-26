# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""One KL pass in vLLM: token-level KL(reference || candidate) from vLLM's full-vocabulary logits.

Protocol (same as the llama.cpp check and E3): the first CHUNKS x CTX tokens of a fixed text, one
prompt per chunk; the scored rows predict tokens FIRST..CTX-1 of each chunk. Single process
(VLLM_ENABLE_V1_MULTIPROCESSING=0); prompt_logprobs=1 forces logits at every prompt position and
LogitsProcessor.forward is wrapped to keep their log-softmax. max_num_batched_tokens 512 (chunked
prefill), so later positions attend to keys read back from the cache.

A reference pass (--ref) stores the log-probabilities; a candidate pass reads them and writes
D = mean KL, per-chunk D and top-1 agreement. The postmean repair is on when SPINTARE_KVPM is set.
Usage: python -m spintare_vllm.kld --model DIR --kv DTYPE --refdir DIR --out F.json [--ref]
       [--backend SPINTARE] [--text FILE] [--util 0.85] [--graph]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

DEFAULT_CTX, DEFAULT_FIRST, DEFAULT_CHUNKS = 2048, 1024, 4


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="spintare_vllm.kld")
    ap.add_argument("--model", required=True)
    ap.add_argument("--kv", required=True, help="vLLM --kv-cache-dtype of this pass")
    ap.add_argument("--refdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", action="store_true", help="store the reference log-probabilities")
    ap.add_argument("--backend", default="SPINTARE")
    ap.add_argument("--text", required=True)
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--graph", action="store_true", help="CUDA graphs + torch.compile")
    ap.add_argument(
        "--ctx", type=int, default=DEFAULT_CTX, help="window length (default 2048, VP1 / VP2)"
    )
    ap.add_argument(
        "--first", type=int, default=DEFAULT_FIRST, help="first scored position of each window"
    )
    ap.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS, help="windows per text")
    a = ap.parse_args(argv)
    ctx, first, chunks = a.ctx, a.first, a.chunks

    import vllm.model_executor.layers.logits_processor as lpm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if os.environ.get("SPINTARE_NO_HADACORE") == "1":  # machine workaround (driver cannot JIT it)
        import vllm.v1.attention.ops.int4_per_token_head as i4

        i4._HADACORE_AVAILABLE = False
        print("[spintare] hadacore disabled (Triton RHT fallback)", file=sys.stderr)

    import spintare_vllm
    from spintare_vllm import addback as ab
    from spintare_vllm import backend as sb

    spintare_vllm.register()
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(Path(a.text).read_text(encoding="utf-8"), add_special_tokens=False).input_ids
    if len(ids) < ctx * chunks:
        raise SystemExit(f"text has {len(ids)} tokens, need {ctx * chunks}")
    bos = tok.bos_token_id if tok("a").input_ids[:1] == [tok.bos_token_id] else None
    captured: list[torch.Tensor] = []
    # prompt rows seen in this window; rows before first - 1 are dropped at capture (long windows)
    seen = [0]
    orig = lpm.LogitsProcessor.forward

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        out = orig(self, *args, **kwargs)
        if out is not None and out.dim() == 2 and out.shape[0] > 1:
            start = max(0, first - 1 - seen[0])
            if start < out.shape[0]:
                captured.append(out[start:].float().log_softmax(-1).cpu())
            seen[0] += out.shape[0]
        return out

    lpm.LogitsProcessor.forward = forward  # type: ignore[method-assign]
    llm = LLM(
        model=a.model,
        kv_cache_dtype=a.kv,
        attention_backend=a.backend,
        enforce_eager=not a.graph,
        max_model_len=ctx + 64,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        enable_prefix_caching=False,
        gpu_memory_utilization=a.util,
        seed=0,
    )
    table = sb.get_table()
    mf = ab.get_file()
    res: dict[str, object] = {
        "model": a.model,
        "kv": a.kv,
        "backend": a.backend,
        "graph": a.graph,
        "postmean_file": table.path if table else None,
        "postmean_md5": table.md5 if table else None,
        "spintare_impls": sb.INSTANCES["n"],
        "addback_file": mf.path if mf else None,
        "addback_md5": mf.md5 if mf else None,
        "addback_gate": ab.gate_mode() if mf else None,
        "addback_impls": ab.INSTANCES["n"],
        "text": a.text,
        "ctx": ctx,
        "first": first,
        "chunks": chunks,
        "script_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
    }
    if a.backend == sb.BACKEND_NAME and sb.INSTANCES["n"] == 0:
        raise SystemExit("backend SPINTARE requested but no SpinTare attention impl was built")
    if a.backend == ab.BACKEND_NAME and ab.INSTANCES["n"] == 0:
        raise SystemExit("backend SPINTARE_AB requested but no add-back impl was built")
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)
    refdir = Path(a.refdir)
    refdir.mkdir(parents=True, exist_ok=True)
    kls: list[torch.Tensor] = []
    same: list[torch.Tensor] = []
    t0 = time.time()
    for c in range(chunks):
        chunk = ids[c * ctx : (c + 1) * ctx]
        if bos is not None:
            chunk = [bos] + chunk[1:]
        captured.clear()
        seen[0] = 0
        llm.generate([TokensPrompt(prompt_token_ids=chunk)], sp, use_tqdm=False)
        lp = torch.cat(captured, 0)
        # row r predicts token r + 1; vLLM also computes the last row (the generated token); rows
        # before first - 1
        # were dropped at capture, so lp starts at row first - 1
        if seen[0] not in (ctx - 1, ctx):
            raise SystemExit(f"saw {seen[0]} rows, expected {ctx - 1} or {ctx}")
        lp = lp[: ctx - first]
        if a.ref:
            torch.save(lp, refdir / f"ref_c{c}.pt")
            continue
        ref = torch.load(refdir / f"ref_c{c}.pt")
        kls.append((ref.exp() * (ref - lp)).sum(-1))
        same.append((ref.argmax(-1) == lp.argmax(-1)).float())
    res["secs"] = round(time.time() - t0, 1)
    res["addback_writes"] = ab.INSTANCES["writes"]
    if not a.ref:
        k = torch.cat(kls)
        res.update(
            D=float(k.mean()),
            D_chunks=[float(x.mean()) for x in kls],
            top1_same=float(torch.cat(same).mean()),
            n=int(k.numel()),
        )
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("KLD", json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
