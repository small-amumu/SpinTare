# SPDX-License-Identifier: Apache-2.0
"""PF1 workload driver: offline vLLM engine, N random-token requests (fixed seed, fixed input /
output length, ignore_eos), wall-clock throughput, then an optional torch-profiler pass on a smaller
workload.

Arms (kv cache dtype, attention backend, add-back env):
  bf16        auto,                  vLLM default backend (recorded)
  bf16_triton auto,                  TRITON_ATTN
  fp8         fp8_per_token_head,    TRITON_ATTN
  int4        int4_per_token_head,   TRITON_ATTN
  fp8_ab      fp8_per_token_head,    SPINTARE_AB (SPINTARE_KVMU = --mu, SPINTARE_GATE = 1)
  int4_ab     int4_per_token_head,   SPINTARE_AB (same)
Usage: python -m spintare_vllm.pf_bench --model DIR --arm ARM --n 1000 --in-len 512 --out-len 256
--out F.json
       [--mu FILE] [--util 0.9] [--eager] [--profile-dir DIR --profile-n 128 --profile-out-len 64]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# in-process engine core for every arm: the plugin's impl counters and the attention impls are
# visible here, and the
# profiler trace holds the whole step in one process (a stated condition of the throughput numbers)
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ARMS = {
    "bf16": ("auto", None, False),
    "bf16_triton": ("auto", "TRITON_ATTN", False),
    "fp8": ("fp8_per_token_head", "TRITON_ATTN", False),
    "int4": ("int4_per_token_head", "TRITON_ATTN", False),
    "fp8_ab": ("fp8_per_token_head", "SPINTARE_AB", True),
    "int4_ab": ("int4_per_token_head", "SPINTARE_AB", True),
}


def prompts(n: int, length: int, vocab: int, seed: int) -> list[list[int]]:
    import torch

    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, vocab - 1000, (n, length), generator=g).tolist()


def environment(llm) -> dict:  # type: ignore[no-untyped-def]
    """what actually ran: module paths, versions, device, and the attention backend vLLM
    selected."""
    import subprocess

    import torch
    import triton
    import vllm

    drv = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    impls = set()
    try:
        ctx = llm.llm_engine.vllm_config.compilation_config.static_forward_context
        impls = {
            type(getattr(m, "impl", None)).__name__ for m in ctx.values() if hasattr(m, "impl")
        }
    except Exception:  # noqa: BLE001  (the engine may live in another process)
        pass
    smi = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    apps = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    return dict(
        torch_uuid=str(torch.cuda.get_device_properties(0).uuid),
        cards=smi,
        compute_apps=apps,
        vllm_file=vllm.__file__,
        vllm_version=getattr(vllm, "__version__", "?"),
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        triton=triton.__version__,
        driver=drv[0] if drv else "?",
        gpu=torch.cuda.get_device_name(0),
        cc=".".join(map(str, torch.cuda.get_device_capability(0))),
        visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        attn_impls=sorted(impls) or None,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--in-len", type=int, default=512)
    ap.add_argument("--out-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu", default="")
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument(
        "--num-gpu-blocks", type=int, default=0, help="fix the KV block count (0 = profiled)"
    )
    ap.add_argument("--profile-dir", default="")
    ap.add_argument("--profile-n", type=int, default=128)
    ap.add_argument("--profile-out-len", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kv, backend, ab = ARMS[a.arm]
    if ab:
        if not a.mu:
            raise SystemExit("add-back arms need --mu (the calibrated pre-RoPE mean file)")
        os.environ["SPINTARE_KVMU"] = a.mu
        os.environ["SPINTARE_GATE"] = "1"
    if os.environ.get("SPINTARE_NO_HADACORE") == "1":
        import vllm.v1.attention.ops.int4_per_token_head as i4

        i4._HADACORE_AVAILABLE = False
    import spintare_vllm
    from spintare_vllm import addback
    from vllm import LLM, SamplingParams
    from vllm.config import ProfilerConfig
    from vllm.inputs import TokensPrompt

    spintare_vllm.register()
    kw: dict = {}
    if a.profile_dir:
        kw["profiler_config"] = ProfilerConfig(
            profiler="torch", torch_profiler_dir=a.profile_dir, torch_profiler_with_stack=False
        )
    if backend:
        kw["attention_backend"] = backend
    if a.num_gpu_blocks:
        kw["num_gpu_blocks_override"] = a.num_gpu_blocks
    llm = LLM(
        model=a.model,
        kv_cache_dtype=kv,
        enforce_eager=a.eager,
        gpu_memory_utilization=a.util,
        max_model_len=a.in_len + a.out_len + 16,
        max_num_seqs=a.max_num_seqs,
        enable_prefix_caching=False,
        seed=a.seed,
        **kw,
    )
    if ab and addback.INSTANCES["n"] == 0:
        raise SystemExit("SPINTARE_AB requested but no add-back impl was built")
    cfg = llm.llm_engine.vllm_config
    vocab = cfg.model_config.get_vocab_size()
    sp = SamplingParams(
        max_tokens=a.out_len, min_tokens=a.out_len, ignore_eos=True, temperature=0.0
    )
    warm = [TokensPrompt(prompt_token_ids=p) for p in prompts(8, a.in_len, vocab, a.seed + 99)]
    llm.generate(
        warm, SamplingParams(max_tokens=8, ignore_eos=True, temperature=0.0), use_tqdm=False
    )
    ps = [TokensPrompt(prompt_token_ids=p) for p in prompts(a.n, a.in_len, vocab, a.seed)]
    t0 = time.perf_counter()
    outs = llm.generate(ps, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    dig = hashlib.sha1()
    for o in sorted(
        outs, key=lambda o: int(o.request_id) if str(o.request_id).isdigit() else str(o.request_id)
    ):
        dig.update(bytes(str(list(o.outputs[0].token_ids)), "ascii"))
    res = dict(
        arm=a.arm,
        model=a.model,
        kv=kv,
        backend_requested=backend,
        backend_used=str(getattr(cfg, "attention_config", None)),
        n=a.n,
        in_len=a.in_len,
        out_len=a.out_len,
        seed=a.seed,
        eager=a.eager,
        util=a.util,
        max_num_seqs=a.max_num_seqs,
        num_gpu_blocks_override=a.num_gpu_blocks,
        num_gpu_blocks=cfg.cache_config.num_gpu_blocks,
        block_size=cfg.cache_config.block_size,
        wall_s=dt,
        out_tokens=n_out,
        out_tok_s=n_out / dt,
        total_tok_s=(n_out + a.n * a.in_len) / dt,
        req_s=a.n / dt,
        all_full_length=all(len(o.outputs[0].token_ids) == a.out_len for o in outs),
        addback_impls=addback.INSTANCES["n"],
        mu=a.mu or None,
        env=environment(llm),
        mu_md5=hashlib.md5(Path(a.mu).read_bytes()).hexdigest() if a.mu else None,
        script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
        out_digest=dig.hexdigest(),
    )
    print("PF", json.dumps(res), flush=True)
    if a.profile_dir:
        pp = [
            TokensPrompt(prompt_token_ids=p)
            for p in prompts(a.profile_n, a.in_len, vocab, a.seed + 7)
        ]
        spp = SamplingParams(
            max_tokens=a.profile_out_len,
            min_tokens=a.profile_out_len,
            ignore_eos=True,
            temperature=0.0,
        )
        llm.start_profile(f"pf1_{a.arm}")
        t1 = time.perf_counter()
        llm.generate(pp, spp, use_tqdm=False)
        res["profile_wall_s"] = time.perf_counter() - t1
        llm.stop_profile()
        res.update(
            profile_dir=a.profile_dir, profile_n=a.profile_n, profile_out_len=a.profile_out_len
        )
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("PF DONE", a.arm, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
