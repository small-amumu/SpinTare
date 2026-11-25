# SpinTare for vLLM

A vLLM general plugin (`spintare_vllm`) plus a patch to vLLM's Triton attention kernels. Base: vLLM commit
`5cb0fd265d65a8ab9a39517ddb33a9996fe45398` (0.34.0rc1.dev468), the build of the reported vLLM cells.

| Backend (`--attention-backend`) | Switch | What it does |
|---|---|---|
| `SPINTARE_AB` | `SPINTARE_KVMU=<file.kvmr>` | rotary centring with exact add-back and the exemption gate. The write path caches `R(p)(k - mu)` unless the gate exempts the row (`z = 1 - k.mu / |mu|^2 >= 0.5`), which it marks in the sign of the row's K scale; the patched kernel adds `q . R(p) mu` back to every centred row's score in fp32. Needs the patch and a per-token-head cache (`int4_per_token_head`, `int8_per_token_head`, `fp8_per_token_head`), NeoX RoPE over the whole head, no ALiBi |
| `SPINTARE` | `SPINTARE_KVPM=<file>` | static post-RoPE mean subtracted at the cache write (works on unpatched vLLM) |

`SPINTARE_GATE` = `1` (default) | `centre` (no gate) | `exempt` (every row uncentred: must reproduce the
plain cache; test mode). Without their files both backends are the stock Triton backend.

## Install

```bash
git clone https://github.com/vllm-project/vllm && cd vllm
git checkout 5cb0fd265d65a8ab9a39517ddb33a9996fe45398
git apply <this repo>/patches/vllm/spintare-addback.patch
pip install -e .                      # or the precompiled-wheel route of vLLM's docs
pip install -e <this repo>/integrations/vllm
```

The patch touches `vllm/v1/attention/backends/triton_attn.py`, `ops/triton_unified_attention.py` and
`ops/int4_per_token_head.py`; with no add-back arguments every kernel takes its unchanged path.

## Use

```bash
python -m spintare_vllm calib-mu $MODEL data/long_text.txt qwen25_7b.kvmr --T 2048   # pre-RoPE mean
SPINTARE_KVMU=qwen25_7b.kvmr SPINTARE_GATE=1 vllm serve $MODEL \
    --attention-backend SPINTARE_AB --kv-cache-dtype int4_per_token_head --enforce-eager
python -m spintare_vllm check --model $MODEL --text data/long_text.txt --no-postmean  # check
python -m spintare_vllm kld --model $MODEL --text data/long_text.txt --kv auto --refdir ref \
    --out ref.json --ref                                    # 16-bit reference
python -m spintare_vllm kld --model $MODEL --text data/long_text.txt --kv int4_per_token_head \
    --refdir ref --out d.json                               # KL against it
```

The plugin registers itself through the `vllm.general_plugins` entry point. The task cells start
the server through `experiments/spintare/tasks/run_vllm_cell.sh`, which also sets the kernel variant of
the registered runs (`SPINTARE_AB_WRITE=fused`, `SPINTARE_AB_VARIANT=1`).

## Tests

```bash
cd integrations/vllm
python -m pytest -q tests                  # CPU: add-back algebra, NeoX tables, KVMR / KVPM readers
python -m pytest -q tests_gpu              # CUDA + patched vLLM: op-level check of the kernels
```

`tests_gpu/test_addback_kernel.py` writes keys with a large shared offset and a few sink-like rows through
the backend's own write path, reads them through the patched kernel and compares with a float64 reference;
the exempt-everything mode must equal `mu = 0` bit for bit.
