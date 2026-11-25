# Setup

## 1. Python

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test,layout,figures]"     # torch >= 2.3, transformers >= 4.56, gguf, matplotlib
pytest -q tests                              # CPU unit tests
```

The vLLM experiments need vLLM at commit 5cb0fd265 with `integrations/patches/vllm/spintare-addback.patch` and the plugin
in `integrations/vllm/` (see its README). The reported runs used torch 2.13.0 and transformers 5.18.0.

## 2. Paths (`experiments/scripts/env.sh`)

No script contains a machine path. Every location is an environment variable with a default under the
repository (`work/`). Source `experiments/scripts/env.sh` once per shell; every experiment driver also sources it.

| Variable | Default | Holds |
|---|---|---|
| `KVREL_WORK` | `work/` | run logs, intermediate files, raw results |
| `KVREL_MODELS` | `$KVREL_WORK/models` | Hugging Face model directories, named as in the scripts (e.g. `Qwen2.5-7B-Instruct`) |
| `KVREL_GGUF` | `$KVREL_WORK/gguf` | GGUF conversions (`<tag>-q8_0.gguf`) |
| `KVREL_DATA` | `$KVREL_WORK/data` | texts (`experiments/data/*.txt` of this repository), LongBench |
| `KVREL_RESULTS` | `experiments/results/` | small frozen results (shipped) |
| `KVREL_LLAMA_CPP` | `$KVREL_WORK/llama.cpp` | SpinTare llama.cpp tree (`experiments/scripts/build_llama_cpp.sh spintare`) |
| `KVREL_LLAMA_CPP_BASE` | `$KVREL_WORK/llama.cpp-base` | unpatched upstream 3bb63cb |
| `KVREL_LLAMA_CPP_PM` | `$KVREL_WORK/llama.cpp-pm` | static post-RoPE mean tree (`LLAMA_KV_POSTMEAN`): static-mean baseline, `spintare check` |
| `KVREL_LLAMA_CPP_KROT` | `$KVREL_LLAMA_CPP` | tree whose `gguf-py` the probes use |
| `KVREL_PYTHON` | `python3` | interpreter used by the drivers |
| `KVREL_ENV_PREFIX` | empty | optional prefix whose `lib/` goes in front of `LD_LIBRARY_PATH` (newer libstdc++ for some vLLM builds) |
| `KVREL_GGML_LIB` | `$KVREL_LLAMA_CPP/build/bin/libggml-base.so` | ggml library for the bit-exact quantizer check of the C23 simulation |

To use the shipped texts: `mkdir -p work && ln -s ../data work/data` (or set `KVREL_DATA=$PWD/data`).
To run the aggregations on the shipped results: `experiments/scripts/stage_results.sh`.

## 3. llama.cpp

```bash
experiments/scripts/build_llama_cpp.sh all 120        # SpinTare tree + unpatched upstream + postmean tree, sm_120
experiments/scripts/build_llama_cpp.sh spintare       # only the SpinTare tree, native architecture
```

Each tree is a fresh clone of upstream at 3bb63cb owned by the script (it refuses to reset a directory it
did not create). Switches of the SpinTare tree, all read once at start-up:

| Variable | Effect |
|---|---|
| `LLAMA_KV_DEBIAS_MU=<file>` | rotary centring with the pre-RoPE mean of a KVMU file (`experiments/spintare/calib_and_diag/mu_calib.py`) |
| `LLAMA_KV_DEBIAS_MU_PATH=kqpos` | decode add-back computed from positions inside the flash-attention kernel (the reported path; `kqbias` = table, `kadd` = rebuild K) |
| `LLAMA_KV_MU_GATE=1` | exemption gate (needs `kqpos`; single KV stream) |

With no switch the tree follows the upstream path. GGUF conversion:
`python $KVREL_LLAMA_CPP/convert_hf_to_gguf.py $KVREL_MODELS/<model> --outtype q8_0 --outfile
$KVREL_GGUF/<tag>-q8_0.gguf`. Llama-family GGUFs permute q / k channels: reorder mean files with
`python -m spintare permute`.

## 4. Data

- `experiments/data/*.txt`: shipped calibration and evaluation texts (`experiments/data/README.md`).
- LongBench: `experiments/scripts/get_longbench.sh "$KVREL_WORK/data/data"` (pinned revision, sha256-checked).
- RULER: NVIDIA RULER `scripts/data/prepare.py` at commit c3f5e3b, 50 samples, seed 42, per model tokenizer.
- GSM8K (test) and MATH-500 for the reasoning cells, under `$KVREL_WORK/reason/`.

## 5. Hardware

The engine task tables and the decode-cost measurements ran on an NVIDIA RTX 5090 D and an RTX PRO
5000; the GPU of every table and figure is in `docs/reproduction.md`. Timing results are tied to the
card they were measured on.
