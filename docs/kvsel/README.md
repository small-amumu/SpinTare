# kvsel: probe -> KV-cache configuration

`kvsel` reads one 16-bit probe of a model (the SpinTare census probe,
`experiments/spintare/detection_gate/kvprobe_unified.py`) plus the model's `config.json` and the
deployment (engines, context length, batch, GPU memory), and returns a KV-cache configuration with the
reason and the evidence status of every rule it applied. It does not run or patch any kernel.

Per K format the configuration check decides plain, rotary centring with exact add-back and the
exemption gate, the static post-RoPE mean, or refuse; refused formats fall back to 8-bit K (llama.cpp)
or bf16 (vLLM). Among the safe formats that fit the GPU budget, the one with the fewest bytes is chosen.

Every reason carries a status: `evidence` (a measurement, or a fact read from the engine / patch source),
`check` (an engine run is needed before serving, e.g. the engine-KL check of `spintare check`) or
`to_test` (no measurement yet).

## Use

```bash
python -m kvsel probe --model $M --order hf --out probe.json            # one 16-bit forward
python -m kvsel select probe.json --config $M/config.json \
    --engine llama.cpp:spintare --ctx 32768 --gpu-kv-gib 6
```

Engines: `llama.cpp`, `vllm`; suffix `:spintare` (add-back + gate kernel) or `:postmean` (static-mean
patch). Without `--config` the selector does not know NoPE layers or the RoPE scaling type: add-back
kernels then need `--model-type` and `--rope-type` (`none` = no scaling).

## Rules

| Id | Rule |
|---|---|
| F0 | `b_top2` in engine coordinates (llama.cpp Hadamard basis, GGUF channel order for Llama) |
| F1 | `b_top2 > tau_b` -> run the format plain, no centring |
| F2 | flagged and a covered add-back kernel -> rotary centring with exact add-back; engine-KL check before serving |
| F3 | exemption gate always on with centring |
| F4 | static post-RoPE mean only inside its 2K calibration window |
| F5 | unflagged formats are not centred even if centring lowers KL |
| F6 | fallback: 8-bit K (llama.cpp), bf16 (vLLM) |
| F7 | V stays q8_0 on the llama.cpp flat path |
| F8 | vLLM INT4 / FP8 decisions use the q4_0 cell as a proxy (no fitted threshold for 128-wide blocks) |
| F9 | decode / throughput cost of the repair is reported with the configuration (llama.cpp: add-back + gate, `results/spintare/decode_overhead/res_sg2_time_agg.json`) |
| C1 | the safe configuration with the fewest GPU bytes that fits the budget |
| C6 | partial / interleaved RoPE and NoPE layers: the add-back formula holds with the model's own RoPE (X5), but no kernel implements it |
| C12 | engine coordinates come from the engine build, not from the user |
| C13 | add-back / static-mean coverage per (engine, model graph, RoPE scaling); see `kvsel/evidence.py` `KERNEL_*` |

Kernel coverage (C13), read from the patch sources:

| Engine, repair | Model graphs | RoPE scaling |
|---|---|---|
| llama.cpp add-back + gate | qwen2, qwen3, gpt_neox (whole-head rotary only) | none, linear (YaRN trips the gated decode path) |
| llama.cpp static mean | llama (incl. Mistral), qwen2, qwen3 | - (only inside 2K) |
| vLLM add-back plugin | llama, mistral, qwen2, qwen3 (what `calib-mu` accepts) | none, linear, llama3; refuses partial rotation and attention_scaling != 1 |

## Offline check

```bash
python analysis/kvsel/offline_check.py        # CPU only; writes results/kvsel/offline_check/
pytest -q tests/kvsel
```

It re-scores the selector's flat decision on the census cells with the frozen `tau_b` (fitting set:
in-sample BA 0.983, leave-one-family-out 0.967; held-out PRED-P 20 and N13 12 cells all correct; serving
policy with 8-bit fallback: mean task difference -0.68 at 6.00 bits) and prints the selector's choice
for every probed model under three deployments. Input: `results/kvsel/inputs/model_configs.json` (the
models' `config.json`, `_name_or_path` removed).

## X5: add-back on partial / interleaved RoPE and NoPE layers

`experiments/kvsel/x5/`: CPU-only check of rotary centring with exact add-back on GLM-4-9B (partial 64 / 128,
interleaved), Phi-4-mini (partial 96 / 128, LongRoPE scaling), SmolLM3-3B (NoPE every fourth layer) and
Qwen2.5-1.5B (reference, whole-head NeoX); layer-wise attention KL, K only, one 2048-token text. With the
add-back computed from the model's own RoPE, gated centring lowered the `q4_0` attention KL on every
layer of every model (median ratio to plain `q4_0`: GLM 0.33, Phi 0.41, SmolLM3 RoPE layers 0.29 / NoPE
layers 0.90, Qwen2.5 0.37). An add-back that gets one RoPE factor wrong (rotary width, pair layout,
attention scaling, rotating mu on a NoPE layer) was about 3x to more than 1000x worse than no repair.
Results: `results/kvsel/x5/` (model paths shown as `<models>/...`).

The committed `x5_addback_check.py` and `x5_run.sh` are the files of the run (md5 `a046e836...` and
`a27c6394...`, recorded in `results/kvsel/x5/run.log` and in each result's `script_md5`) plus the
repository's two license header lines; `x5_addback_check.py` also has one docstring word changed. In the
repository `ggml_ref.py` re-exports `spintare/ggml_ref.py`, which is byte-identical to the copy the run
imported.
