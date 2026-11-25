# Exemption gate in simulation (C23)

Whole-model effect of the exemption gate: one teacher-forced forward pass in which every layer's keys are
replaced by an arm's reconstruction (values, queries and everything else exact), compared with the exact
pass by the next-token KL on positions 2048-4095 of a 4096-token window. Ten models (Qwen2.5-1.5B / 7B,
Qwen3-8B, Llama-3.1-8B, Llama-3.2-3B, Mistral-7B, Granite-3.1-8B, OLMo-2-7B, Yi-1.5-6B, DeepSeek-LLM-7B)
x eight windows from five texts; float32 weights, exact attention.

The pre-RoPE mean `mu` of each layer and KV head comes from another window; a row is exempt when
`z = 1 - k.mu / |mu|^2 >= 0.5`. Arms: exact (harness floor), raw Q4_0, centred Q4_0, centred with the
gated rows raw (the exemption gate), with only row 0 raw, with the gated rows in Q8_0, exact or fp16.

| File | Role |
|---|---|
| `c23_kl.py` | `real` (one model) and `agg` (all models) |
| `c22_q.py` | exact ggml Q4_0 / Q8_0 round trips; every run compares them bit for bit with `libggml-base` of a llama.cpp build on three layers and stops on a mismatch |
| `sim_utils.py` | exact attention and text windows (the same code the registered runs imported) |
| `run_c23.sh` | the ten models on one GPU |

Frozen aggregate: `results/spintare/exemption_gate_c23/res_c23_agg.json`, read by
`analysis/spintare/figures/make_eval_figs.py` (exemption-gate table). Panel (a) of `fig_sink_gate` comes
from LC1, not from this run. Texts: `data/{long,qasper,musique,mfqa,diag_long}_text.txt` via `KVREL_DATA`.
