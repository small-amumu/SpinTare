# E2: repair vs context length in llama.cpp (up to 128K)

Source of the context-length table, measured on one RTX PRO 5000 (2026-11-23):
static post-RoPE mean (postmean) KL rises 211x by 64K on R1-Distill-Qwen-7B, against 2.0x with rotary
add-back; 1.9-4.6x by 32K on Qwen3-0.6B and Qwen2.5-7B, against 0.84-1.03x; on Llama-3.1-8B postmean is
1.52-1.57x worse than no repair at every length up to 128K.

| File | Role |
|---|---|
| `e2a_run.sh` | phase A (X-0081 / X-0085): f16 / q4_0 / q4_0norot / q8_0 per length, no repair |
| `e2b_run.sh` | phase B (X-0086): q4_0 / pm / abmu / abbias per length |
| `run_all.sh` | the full grid, in the order of the original LSF jobs |
| `e2_analyze.py` | scores the registered predictions P1-P4 + activation check from the `run.log` files |

Frozen results: `experiments/results/spintare/long_context_llamacpp/`
- `e2_final.json`, `run_logs/e2{a,b}/<tag>/run.log`: the RTX PRO 5000 run. It ran the `run_all.sh` lines of the
  four table models (R1-Distill-Qwen-7B, Qwen2.5-7B, Qwen3-0.6B, Llama-3.1-8B) with these scripts unchanged, on
  the SpinTare and postmean trees (`experiments/scripts/build_llama_cpp.sh`, the CMake options of the research tree below),
  TF32 off (`GGML_CUDA_CUBLAS_NO_TF32=1`).
- `control_vst3.json`: its control. The 41 cells measured earlier on the same card (2026-11-16) reproduce
  exactly (identical printed D).
- `v100_replication/`: the original run of the full grid (2026-11-06/07, one node with 2 x V100-16GB, LSF; one
  research tree, `llama.cpp-krot-llama` with its learned-rotation switch off, for every cell), including the
  models outside the table (Mistral-7B with no injected offset, Qwen2.5-1.5B, Qwen2.5-14B).
- `kvmu/`: the mean files (`pm_*` static post-RoPE mean on 2048 tokens, `mu_*` pre-RoPE mean).

Re-derive on CPU:

```bash
python e2_analyze.py ../../../results/spintare/long_context_llamacpp/run_logs --json /tmp/e2.json
```

gives the content of `e2_final.json` (and `v100_replication/run_logs` that of
`v100_replication/e2_final.json`). The registered verdicts are the same on both GPUs (P1 fails, P2 holds, P3 fails, P4 holds, activation check holds).

Setup: K q4_0 with the engine's Hadamard rotation, V q8_0, `-fa on -b 2048`, q8_0 weight GGUFs, f16 K reference
per length, `32768 / L` chunks (16384 scored tokens) up to 32K and one chunk beyond. 64K / 128K use `pg19_long` =
`experiments/data/pg19_text.txt` + `experiments/data/pg19_text2.txt` (md5 0a55cfc4; `run_all.sh` builds it), whose first part is the 32K
text, so the long points are a single block of the same PG19 text family. `pm` runs on the postmean tree (the
KVMU mean file is converted to the KVPM header at run time), the other cells on the SpinTare tree (switches off =
upstream), with `LLAMA_KV_DEBIAS_MU_PATH=kadd` for `abmu`.

Cross-GPU check (V100 vs RTX PRO 5000, the 63 cells of the V100 table): without the 8-bit K row, cells differ
by 3.7 % (median) and up to 22.6 %, and every ratio above keeps its direction and size (V100: 224x / 2.2x,
1.9-5.0x / 0.85-1.02x, 1.51-1.62x). The 8-bit K row is 3-4.7x higher on the PRO 5000 (for example Llama-3.1-8B
0.00051 vs 0.00011). Both builds compile the same FlashAttention K/V pairs (q4_0-q4_0, q8_0-q8_0, f16-f16,
bf16-bf16), so the q8_0 cell runs llama.cpp's native q8_0 vector kernel while the f16 reference is converted to
f16 on both GPUs; the difference is in the GPUs' kernel numerics, not in the protocol.
