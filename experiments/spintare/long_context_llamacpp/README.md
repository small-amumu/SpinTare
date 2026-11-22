# E2: repair vs context length in llama.cpp (up to 128K)

Source of every llama.cpp long-context number in the paper (§1, §3, §5):
static post-RoPE mean (postmean) KL rises 224x by 64K on R1-Distill-Qwen-7B, against 2.2x with rotary add-back;
1.9-5.0x by 32K on Qwen3-0.6B and Qwen2.5-7B, against 0.85-1.02x; on Llama-3.1-8B postmean is 1.5-1.6x worse than no
repair at every length up to 128K.

| File | Role |
|---|---|
| `e2a_run.sh` | phase A (X-0081 / X-0085): f16 / q4_0 / q4_0norot / q8_0 per length, no repair |
| `e2b_run.sh` | phase B (X-0086): q4_0 / pm / abmu / abbias per length |
| `run_all.sh` | the full grid, in the order of the original LSF jobs |
| `e2_analyze.py` | scores the registered predictions P1-P4 + activation check from the `run.log` files |

Frozen results: `results/spintare/long_context_llamacpp/` (`e2_final.json`; `run_logs/e2{a,b}/<tag>/run.log` as pulled
from the run machine; `kvmu/` = the mean files: `pm_*` static post-RoPE mean on 2048 tokens, `mu_*` pre-RoPE mean).
Re-derive on CPU:

```bash
python e2_analyze.py ../../../results/spintare/long_context_llamacpp/run_logs --json /tmp/e2.json
```

gives `e2_final.json` exactly (checked 2026-11-15).

Setup of the original runs (2026-11-06/07, one node with 2 x V100-16GB, LSF): K q4_0 with the engine's
Hadamard rotation, V q8_0, `-fa on -b 2048`, q8_0 weight GGUFs, f16 K reference per length, `32768 / L` chunks
(16384 scored tokens) up to 32K and one chunk beyond. 64K / 128K use `pg19_long` = `data/pg19_text.txt` +
`data/pg19_text2.txt` (md5 0a55cfc4; `run_all.sh` builds it), whose first part is the 32K text, so the long points are
a single block of the same PG19 text family.

Differences from the original runs:
- The original used one research tree (`llama.cpp-krot-llama`, learned-rotation switch off) for every cell. Here
  `pm` runs on the postmean tree (the KVMU mean file is converted to the KVPM header at run time) and the other cells
  on the SpinTare tree (switches off = upstream), with `LLAMA_KV_DEBIAS_MU_PATH=kadd` for `abmu`.
- Checked 2026-11-16 (VST3): on one RTX PRO 5000, this script and the two shipped trees print the same D as the
  research tree in all 41 cells of the 2K / 8K / 32K grid plus R1 at 64K and Llama-3.1-8B at 128K.
- Numbers depend on the GPU: the shipped ones are from V100. On the PRO 5000 single cells differ by 3 % (median) and
  up to 23 %; the ratios quoted above come out as 211x / 2.0x, 1.9-4.6x / 0.84-1.03x and 1.52-1.57x.
