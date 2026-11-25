# SpinTare tools

> **Note:** the `calib` / postmean repair below is the static-mean variant (the static-mean baseline), kept as a tool. The repair reported for SpinTare is rotary centring with exact position-wise add-back plus the exemption gate, in `patches/llama.cpp/spintare-rotary-addback-exemption-gate.patch` (pre-RoPE mean from `experiments/spintare/calib_and_diag/mu_calib.py`, enabled with `LLAMA_KV_DEBIAS_MU` / `LLAMA_KV_MU_GATE`); see `docs/architecture.md`.


Tools to check, before deployment, whether a low-bit K cache (llama.cpp `q4_0`, `q4_1`, `q8_0`) damages a
model without any error, to measure where the damage comes from, and to apply the static-mean repair.

The engine tools (`check`, `restore-engine`) need llama.cpp built with `patches/llama.cpp/postmean-static-mean.patch`
(`scripts/build_llama_cpp.sh postmean`; `restore-engine` also runs on the SpinTare tree). Postmean
reads a KVPM file through `LLAMA_KV_POSTMEAN`; the patch hooks it into the qwen2, qwen3, llama and qwen2vl
graphs only, and other architectures ignore it (the check notices this: a D that does not change under postmean
is reported as "patch did not run"). Postmean does not support context shifting.

Install: `pip install -e .` (Python >= 3.10, PyTorch, transformers; `pip install -e .[layout]` adds the `gguf`
package for `layout`). Every tool runs as `python -m spintare <command>` (or `spintare <command>` after
installation); `python -m spintare --help` lists the commands. Tools that load a Hugging Face checkpoint never run
the checkpoint's own Python code unless `--trust-remote-code` is given (where the tool offers it).

## Check (llama.cpp)

| Command | What it does |
|---|---|
| `check` | The pre-deployment check. For each candidate K format (sorted smallest first), it measures D, the mean output KL against the 16-bit K cache (`llama-perplexity --kl-divergence`, 4 chunks of 2048 tokens, V `q8_0` in both runs). D < 0.1016 allows the format; otherwise it retries with the postmean repair; a repaired D' > 0.3577 raises an alarm. If nothing passes it falls back to `q8_0`, then to the 16-bit cache. The output records `LLAMA_ATTN_ROT_DISABLE` and the patch's `GGML_CUDA_*` kernel switches when they are set, as they change D. |
| `restore-engine` | Damage attribution in the engine: D with the K of selected layers kept in F16 (`LLAMA_KV_F16_LAYERS`, from the patch). A diagnostic of where the damage comes from, not a cache mode. |

## Repair

| Command | What it does |
|---|---|
| `calib`, `calib-vl` | Static post-RoPE key mean per layer and KV head, written as a KVPM file for `LLAMA_KV_POSTMEAN`. The engine subtracts the mean after RoPE, before the cache write, and changes nothing on the attention side: because the mean m is one static vector per layer and KV head, the term `q . m` is the same for every key of a query and cancels in the softmax. This holds without attention sinks, without a logit softcap and without context shifting, which is the case for the architectures the patch hooks. The Hugging Face simulation adds `q . m` back explicitly, which gives the same softmax. Removing a key mean whose score term cancels is a known idea (SageAttention smooths K by its mean over tokens). |
| `permute`, `layout` | Llama-family GGUFs permute the q / k channels per head. `calib` writes the Hugging Face order, so for Llama / Mistral the file must be reordered with `permute` (calib prints a note); a file in the wrong order is accepted by the engine silently and only shows up as a worse D'. `layout` reports which order a GGUF's K weights use. |

## Measurement (Hugging Face simulation of the engine's quantiser)

| Command | What it does |
|---|---|
| `offset` | Offset ratio of the keys per layer: shared mean versus content, over the whole head and per 32-element block. Measured on the pre-RoPE keys in the Hugging Face channel order, without the engine's rotation. |
| `restore` | Damage attribution in simulation. Every layer is quantised, then layers are restored one at a time and cumulatively. |
| `inject`, `inject-verify` | Writes a controlled shared key offset into a released checkpoint, then verifies the injected checkpoint. |

All simulated quantisers except `attn-kl-vllm` (vLLM's format) follow ggml's reference rules on 32-element blocks. They work in the engine's channel
order and basis, which includes the upstream llama.cpp K rotation (turn it off with `LLAMA_ATTN_ROT_DISABLE=1`).
The simulation quantises K only and keeps V exact, while the engine check runs with a `q8_0` V cache in both the
reference and the test; simulated D is therefore close to, but not the same number as, the engine D on which the
check's thresholds are defined. See `spintare/sim.py`.

## Baselines

| Command | What it does |
|---|---|
| `probe` | One 16-bit forward pass that computes every baseline feature: measured effective bits per block, key reconstruction error, simulated attention KL, attention-output cosine, and the key-bias statistic of vLLM PR #63081. Its quantisers are those of `spintare/sim.py` (ggml's rules, fp16 block scales). |
| `rules` | The baseline decision rules on those features: vLLM PR #63081, the offset-ratio rule, the mlx-quant-fidelity KL band, a SageAttention-style cosine check, and fitted thresholds for the remaining features. |
| `attn-kl-vllm` | Simulated attention KL for vLLM's `int4_per_token_head` format (the llama.cpp formats are covered by `probe`). |

## Small-sample task evaluation

| Command | What it does |
|---|---|
| `longbench` | LongBench subtasks through `llama-server`, following the LongBench prompts, truncation and metrics, with greedy decoding. The deviations from the official evaluation (first-line scoring for the three multi-hop QA tasks, truncation within the server context, the built-in Llama 3.1 template) are listed at the top of `spintare/tasks/longbench.py`; they apply to both runs of a comparison, so the paired difference is much less affected than the absolute scores. Get the data with `scripts/get_longbench.sh` (pinned dataset revision, sha256-checked; set `HF_ENDPOINT` for a mirror). |
| `paired-ci` | Paired bootstrap interval of the score difference against the 16-bit cache, paired by question id and resampled within each task (stratified), and the verdict: collapse (drop of 20 points or more, relative drop of 50% or more, or a repetition-loop rate of 20% or more), lossless (interval lower bound within the 3-point margin), partial drop (interval upper bound below the margin), or undecided. |

## Thresholds

| Constant | Used by | Origin |
|---|---|---|
| D < 0.1016 (allow) | `check`, `rules` (check) | Fitted outside this repository on labelled (model, K format) cells: an ordinal model of the task outcome (within the 3-point margin / partial drop / collapse) against the engine D. 0.1016 is the D at which the predicted probability of staying within the margin is 5/6. The labelled cells and the fit are not part of this repository, so the constant cannot be re-derived from it. |
| D' > 0.3577 (alarm) | `check`, `rules` (check) | Same fit (not in this repository): the D at which the predicted probability of collapse is 0.5. Applied only to the repaired D'. |
| key-bias statistic > 100 | `rules` (bias) | The warning rule of vLLM PR #63081: in the first attention layer with a key bias, the largest absolute bias on the slowest-rotating eighth of the rotary pairs. |
| r <= 3 allow, r > 15 alarm | `rules` (ratio) | Chosen by reading the offset-ratio scatter of the labelled cells; not fitted. |
| D <= 0.01 | `rules` (mlx) | The mean-KLD edge of the "good" quality band for KV caches in the README of mlx-quant-fidelity v0.10. Only the mean band is re-implemented; the tool's full verdict also uses the 99% KLD and the top-token flip rate. |
| out_cos >= 0.998 | `rules` (cos) | Chosen value for a SageAttention-style per-layer accuracy check; not a threshold published by SageAttention. |
| beff, akl, kmse | `rules` | No published threshold: fitted with `fit_threshold` (maximum balanced accuracy, ties to the strictest) on labelled cells and then applied unchanged; the labelled cells are not part of this repository. |

## References

- J. Zhang, J. Wei, H. Huang, P. Zhang, J. Zhu, J. Chen. SageAttention: Accurate 8-Bit Attention for
  Plug-and-play Inference Acceleration. ICLR 2025.
- vLLM pull request #63081 (key-bias warning for quantised KV caches), github.com/vllm-project/vllm.
- vLLM `int4_per_token_head` KV format: `vllm/v1/attention/ops/int4_per_token_head.py` at commit 0c16eee3.
- mlx-quant-fidelity v0.10, github.com/IonDen/mlx-quant-fidelity (KV quality bands in the README).
- llama.cpp K rotation: `src/llama-kv-cache.cpp` at commit 3bb63cb (ggml-org/llama.cpp PR #21038).
- Y. Bai et al. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. ACL 2024.

Models and datasets are not included; the scripts download or read them from paths you give.
