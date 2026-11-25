# Architecture

```
HF checkpoint --> one 16-bit forward (experiments/spintare/detection_gate/kvprobe_unified.py)
                  |  post-RoPE keys per layer, in the engine's channel order (GGUF permutation) and basis
                  |  (Hadamard on / off), per 32-element block: b = log2(2 sqrt(3) sigma / step)
                  v
             configuration check: b_top2 = energy mean of the two worst layers
                  b_top2 <= tau_b = 2.028 -> centre;  otherwise -> plain 4-bit
                  |
                  v  centre
llama.cpp  (integrations/patches/llama.cpp/spintare-rotary-addback-exemption-gate.patch, upstream 3bb63cb)
 write:  z = 1 - k.mu / |mu|^2;  z < 0.5 -> cache Q(R(p)(k - mu)), flag 1;  else cache Q(R(p) k), flag 0
         (mu: pre-RoPE key mean per layer and KV head, KVMU file; fused write kernels GGML_OP_KV_MU_GATE)
 read:   score = q . K_hat + flag * q . R(p) mu   (decode: computed from positions inside the
         flash-attention vec kernel, kq-rope-bias.cu; prefill: R(p) mu added to the dequantized keys)
 check:  spintare check (output KL D against the 16-bit cache: allow if D < 0.1016, else retry with
         the static mean; alarm if the repaired D' > 0.3577)

vLLM  (integrations/patches/vllm/spintare-addback.patch, 5cb0fd265 + integrations/vllm, backend SPINTARE_AB)
 write:  same rule; an exempt row is marked by the sign of its per-token-head K scale
 read:   the Triton unified-attention kernel adds q . R(p) mu to every centred row's score in fp32
```

Static post-RoPE mean (the baseline): `LLAMA_KV_POSTMEAN` in the postmean tree
(`integrations/patches/llama.cpp/postmean-static-mean.patch`), the `SPINTARE` vLLM backend or the vLLM import hook
`experiments/spintare/tasks/vllm_site/kv_postmean.py`.

## Python package `src/spintare/`

| Module | Role |
|---|---|
| `ggml_ref.py` | the single definition of ggml's quantizers (`q4_0`, `q4_1`, `q8_0`), the K-rotation size rule, the Hadamard matrix and the GGUF q / k permutation; registered numeric variants kept by name |
| `sim.py` | simulation primitives on top of `ggml_ref` (hooks, plans, KL rows) |
| `check/` | `check` (engine check), `restore-engine` (layer-wise restoration in the engine) |
| `fix/` | `calib`, `calib-vl` (KVPM files), `permute`, `layout` |
| `measure/` | `offset`, `restore`, `inject`, `inject-verify` |
| `baselines/` | `probe`, `rules`, `attn-kl-vllm` (detection baselines) |
| `tasks/` | `longbench` (through llama-server), `paired-ci` |

## Experiments and analysis

`experiments/spintare/<study>/` holds the scripts of one study; `experiments/results/spintare/<study>/` its frozen
outputs; `experiments/analysis/spintare/` re-derives the tables and figures from `experiments/results/` on a CPU. The mapping from
every result and figure to its scripts is in `docs/reproduction.md`.
