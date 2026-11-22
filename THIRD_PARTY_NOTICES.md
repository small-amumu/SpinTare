# Third-party notices

SpinTare's own code is licensed under Apache-2.0 (see LICENSE and NOTICE). The following parts reproduce or follow
third-party work:

- **LongBench** (THUDM/LongBench, MIT License; the dataset is now hosted as zai-org/LongBench): the prompt templates, middle truncation and the answer
  normalisation / F1 / retrieval metrics in `spintare/tasks/longbench.py` are reproduced from the repository's `pred.py`
  and `metrics.py` (with the scoring, truncation and chat-template deviations listed at the top of that file). The LongBench task files (questions and answers) are not redistributed;
  `scripts/get_longbench.sh` downloads them from the original source, under the dataset's own terms. The
  texts in `data/` (except the PG19 ones) are concatenated LongBench contexts, included with attribution
  for calibration and evaluation only (`data/README.md`).
- **llama.cpp** (ggml-org/llama.cpp, MIT License): `spintare/sim.py` (used by `spintare/baselines/probe.py`)
  independently re-implements ggml's reference quantisation rules (q4_0, q4_1, q8_0) and the upstream K rotation
  size rule;
  `spintare/fix/permute.py` re-implements the q/k channel permutation of the GGUF converter. No llama.cpp source
  is copied into these files.
  `patches/llama.cpp/spintare-rotary-addback-exemption-gate.patch` and `postmean-static-mean.patch` apply to
  llama.cpp commit 3bb63cb and contain unchanged context lines from it: MIT License, Copyright (c) 2023-2026 The
  ggml authors; full text in `patches/llama.cpp/LICENSE-llama.cpp.txt`. The SpinTare changes in the patches are
  Apache-2.0. The patched upstream files remain under the MIT License.
- **vLLM** (vllm-project/vllm, Apache-2.0, Copyright the vLLM project contributors):
  `patches/vllm/spintare-addback.patch` modifies `vllm/v1/attention/backends/triton_attn.py`,
  `vllm/v1/attention/ops/triton_unified_attention.py` and `vllm/v1/attention/ops/int4_per_token_head.py` of commit
  5cb0fd265 (Apache-2.0); `integrations/vllm/` subclasses vLLM's Triton attention backend and
  `experiments/spintare/tasks/vllm_site/kv_postmean.py` copies the attention `forward` of vLLM's Qwen2 / Qwen3
  models with one added line (Apache-2.0).
  `experiments/spintare/detection_gate/kvprobe_pth.py` re-states the randomized Hadamard transform and the INT4
  rounding of `int4_per_token_head.py` in PyTorch.
  `spintare/baselines/attn_kl_vllm.py` re-states the `int4_per_token_head` KV format of
  `vllm/v1/attention/ops/int4_per_token_head.py` (commit 0c16eee3) in PyTorch for an offline simulation, and `spintare/baselines/probe.py` / `spintare/baselines/rules.py` implement the
  key-bias warning rule of vLLM PR #63081, as baselines.

The MIT License text applies to the reproduced LongBench prompts and metrics:

MIT License

Copyright (c) 2023 THU-KEG & Zhipu AI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
