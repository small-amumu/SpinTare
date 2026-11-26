# SmoothAttention baseline

SmoothAttention (QServe) scales the pre-RoPE keys by `lambda = max_t |K| ** 0.5` per layer, KV head and
channel (RoPE pairs tied) and folds `1 / lambda` into the key projection and `lambda` into the query
projection, so the model computes the same scores while the cached keys are smaller. Qwen3 normalises q and
k with one weight vector shared by all heads, so there `lambda` is one vector per layer (maximum over KV
heads). Calibration: the first 2048 tokens of `experiments/data/long_text.txt`, 16-bit forward.

| File | Role |
|---|---|
| `sa_fold.py` | calibrate, fold, write the folded checkpoint and gate it: in fp32 the fold must leave the output unchanged (mean KL <= 1e-6), and the bf16 files must stay within 3x the bf16 rounding noise of the unfolded model |
| `sa_apply.py` | re-create a folded checkpoint from a recorded `lambda` file and check that every output shard has the recorded md5 |
| `gguf_diff.py` | compare an original GGUF with the GGUF of its folded checkpoint tensor by tensor (only the q / k tensors may differ) |

`experiments/results/spintare/smoothattention/` holds the four recorded `lambda` files (Qwen2.5-7B-Instruct,
Qwen3-0.6B, Qwen3-1.7B, DeepSeek-R1-Distill-Qwen-7B) and their fold records. The released `lambda` files
drop the run machine's model path, so their file md5 differs from `lambda_md5` in the record;
`sa_apply.py` checks `lambda_tensor_md5` (the md5 of the tensor) instead.

```bash
python sa_apply.py --model $KVREL_MODELS/Qwen2.5-7B-Instruct --out $KVREL_MODELS/Qwen2.5-7B-Instruct-SA \
    --lambda ../../../results/spintare/smoothattention/Qwen2.5-7B-Instruct-SA.lambda.pt \
    --gate-json ../../../results/spintare/smoothattention/Qwen2.5-7B-Instruct-SA.gate.json \
    --result sa_apply_qwen25_7b.json
# llama.cpp: convert the folded checkpoint to a q8_0 GGUF, then
PYTHONPATH=$KVREL_LLAMA_CPP/gguf-py python gguf_diff.py orig.gguf sa.gguf qwen2 gguf_diff.json
```

`sa_fold.py` is kept byte for byte as it ran (`sa_apply.py` checks its md5), so it is not reformatted.
