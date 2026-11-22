"""Postmean for SGLang Qwen2 / Qwen3 attention (B4-NV, SGLang amendment): subtract a static
post-RoPE key mean m from K before the KV cache write. Loaded only for postmean cells (this
directory on PYTHONPATH), inert unless SGLANG_KV_POSTMEAN=<file.kvmu>. SGLang files are not edited:
an import hook patches the classes after their module runs. Qwen2Attention.forward is replaced (same
body plus K - m); Qwen3Attention.forward_prepare_native (the CUDA path) is wrapped. The layer index
comes from the RadixAttention layer_id.
"""

import importlib.abc
import importlib.machinery
import os
import struct
import sys

_PATH = os.environ.get("SGLANG_KV_POSTMEAN")
_TARGETS = {
    "sglang.srt.models.qwen2": "Qwen2Attention",
    "sglang.srt.models.qwen3": "Qwen3Attention",
}


def _load(path: str):
    import torch

    raw = open(path, "rb").read()
    assert raw[:4] == b"KVMU", "SGLANG_KV_POSTMEAN: bad magic"
    version, n_layer, n_head_kv, head_dim = struct.unpack("<4i", raw[4:20])
    assert version == 1
    data = torch.frombuffer(bytearray(raw[20:]), dtype=torch.float32)
    assert data.numel() == n_layer * n_head_kv * head_dim
    return data.view(n_layer, n_head_kv * head_dim)


def _patch(mod, cls_name: str) -> None:
    import torch

    table = _load(_PATH)
    cls = getattr(mod, cls_name)
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        m = table[self.attn.layer_id]
        assert m.numel() == self.kv_size, "postmean hook: shape mismatch (tensor parallel > 1?)"
        # On the GPU from the start: a per-step host-to-device copy breaks CUDA graph capture.
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.register_buffer(
            "_kv_pm", m.to(device=dev, dtype=torch.get_default_dtype()), persistent=False
        )

    cls.__init__ = __init__
    if cls_name == "Qwen2Attention":

        def forward(self, positions, hidden_states, forward_batch):
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
            k = k - self._kv_pm.to(device=k.device, dtype=k.dtype)
            attn_output = self.attn(q, k, v, forward_batch)
            output, _ = self.o_proj(attn_output)
            return output

        cls.forward = forward
    else:
        orig_prep = cls.forward_prepare_native

        def forward_prepare_native(self, positions, hidden_states):
            q, k, v = orig_prep(self, positions, hidden_states)
            return q, k - self._kv_pm.to(device=k.device, dtype=k.dtype), v

        cls.forward_prepare_native = forward_prepare_native
    print(
        f"[postmean] {cls_name} patched with {_PATH} ({table.shape[0]} layers)",
        file=sys.stderr,
        flush=True,
    )


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch(module, _TARGETS[name])

        spec.loader.exec_module = exec_module
        return spec


if _PATH:
    sys.meta_path.insert(0, _Finder())
