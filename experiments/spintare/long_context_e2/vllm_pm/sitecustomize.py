# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Postmean for vLLM Qwen2 attention (B4-V8): subtract a static post-RoPE key mean m from K before
the cache write.

Loaded only for the postmean cells (this directory is put on PYTHONPATH) and inert unless
VLLM_KV_POSTMEAN=<file.kvmu> is set. The file is the KVMU layout written by scripts/pm_calib.py:
float32 m[n_layer][n_head_kv][head_dim]. q . m is the same for every key of one query, so softmax is
unchanged in exact arithmetic; only the cached K changes. The vLLM package itself is not edited: an
import hook patches Qwen2Attention after vllm.model_executor.models.qwen2 is executed, in every
process that imports it.
"""

import importlib.abc
import importlib.machinery
import os
import struct
import sys

_PATH = os.environ.get("VLLM_KV_POSTMEAN")
_TARGET = "vllm.model_executor.models.qwen2"


def _load(path: str):
    import torch

    raw = open(path, "rb").read()
    assert raw[:4] == b"KVMU", "VLLM_KV_POSTMEAN: bad magic"
    version, n_layer, n_head_kv, head_dim = struct.unpack("<4i", raw[4:20])
    assert version == 1, "VLLM_KV_POSTMEAN: bad version"
    data = torch.frombuffer(bytearray(raw[20:]), dtype=torch.float32)
    assert data.numel() == n_layer * n_head_kv * head_dim, "VLLM_KV_POSTMEAN: short file"
    return data.view(n_layer, n_head_kv * head_dim)


def _patch(mod) -> None:
    import torch
    from vllm.model_executor.models.utils import extract_layer_index

    table = _load(_PATH)
    cls = mod.Qwen2Attention
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        assert not self.qk_norm, "postmean hook: QK-norm graphs are not covered"
        li = extract_layer_index(self.attn.layer_name)
        m = table[li]
        assert m.numel() == self.kv_size, (
            "postmean hook: shape does not match the model (tensor parallel > 1?)"
        )
        self.register_buffer("_kv_pm", m.to(dtype=torch.get_default_dtype()), persistent=False)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k = k - self._kv_pm.to(device=k.device, dtype=k.dtype)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    cls.__init__ = __init__
    cls.forward = forward
    print(
        f"[postmean] Qwen2Attention patched with {_PATH} ({table.shape[0]} layers)",
        file=sys.stderr,
        flush=True,
    )


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if _PATH:
    sys.meta_path.insert(0, _Finder())
