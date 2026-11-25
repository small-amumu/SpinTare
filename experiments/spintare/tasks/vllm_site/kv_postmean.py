# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""kv_postmean: static postmean for the vLLM kernel build (5cb0fd265d): subtract a static post-RoPE
key mean m from K before the cache write, for Qwen2 (Qwen2.5) and Qwen3 attention.

Port of the earlier vLLM 0.30.1 hooks (Qwen2, Qwen3) to this build. Inert unless
VLLM_KV_POSTMEAN=<file.kvmu> is set. The file is the KVMU layout of the October B3 calibration:
float32 m[n_layer][n_head_kv * head_dim] of the post-RoPE key (after k_norm for Qwen3), the tensor
subtracted here. q . m is the same for every key of one query, so softmax is unchanged in exact
arithmetic; only the cached (quantized) K changes. vLLM is not edited: an import hook patches the
attention class after its module is executed, in every process that imports it. Each forward below
is this build's forward with one added line; the hook refuses to patch (raises) if the build's
forward source differs from the one copied here (md5 of its dedented source).
"""

import importlib.abc
import importlib.machinery
import os
import struct
import sys

_PATH = os.environ.get("VLLM_KV_POSTMEAN")
# module -> (class, md5 of the dedented forward source in 5cb0fd265d)
_TARGETS = {
    "vllm.model_executor.models.qwen2": ("Qwen2Attention", "7f823b896cc45e33880336b56594636a"),
    "vllm.model_executor.models.qwen3": ("Qwen3Attention", "116f239993f6fe024c372eba4a2d0a3c"),
}


def _load(path: str):
    import torch

    raw = open(path, "rb").read()
    if raw[:4] != b"KVMU":
        raise RuntimeError("VLLM_KV_POSTMEAN: bad magic")
    version, n_layer, n_head_kv, head_dim = struct.unpack("<4i", raw[4:20])
    if version != 1:
        raise RuntimeError("VLLM_KV_POSTMEAN: bad version")
    data = torch.frombuffer(bytearray(raw[20:]), dtype=torch.float32)
    if data.numel() != n_layer * n_head_kv * head_dim:
        raise RuntimeError("VLLM_KV_POSTMEAN: file size does not match its header")
    return data.view(n_layer, n_head_kv * head_dim)


def _forward_md5(path: str, cls_name: str) -> str:
    import ast
    import hashlib
    import textwrap

    src = open(path, encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name == "forward":
                    seg = ast.get_source_segment(src, m, padded=True)
                    return hashlib.md5(textwrap.dedent(seg).encode()).hexdigest()
    return "missing"


def _qwen2_forward(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    if self.qk_norm:
        total_tokens = q.shape[0]
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(total_tokens, self.q_size)
        k = k.view(total_tokens, self.kv_size)
    q, k = self.rotary_emb(positions, q, k)
    k = k - self._kv_pm.to(device=k.device, dtype=k.dtype)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


def _qwen3_forward(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
    q_by_head = self.q_norm(q_by_head)
    q = q_by_head.view(q.shape)
    k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
    k_by_head = self.k_norm(k_by_head)
    k = k_by_head.view(k.shape)
    q, k = self.rotary_emb(positions, q, k)
    k = k - self._kv_pm.to(device=k.device, dtype=k.dtype)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


_FORWARD = {"Qwen2Attention": _qwen2_forward, "Qwen3Attention": _qwen3_forward}


def _patch(mod, cls_name: str, want_md5: str) -> None:
    import torch
    from vllm.model_executor.models.utils import extract_layer_index

    got = _forward_md5(mod.__file__, cls_name)
    if got != want_md5:
        raise RuntimeError(
            f"postmean hook: {cls_name}.forward source md5 {got} != {want_md5}; not patching"
        )
    table = _load(_PATH)
    cls = getattr(mod, cls_name)
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        li = extract_layer_index(self.attn.layer_name)
        if li >= table.shape[0]:
            raise RuntimeError(
                f"postmean hook: layer {li} not in the file ({table.shape[0]} layers)"
            )
        m = table[li]
        if m.numel() != self.kv_size:
            raise RuntimeError(
                "postmean hook: mean size does not match the model's kv_size (tensor parallel > 1?)"
            )
        dev = next(
            self.qkv_proj.parameters()
        ).device  # on the GPU already: no host copy per layer and step
        self.register_buffer(
            "_kv_pm", m.to(device=dev, dtype=torch.get_default_dtype()), persistent=False
        )

    cls.__init__ = __init__
    cls.forward = _FORWARD[cls_name]
    print(
        f"[postmean] {cls_name} patched with {_PATH} ({table.shape[0]} layers) in pid {os.getpid()}",
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
        cls_name, want = _TARGETS[name]

        def exec_module(module):
            orig_exec(module)
            _patch(module, cls_name, want)

        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    """Install the import hook (called by rl_site/sitecustomize.py before anything imports vLLM
    models)."""
    if _PATH:
        sys.meta_path.insert(0, _Finder())
