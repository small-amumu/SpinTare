# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""SpinTare postmean as a vLLM attention backend (Triton backend + one subtraction at cache write).

The key of every new token has the static post-RoPE mean m[layer, kv_head] subtracted right before
it is quantized into the KV cache. The Triton unified attention reads every key, including the keys
written in the same step, from the cache, so all keys of one softmax are in the subtracted domain:
q . m is the same for every key of a query and cancels in the softmax. Only the cached K changes.

Enabled by SPINTARE_KVPM=<file>; without it the backend behaves exactly like TRITON_ATTN.
Refused (raises at start-up) where the cancellation does not hold or is not covered: learned
attention sinks (the sink logit does not shift by q . m), encoder / cross attention, chunked
attention with lookback, and the fused RoPE + cache-write path (it writes before the subtraction).
ALiBi and sliding windows are fine: they bias or mask by position, not by key content.
"""

from __future__ import annotations

import os
import sys

import torch

from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionImpl

from spintare_vllm import kvpm

ENV_FILE = "SPINTARE_KVPM"
BACKEND_NAME = "SPINTARE"
_TABLE: kvpm.MeanTable | None = None
_TABLE_LOADED = False
INSTANCES = {"n": 0}  # impls built in this process (the single-process KL pass checks it)


def get_table() -> kvpm.MeanTable | None:
    """The mean table named by SPINTARE_KVPM, loaded once per process (None when unset)."""
    global _TABLE, _TABLE_LOADED
    if not _TABLE_LOADED:
        path = os.environ.get(ENV_FILE)
        _TABLE = kvpm.load(path) if path else None
        _TABLE_LOADED = True
        if _TABLE is not None:
            print(
                f"[spintare] postmean table {_TABLE.path} md5 {_TABLE.md5} "
                f"({_TABLE.n_layer} layers, {_TABLE.n_head_kv} KV heads, d {_TABLE.head_dim})",
                file=sys.stderr,
                flush=True,
            )
    return _TABLE


class SpinTareTritonImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        INSTANCES["n"] += 1
        self._pm_table = get_table()
        self._pm: torch.Tensor | None = None
        self._pm_layer: int | None = None
        if self._pm_table is None:
            return
        if self.sinks is not None:
            raise ValueError("postmean: learned attention sinks do not shift by q . m")
        if self.attn_type != AttentionType.DECODER:
            raise ValueError(f"postmean: attention type {self.attn_type} is not supported")
        if self.chunk_lookback != -1:
            raise ValueError("postmean: chunked attention with lookback is not supported")
        if self._pm_table.head_dim != self.head_size:
            raise ValueError(
                f"postmean: table head_dim {self._pm_table.head_dim} != layer {self.head_size}"
            )

    def fused_rope_kvcache_supported(self) -> bool:
        # The fused path writes the cache inside the RoPE kernel, before any subtraction.
        if self._pm_table is not None:
            return False
        return super().fused_rope_kvcache_supported()

    def _build_pm(self, layer: AttentionLayer, key: torch.Tensor) -> torch.Tensor:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.models.utils import extract_layer_index

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("postmean: first cache write happened during graph capture")
        table = self._pm_table
        assert table is not None
        li = extract_layer_index(layer.layer_name)  # type: ignore[attr-defined]
        if not 0 <= li < table.n_layer:
            raise ValueError(f"postmean: layer {li} outside the table ({table.n_layer} layers)")
        heads = kvpm.heads_for_rank(
            table.n_head_kv,
            self.num_kv_heads,
            get_tensor_model_parallel_rank(),
            get_tensor_model_parallel_world_size(),
        )
        self._pm_layer = li
        return table.data[li, heads].to(device=key.device, dtype=key.dtype).contiguous()

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self._pm_table is not None:
            if self._pm is None:
                self._pm = self._build_pm(layer, key)
            key = key - self._pm  # [T, H_kv, D] - [H_kv, D]
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)


class SpinTareTritonBackend(TritonAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return BACKEND_NAME

    @staticmethod
    def get_impl_cls() -> type[SpinTareTritonImpl]:
        return SpinTareTritonImpl
