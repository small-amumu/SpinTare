# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Shared Hugging Face simulation of llama.cpp's quantised K cache.

All tools that simulate the engine use this module; the quantiser, the channel order and the
engine's own K rotation are defined once in spintare/ggml_ref.py and re-exported here:
  - quantisers follow ggml's reference rules on 32-element blocks (q4_0: d = signed max / -8,
    q = clamp(floor(x / d + 8.5), 0, 15); q4_1: asymmetric min / max; q8_0: absmax / 127, roundf);
  - channel order: llama.cpp's converter permutes the q / k projections of Llama-family models per
    head, so engine channel 2j + s holds Hugging Face channel s * d / 2 + j ("gguf" order); Qwen2 /
    Qwen3 keep the HF order;
  - basis: upstream llama.cpp rotates K (and Q) with an orthonormal Walsh-Hadamard transform of size
    n_rot_k (64, doubled while it divides the head dim) before the cache write;
    LLAMA_ATTN_ROT_DISABLE=1 turns it off. The simulator reproduces this upstream behaviour so that
    block statistics are taken in the engine's basis. Upstream also rotates a quantised V with a
    fixed size-64 transform (n_rot_v in src/llama-kv-cache.cpp); the simulator keeps V exact and
    does not model that.

The forward hook is a transformers AttentionInterface: it receives the post-RoPE query / key / value
of every layer and computes attention in float32 with the key optionally quantised in engine
coordinates.
"""

from __future__ import annotations

import itertools
import math
import warnings
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

FORMATS = ("q4_0", "q4_1", "q8_0")
LLAMA_ORDER_TYPES = ("llama", "mistral")  # architectures whose GGUF permutes q / k per head


# re-exported for the measure / baselines tools
from spintare.ggml_ref import gguf_perm, hadamard, n_rot_k, quant, steps  # noqa: E402,F401


@dataclass
class Engine:
    """Maps Hugging Face post-RoPE keys / queries to llama.cpp's cache coordinates and back."""

    head_dim: int
    order: str  # "hf" or "gguf"
    basis: str  # "identity" or "hadamard"
    device: torch.device
    perm: torch.Tensor | None = field(init=False, default=None)
    inv_perm: torch.Tensor | None = field(init=False, default=None)
    rot: torch.Tensor | None = field(init=False, default=None)
    nrot: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.order == "gguf":
            self.perm = gguf_perm(self.head_dim).to(self.device)
            self.inv_perm = torch.argsort(self.perm)
        self.nrot = n_rot_k(self.head_dim) if self.basis == "hadamard" else 0
        if self.basis == "hadamard" and not self.nrot:
            warnings.warn(
                f"head dim {self.head_dim} is not a multiple of 64: llama.cpp applies no K rotation"
            )
        if self.nrot:
            self.rot = hadamard(self.nrot, self.device)

    def to_engine(self, x: torch.Tensor) -> torch.Tensor:
        if self.perm is not None:
            x = x[..., self.perm]
        if self.rot is not None:
            s = x.shape
            x = (x.reshape(*s[:-1], s[-1] // self.nrot, self.nrot) @ self.rot).reshape(s)
        return x

    def from_engine(self, x: torch.Tensor) -> torch.Tensor:
        if self.rot is not None:
            s = x.shape
            x = (x.reshape(*s[:-1], s[-1] // self.nrot, self.nrot) @ self.rot.T).reshape(s)
        if self.inv_perm is not None:
            x = x[..., self.inv_perm]
        return x

    def quant_key(
        self, k: torch.Tensor, fmt: str, mean: torch.Tensor | None = None
    ) -> torch.Tensor:
        """k: [..., d] post-RoPE HF keys (float32). With `mean` (broadcastable, HF order) the static
        mean is removed before the cache write and added back after dequantisation; the engine
        instead leaves q . m out, which gives the same softmax (the LLAMA_KV_POSTMEAN repair)."""
        x = k - mean if mean is not None else k
        y = self.from_engine(quant(self.to_engine(x), fmt))
        return y + mean if mean is not None else y


def default_order(model_type: str) -> str:
    return "gguf" if model_type in LLAMA_ORDER_TYPES else "hf"


@dataclass
class Plan:
    """What the attention hook does in the next forward pass."""

    # None: exact K everywhere
    fmt: str | None = None
    # layers whose K is quantised (None: all)
    layers: set[int] | None = None
    # postmean repair: layer -> [kv_heads, d] (HF order)
    means: dict[int, torch.Tensor] | None = None
    # (layer, q, k, v, score_fn, scale); score_fn(q, k) -> masked scores
    capture: Callable[..., None] | None = None


_LIVE: weakref.WeakValueDictionary[str, Sim] = weakref.WeakValueDictionary()
_SERIAL = itertools.count()  # registration names are never reused


def _hook(name: str) -> Callable[..., tuple[torch.Tensor, None]]:
    """Attention function registered under `name`; holds only the name, so a Sim can be freed."""

    def attn(*args: Any, **kw: Any) -> tuple[torch.Tensor, None]:
        sim = _LIVE.get(name)
        if sim is None:
            raise RuntimeError(f"{name}: the Sim that owned this model was freed")
        return sim._attn(*args, **kw)

    return attn


class Sim:
    """A model plus an attention hook that quantises K per Plan. Usage:
        sim = Sim(model_dir, device, dtype); logp = sim.logprobs(ids, Plan(fmt="q4_0"))
    Only K is quantised; V stays exact. The engine check runs with a q8_0 V cache in both the
    reference and the test (and upstream rotates a quantised V too), so simulated D is close to, but
    not the same number as, the engine D that the check's thresholds refer to. Batch size 1 without
    padding: the hook rebuilds the causal (and sliding-window) mask and does not read the attention
    mask."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        dtype: str = "float32",
        order: str | None = None,
        basis: str = "hadamard",
    ) -> None:
        self.device = torch.device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        name = f"spintare_sim_{next(_SERIAL)}"  # one registration per instance
        _LIVE[name] = self
        AttentionInterface.register(name, _hook(name))
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=getattr(torch, dtype), attn_implementation=name
            )
            .to(self.device)
            .eval()
        )
        cfg = getattr(self.model.config, "text_config", None) or self.model.config
        self.cfg = cfg
        self.n_layer = int(cfg.num_hidden_layers)
        self.head_dim = int(
            getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        )
        assert self.head_dim % 32 == 0, (
            f"head dim {self.head_dim} is not a multiple of the 32-element block"
        )
        self.engine = Engine(
            self.head_dim, order or default_order(cfg.model_type), basis, self.device
        )
        self.plan = Plan()
        self.seen: set[int] = set()

    def encode(self, path: Path, n_tok: int) -> torch.Tensor:
        ids = self.tok(path.read_text(encoding="utf-8"), return_tensors="pt").input_ids[:, :n_tok]
        assert ids.shape[1] == n_tok, f"{path}: only {ids.shape[1]} tokens, need {n_tok}"
        return ids.to(self.device)

    def _attn(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float | None = None,
        dropout: float = 0.0,
        **kw: Any,
    ) -> tuple[torch.Tensor, None]:
        li = int(getattr(module, "layer_idx", len(self.seen)))
        self.seen.add(li)
        assert query.shape[0] == 1, "batch size 1 only (the causal mask is rebuilt here)"
        q, k, v = query.float(), key.float(), value.float()  # [1, H, L, d], [1, KVH, L, d]
        p = self.plan
        if p.capture is not None:
            scale = scaling if scaling is not None else 1.0 / math.sqrt(q.shape[-1])
            p.capture(
                li, q, k, v, lambda qx, kx: self._scores(qx, kx, scaling, kw, q.shape[2]), scale
            )
        if p.fmt is not None and (p.layers is None or li in p.layers):
            mean = p.means[li][None, :, None, :].to(k) if p.means is not None else None
            k = self.engine.quant_key(k, p.fmt, mean)
        h, kvh, n = q.shape[1], k.shape[1], q.shape[2]
        if h != kvh:
            k, v = k.repeat_interleave(h // kvh, 1), v.repeat_interleave(h // kvh, 1)
        s = self._scores(q, k, scaling, kw, n)
        out = torch.einsum("bhlm,bhmd->bhld", torch.softmax(s, -1), v)
        return out.transpose(1, 2).to(query.dtype), None

    @staticmethod
    def _scores(
        q: torch.Tensor, k: torch.Tensor, scaling: float | None, kw: dict, n: int
    ) -> torch.Tensor:
        """Masked scores of queries q [..., Lq, d] (the last Lq of n positions) against keys k [...,
        n, d], with the model's scaling, logit softcap and sliding window."""
        lq, d = q.shape[-2], q.shape[-1]
        s = (q @ k.transpose(-1, -2)) * (scaling if scaling is not None else 1.0 / math.sqrt(d))
        softcap = kw.get("softcap")
        if softcap:
            s = torch.tanh(s / softcap) * softcap
        qpos = torch.arange(n - lq, n, device=q.device)[:, None]
        kpos = torch.arange(n, device=q.device)[None, :]
        mask = kpos <= qpos
        win = kw.get("sliding_window")
        if win:
            mask = mask & (kpos > qpos - int(win))
        return s.masked_fill(~mask, float("-inf"))

    @torch.no_grad()
    def logprobs(self, ids: torch.Tensor, plan: Plan) -> torch.Tensor:
        self.plan, self.seen = plan, set()
        try:
            out = self.model(ids).logits[0].float().log_softmax(-1)
        finally:
            self.plan = Plan()
        n_seen = len(self.seen)
        assert n_seen == self.n_layer, f"hook saw {n_seen} of {self.n_layer} layers"
        return out

    @torch.no_grad()
    def means(self, ids: torch.Tensor) -> dict[int, torch.Tensor]:
        """Static post-RoPE key mean per layer and kv head (HF order), as `spintare calib` computes
        it."""
        acc: dict[int, torch.Tensor] = {}
        self.logprobs(
            ids, Plan(capture=lambda li, q, k, v, sf, sc: acc.__setitem__(li, k[0].mean(1)))
        )
        return acc


def kl_rows(ref: torch.Tensor, test: torch.Tensor, start: int) -> torch.Tensor:
    """Per-position KL(ref || test) for positions >= start (log-probability inputs)."""
    r, t = ref[start:], test[start:]
    return (r.exp() * (r - t)).sum(-1)
