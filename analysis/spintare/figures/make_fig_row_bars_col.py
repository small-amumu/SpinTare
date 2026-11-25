# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""fig_row_bars_col (single column, 3.0 in, included unscaled; plain-bar replacement for
fig_keys_scatter_col).

For the two KV heads of fig_keys_scatter_col (Qwen3-0.6B layer 5, Llama-3.1-8B layer 31; 200
post-RoPE keys, static mean over 512 keys): the norm and the relative 4-bit error of ordinary
(offset-carrying) rows and of the first (sink) row, before and after subtracting the static mean.
Ordinary rows: median over the 199 rows, whisker = quartiles. Error = ||Q(x) - x|| / ||k||, x = k
(before) or k - mean (after), Q = q4_0 blocks of 32 values, no Hadamard.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import make_figs_mechanism as m3

CASES = (
    ("Qwen3-0.6B_L5.npz", "Qwen3-0.6B, layer 5"),
    ("Meta-Llama-3.1-8B-Instruct_L31.npz", "Llama-3.1-8B, layer 31"),
)
C_BEFORE, C_AFTER = "#F4C27A", "#3A9D8F"  # amber / teal
H_BEFORE, H_AFTER = "////", "\\\\\\\\"  # diagonal hatching, opposite directions


def q40(x: np.ndarray) -> np.ndarray:
    y = x.reshape(x.shape[0], -1, 32)
    s = np.abs(y).max(-1, keepdims=True) / 8
    s = np.where(s > 0, s, 1.0)
    return (np.clip(np.round(y / s), -8, 7) * s).reshape(x.shape)


def load(f: str) -> dict[str, np.ndarray]:
    h = m3.best_head(f)
    d = np.load(m3.SRC / "keys3d" / f)
    k, m = d["keys"][h].astype(np.float64), d["mean"][h].astype(np.float64)
    y = k - m
    nb, na = np.linalg.norm(k, axis=1), np.linalg.norm(y, axis=1)
    return {
        "nb": nb,
        "na": na,
        "eb": 100 * np.linalg.norm(q40(k) - k, axis=1) / nb,
        "ea": 100 * np.linalg.norm(q40(y) - y, axis=1) / nb,
    }


def bars(ax, before: np.ndarray, after: np.ndarray, unit: str) -> None:
    """two groups (ordinary rows, sink row), two bars each (before, after)."""
    top = 0.0
    for g, (b, a) in enumerate(((before[1:], after[1:]), (before[:1], after[:1]))):
        for j, (v, c, hat) in enumerate(((b, C_BEFORE, H_BEFORE), (a, C_AFTER, H_AFTER))):
            x = g + (j - 0.5) * 0.38
            h = float(np.median(v))
            ax.bar(x, h, 0.34, color=c, edgecolor="black", lw=0.6, hatch=hat, zorder=3)
            if v.size > 1:
                q1, q3 = np.percentile(v, [25, 75])
                ax.plot([x, x], [q1, q3], color="black", lw=0.7, zorder=4)
                h = max(h, q3)
            top = max(top, h)
            m = float(np.median(v))
            ax.text(
                x,
                h,
                (f"{m:.1f}" if m < 10 else f"{m:.0f}") + unit,
                ha="center",
                va="bottom",
                fontsize=6.2,
                style="italic",
                zorder=5,
            )
    ax.set_ylim(0, 1.2 * top)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ordinary rows", "sink row"], style="italic")
    ax.grid(True, axis="y", zorder=0)
    ax.tick_params(labelsize=6, pad=1.5, length=2)
    for s in ("right", "top"):
        ax.spines[s].set_visible(False)


def main() -> None:
    plt.rcParams["hatch.linewidth"] = 0.5
    fig, ax = plt.subplots(2, 2, figsize=(3.0, 2.7))
    for i, (f, name) in enumerate(CASES):
        c = load(f)
        bars(ax[i, 0], c["nb"], c["na"], "")
        bars(ax[i, 1], c["eb"], c["ea"], "%")
        ax[i, 0].set_ylabel(name.split(",")[0], fontsize=6.5, labelpad=1.5, style="italic")
        print(
            name,
            {k: (round(float(np.median(v[1:])), 2), round(float(v[0]), 2)) for k, v in c.items()},
        )
    ax[0, 0].set_title("Row norm", fontsize=6.8, pad=3, style="italic")
    ax[0, 1].set_title("4-bit row error", fontsize=6.8, pad=3, style="italic")
    h = [
        plt.Rectangle(
            (0, 0), 1, 1, fc=C_BEFORE, ec="black", lw=0.5, hatch=H_BEFORE, label="before centring"
        ),
        plt.Rectangle(
            (0, 0), 1, 1, fc=C_AFTER, ec="black", lw=0.5, hatch=H_AFTER, label="after centring"
        ),
    ]
    fig.tight_layout(pad=0.2, w_pad=0.8, h_pad=0.7, rect=(0, 0, 1, 0.92))
    fig.legend(
        handles=h,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=6.2,
        handlelength=1.4,
        handleheight=0.9,
        handletextpad=0.4,
        columnspacing=1.2,
        prop={"style": "italic", "size": 6.2},
    )
    m3.save(fig, "fig_row_bars_col")


if __name__ == "__main__":
    main()
