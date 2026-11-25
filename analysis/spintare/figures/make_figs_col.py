# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The SpinTare Authors
"""Single-column versions (3.0 in wide, included unscaled, text >= 6 pt),
2026-11-13.

fig_layer_bits_col   : panel (a) of fig_layer_bits (collapsing vs nearly lossless, relative depth).
fig_keys_scatter_col : Qwen3-0.6B L5 (sink row lengthened) and Llama-3.1-8B L31 (small offset), side
by side.
fig_decode_cost_col  : decode slowdown, K4V4 only, two models; table / position kernel / + exemption gate.
Same data and colours as make_figs.py and make_figs_mechanism.py.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import make_figs as mf
import make_figs_mechanism as m3


def fig_layer_bits_col() -> None:
    rows: dict[str, list[tuple[int, float]]] = defaultdict(list)
    label: dict[str, int] = {}
    for r in csv.DictReader(open(m3.SRC / "layers.csv", encoding="utf-8")):
        rows[r["tag"]].append((int(r["layer"]), float(r["b"])))
        label[r["tag"]] = int(r["worst_label"])
    grid = np.linspace(0, 1, 41)
    fig, a = plt.subplots(figsize=(3.0, 1.85))
    for lab, col, dark, name in (
        (0, m3.BLUE, m3.BLUE_D, "Nearly lossless"),
        (2, m3.ORANGE, m3.ORANGE_D, "Collapsing"),
    ):
        curves = []
        for t, v in rows.items():
            if label[t] != lab:
                continue
            v = sorted(v)
            x = np.array([i for i, _ in v]) / (len(v) - 1)
            y = np.array([bb for _, bb in v])
            a.plot(x, y, color=col, lw=0.35, alpha=0.28, zorder=2)
            curves.append(np.interp(grid, x, y))
        cv = np.array(curves)
        mu, sd = cv.mean(0), cv.std(0)
        a.fill_between(grid, cv.min(0), cv.max(0), color=col, alpha=0.10, lw=0, zorder=1)
        a.fill_between(grid, mu - sd, mu + sd, color=col, alpha=0.26, lw=0, zorder=1)
        a.plot(grid, mu, color=col, lw=3.5, alpha=0.18, solid_capstyle="round", zorder=3)
        a.plot(
            grid,
            mu,
            color=dark,
            lw=1.3,
            marker="o",
            markersize=2.4,
            markevery=4,
            mfc="white",
            mec=dark,
            mew=0.7,
            zorder=4,
            label=f"{name} (n={len(cv)})",
        )
    a.axhline(1.0, color=m3.BLACK, lw=0.6, ls=":", zorder=2)
    a.text(
        0.985, 1.06, "1 bit", transform=a.get_yaxis_transform(), ha="right", va="bottom", fontsize=6
    )
    a.set_xlabel("Relative layer depth", fontsize=7, labelpad=1)
    a.set_ylabel("Effective bits (q4_0)", fontsize=7, labelpad=1)
    a.tick_params(labelsize=6, pad=1.5)
    a.set_xlim(-0.02, 1.02)
    a.grid(True, zorder=0)
    hh, ll = a.get_legend_handles_labels()
    a.legend(
        hh[::-1],
        ll[::-1],
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        loc="lower right",
        fontsize=6.3,
        handlelength=1.8,
    )
    fig.tight_layout(pad=0.2)
    m3.save(fig, "fig_layer_bits_col")


def _scatter_panel(ax, f: str, title: str) -> None:
    c = m3.load_case(f, m3.best_head(f))
    m3.halo(ax, c["before"], m3.BLUE, m3.BLUE_D)
    m3.halo(ax, c["after"], m3.GREEN, m3.GREEN_D)
    m3.dots(ax, c["before"], m3.BLUE, m3.BLUE_D, "Keys")
    m3.dots(ax, c["after"], m3.GREEN, m3.GREEN_D, "Keys $-$ static mean")
    m3.offset_arrow(ax, c["mu"])
    m3.sink_arrow(ax, c["s0"], c["s1"])
    m3.star(ax, c["s0"], m3.BLACK, "Sink key")
    m3.star(ax, c["s1"], m3.ORANGE, "Sink key $-$ static mean")
    ax.axhline(0, color=m3.GRAY, lw=0.5, zorder=0)
    ax.axvline(0, color=m3.GRAY, lw=0.5, zorder=0)
    ax.grid(True, zorder=0)
    y = np.abs(np.r_[c["before"][:, 1], c["after"][:, 1]]).max()
    ax.set_ylim(-1.25 * y, 1.22 * y)
    ax.set_title(title, fontsize=6.8, pad=3)
    ax.set_xlabel("Offset direction", fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=6, pad=1.5)
    for s in ax.collections:
        if s.get_sizes().size and s.get_sizes()[0] == 6:
            s.set_sizes([3.5])


def fig_keys_scatter_col() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.0, 2.2))
    _scatter_panel(a, "Qwen3-0.6B_L5.npz", "(a) Qwen3-0.6B, layer 5")
    _scatter_panel(b, "Meta-Llama-3.1-8B-Instruct_L31.npz", "(b) Llama-3.1-8B, layer 31")
    a.set_ylabel("Top residual direction", fontsize=6.5, labelpad=1)
    h, l = a.get_legend_handles_labels()
    fig.tight_layout(pad=0.2, w_pad=0.6, rect=(0, 0, 1, 0.84))
    fig.legend(
        h,
        l,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=6,
        markerscale=1.1,
        handletextpad=0.2,
        columnspacing=0.8,
    )
    m3.save(fig, "fig_keys_scatter_col")


def fig_decode_cost_col() -> None:
    depths = ["0", "8192", "32768"]
    fig, axes = plt.subplots(1, 2, figsize=(3.0, 1.75), sharey=True)
    w = 0.38
    for ax, mod in zip(axes, ("qwen25_7b", "qwen3_1p7b")):
        o = mf.OV2["models"][mod]
        for k, arm in enumerate(("kqbias", "kqpos")):
            name = f"k4v4mu_{arm}"
            vals = [o["overhead_pct"][name][d] for d in depths]
            rounds = [o["overhead_per_round_pct"][name][d] for d in depths]
            err = np.array([[v - min(r), max(r) - v] for v, r in zip(vals, rounds)]).T
            ax.bar(
                np.arange(3) + (k - 0.5) * w,
                vals,
                w,
                yerr=err,
                capsize=1.2,
                color=mf.C[arm],
                edgecolor="white",
                linewidth=0.4,
                error_kw={"elinewidth": 0.5},
                label="Table bias" if arm == "kqbias" else "Position bias",
            )
        g = mf.SG2["models"][mod]["gate_cost_pp"]
        gv = [g[f"k4v4@{d}"]["median"] for d in depths]
        gerr = np.array(
            [
                [v - min(g[f"k4v4@{d}"]["rounds"]), max(g[f"k4v4@{d}"]["rounds"]) - v]
                for v, d in zip(gv, depths)
            ]
        ).T
        base = [o["overhead_pct"]["k4v4mu_kqpos"][d] for d in depths]
        ax.bar(
            np.arange(3) + 0.5 * w,
            gv,
            w,
            bottom=base,
            yerr=gerr,
            capsize=1.2,
            color=mf.C["gate"],
            edgecolor="white",
            linewidth=0.4,
            hatch="////",
            error_kw={"elinewidth": 0.5},
            label="+ exemption gate",
        )
        ax.set_xticks(range(3))
        ax.set_xticklabels(["0", "8K", "32K"])
        ax.set_title(f"{mf.NAME[mod]} K4V4", fontsize=6.8, pad=3)
        ax.set_xlabel("Context depth", fontsize=6.5, labelpad=1)
        ax.tick_params(labelsize=6, pad=1.5)
        ax.grid(True, axis="y")
    axes[0].set_ylabel("Decode slowdown (%)", fontsize=6.5, labelpad=1)
    axes[0].set_ylim(0, 13)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.1),
        fontsize=6,
        handlelength=1.4,
        columnspacing=0.9,
    )
    fig.tight_layout(pad=0.2, w_pad=0.4)
    mf.save(fig, "fig_decode_cost_col")


if __name__ == "__main__":
    fig_layer_bits_col()
    fig_keys_scatter_col()
    fig_decode_cost_col()
    print("written to", m3.OUT)
