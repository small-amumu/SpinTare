# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Single-column fig_length_kl (2026-11-13).

Same data and styling as make_figs.fig_length_kl, laid out 3 + 2 panels for one column (3.0 in wide,
included unscaled, so text stays >= 6 pt); the legend fills the sixth grid slot.
Output: fig_length_kl_col.pdf/.png.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, LogLocator, NullFormatter

import make_figs as m

ARMS = (
    ("none", "A_nr", "No repair"),
    ("static", "B2_nr", "Static mean"),
    ("rotary", "D_nr", "Rotary centring"),
    ("gate", "E_nr", "+ exemption gate"),
)


def panel(ax, model: str) -> None:
    kl = m.LC1["models"][model]["kl"]
    m.calib(ax)
    for arm, key, lab in ARMS:
        ax.plot(m.X, kl[key], marker="o", color=m.C[arm], label=lab, zorder=3, lw=1.1, ms=2.6)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(m.NAME[model], fontsize=7, pad=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 32])
    ax.set_xticklabels(["1K", "4K", "32K"])
    ax.xaxis.set_minor_locator(FixedLocator([8, 16]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    lo, hi = ax.get_ylim()
    if hi / lo < 20:  # less than about one decade: label 1-2-5 ticks instead of powers of ten only
        t = [v for v in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10) if lo <= v <= hi]
        ax.yaxis.set_major_locator(FixedLocator(t))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.tick_params(labelsize=6, pad=1.5)
    ax.grid(True, which="major")


def fig_length_kl_col() -> None:
    models = [
        x for x in ("q25_7b", "r1q_7b", "q25_1p5b", "q3_0p6b", "q3_8b") if x in m.LC1["models"]
    ]
    fig, axes = plt.subplots(2, 3, figsize=(3.0, 2.75))
    slots = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    for ax, model in zip(slots, models):
        panel(ax, model)
    for ax in axes[1, :2]:
        ax.set_xlabel("Position", fontsize=7, labelpad=1)
    for ax in axes[:, 0]:
        ax.set_ylabel("Output KL", fontsize=7, labelpad=1)
    leg = axes[1, 2]
    leg.axis("off")
    h, l = slots[0].get_legend_handles_labels()
    leg.legend(h, l, loc="center", frameon=False, fontsize=6.5, handlelength=1.6, borderaxespad=0)
    fig.tight_layout(w_pad=0.35, h_pad=0.5)
    m.save(fig, "fig_length_kl_col")


if __name__ == "__main__":
    fig_length_kl_col()
    print("written to", m.OUT)
