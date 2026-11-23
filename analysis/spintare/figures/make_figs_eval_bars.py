"""Evaluation bar figures for the SpinTare paper in the style of VecInfer (ACL 2026, Figs. 6-8):
Times, grouped bars with thin black edges, hatching per method, dashed grid behind, framed legend on
top, green arrows with bold rotated gain labels. Colours keep this paper's earlier convention where
it applies (amber = before / no repair, teal = after / SpinTare); 16-bit is the pale reference bar.

Only fig_eval_vllm (appendix fig:vllm) is a paper figure. Its numbers are typed in from the
registered vLLM port rounds (VP3 / PFX1a / PFX1b).
Usage: python make_figs_eval_bars.py  -> fig_eval_vllm.pdf / .png (printed size).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# figures are written next to this file unless FIG_OUT names another directory
OUT = Path(os.environ.get("FIG_OUT", Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.6,
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
    }
)
C_REF, C_BEFORE, C_AFTER = "#EEF2FB", "#F4C27A", "#3A9D8F"  # 16-bit / no repair / SpinTare
C_MID, C_DARK = "#AFC4E8", "#27457A"  # other baselines (blue ramp of the reference style)
GREEN = "#4E8B2F"
EDGE = dict(edgecolor="black", linewidth=0.55, zorder=3)


def style(ax, ylabel: str | None = None) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.45, color="#B5B5B5", zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2)
    ax.tick_params(pad=1.5)


def arrow(ax, x: float, y0: float, y1: float, label: str, dx: float = 0.0, fs: float = 6.0) -> None:
    """green double-headed arrow from y0 to y1 with a bold rotated label beside it."""
    ax.annotate(
        "",
        xy=(x, y1),
        xytext=(x, y0),
        zorder=6,
        arrowprops=dict(
            arrowstyle="<->", color=GREEN, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=5
        ),
    )
    ax.text(
        x + dx,
        (y0 + y1) / 2,
        label,
        rotation=90,
        ha="right",
        va="center",
        color=GREEN,
        fontsize=fs,
        fontweight="bold",
        zorder=7,
    )


def legend(fig_or_ax, handles, ncol: int, **kw) -> None:
    lg = fig_or_ax.legend(
        handles=handles,
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        handlelength=1.6,
        handleheight=0.8,
        columnspacing=1.2,
        borderpad=0.35,
        **kw,
    )
    lg.get_frame().set_linewidth(0.5)


def patch(color: str, hatch: str | None, label: str):
    return mpl.patches.Patch(
        facecolor=color, edgecolor="black", linewidth=0.55, hatch=hatch, label=label
    )


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------------------------- 2
# vLLM (figure*)
def fig_vllm() -> None:
    fig, (a, b, c) = plt.subplots(
        1, 3, figsize=(6.3, 1.85), gridspec_kw=dict(width_ratios=[3.5, 2, 2], wspace=0.34)
    )
    # (a) output KL, log scale: stock / PR #59674 (INT4 only) / SpinTare
    kl = [
        ("Qwen2.5\n1.5B", 7.43158, 0.0138, 0.01021),
        ("Qwen2.5\n7B", 6.54456, 0.0178, 0.01480),
        ("Llama-3.1\n8B", 0.01718, 0.0172, 0.00693),
        ("Qwen2.5\n1.5B", 1.64114, None, 0.00172),
        ("Qwen2.5\n7B", 4.95658, None, 0.00211),
        ("Llama-3.1\n8B", 0.00245, None, 0.00126),
    ]
    w = 0.26
    for i, (_, s, pr, st) in enumerate(kl):
        a.bar(i - w, s, w, color=C_BEFORE, hatch="////", **EDGE)
        if pr is not None:
            a.bar(i, pr, w, color=C_MID, hatch="xxxx", **EDGE)
        else:
            a.text(
                i,
                1.25e-3,
                "n/a",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=5.4,
                fontweight="bold",
            )
        a.bar(i + w, st, w, color=C_AFTER, hatch="\\\\\\\\", **EDGE)
        if i:
            a.axvline(i - 0.5, color="black", lw=0.5 if i != 3 else 0.9, zorder=2)
    a.set_yscale("log")
    a.set_ylim(1e-3, 30)
    a.set_xticks(range(len(kl)))
    a.set_xticklabels([r[0] for r in kl], linespacing=0.95, fontsize=5.8)
    for x_, t_ in ((1, "INT4, per token-head"), (4, "FP8")):
        a.text(
            x_,
            17,
            t_,
            ha="center",
            va="center",
            fontsize=6.2,
            style="italic",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"),
            zorder=8,
        )
    a.set_xlim(-0.5, len(kl) - 0.5)
    a.set_title("(a) Output KL to the 16-bit cache", pad=2.5)
    style(a, "KL (nats, log scale)")
    # (b) attention kernel time, eager, Llama-3.1-8B: stock / add-back before / after the tf32x3
    # kernel
    # PFX1a (rq check DONE), eager, 128 requests 512/64: stock, add-back v0 / torch write (PF1
    # configuration),
    # add-back v1 tf32x3 / fused write (the shipped v1f)
    tm = [("INT4", 600.9, 2530.6, 1135.3), ("FP8", 722.3, 2785.4, 1264.5)]
    for i, (_, s, v0, v1) in enumerate(tm):
        b.bar(i - w, s, w, color=C_BEFORE, hatch="////", **EDGE)
        b.bar(i, v0, w, color=C_MID, hatch="xxxx", **EDGE)
        b.bar(i + w, v1, w, color=C_AFTER, hatch="\\\\\\\\", **EDGE)
        b.text(i, v0 + 40, f"{v0 / s:.1f}$\\times$", ha="center", va="bottom", fontsize=5.6)
        b.text(
            i + w,
            60,
            f"{v1 / s:.2f}$\\times$",
            ha="center",
            va="bottom",
            fontsize=5.4,
            rotation=90,
            color="white",
            fontweight="bold",
            zorder=8,
        )
        if i:
            b.axvline(i - 0.5, color="black", lw=0.5, zorder=2)
    b.set_xticks(range(2))
    b.set_xticklabels([r[0] for r in tm])
    b.set_xlim(-0.5, 1.5)
    b.set_ylim(0, 3300)
    b.set_title("(b) Attention kernel time", pad=2.5)
    style(b, "Time (ms)")
    # (c) end-to-end throughput relative to stock, CUDA graphs, Llama-3.1-8B
    tp = [("INT4", 0.74, 0.90), ("FP8", 0.69, 0.90)]
    w2 = 0.3
    for i, (_, v0, v1) in enumerate(tp):
        c.bar(i - w2 / 2, v0, w2, color=C_MID, hatch="xxxx", **EDGE)
        c.bar(i + w2 / 2, v1, w2, color=C_AFTER, hatch="\\\\\\\\", **EDGE)
        for x, v in ((i - w2 / 2, v0), (i + w2 / 2, v1)):
            c.text(x, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=5.6)
        if i:
            c.axvline(i - 0.5, color="black", lw=0.5, zorder=2)
    c.axhline(1.0, color="black", lw=0.7, ls=(0, (4, 2)), zorder=4)
    c.text(1.47, 1.008, "stock", ha="right", va="bottom", fontsize=5.8, style="italic")
    c.set_xticks(range(2))
    c.set_xticklabels([r[0] for r in tp])
    c.set_xlim(-0.5, 1.5)
    c.set_ylim(0.5, 1.08)
    c.set_title("(c) Throughput vs. stock", pad=2.5)
    style(c, "Relative throughput")
    legend(
        fig,
        [
            patch(C_BEFORE, "////", "Stock vLLM low-bit K cache"),
            patch(C_MID, "xxxx", "PR #59674 (a); unoptimized add-back (b, c)"),
            patch(C_AFTER, "\\\\\\\\", "SpinTare"),
        ],
        4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.19, top=0.77)
    save(fig, "fig_eval_vllm")


if __name__ == "__main__":
    fig_vllm()
