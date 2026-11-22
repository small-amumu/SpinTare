"""Evaluation bar figures for the SpinTare paper in the style of VecInfer (ACL 2026, Figs. 6-8):
Times, grouped bars with thin black edges, hatching per method, dashed grid behind, framed legend on
top, green arrows with bold rotated gain labels. Colours keep this paper's earlier convention where
it applies (amber = before / no repair, teal = after / SpinTare); 16-bit is the pale reference bar.

Numbers are typed in from registered results (sec5_eval.tex tab:main, appendix tab:engine; vLLM port
VP3 / PFX1a /
PFX1b; SG1 engine gate ratios). Usage: python make_figs_eval_bars.py  -> fig_eval_*.pdf / .png (printed size).
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


# ------------------------------------------------------------------------------------------- 1
# recovery (figure*)
def fig_recovery() -> None:
    cells = [
        ("llama.cpp\nq4_0, rot.", 48.04, -46.83, -0.12),
        ("vLLM\nFP8, tensor", 48.85, -45.08, -1.36),
        ("vLLM\nFP8, tok.-head", 48.83, -45.50, -0.39),
        ("vLLM\nINT4, tok.-head", 48.85, -46.20, -1.34),
        ("SGLang\nNVFP4", 48.65, -30.43, -0.82),
    ]
    gsm = [
        ("R1-Qwen-7B\nRTX PRO 5000", 93.5, 0.0, 93.0),
        ("R1-Qwen-7B\nRTX 4080 S", 93.0, 0.0, 94.0),
        ("R1-Qwen-1.5B", 87.0, 0.0, 87.5),
    ]
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(6.3, 2.2), gridspec_kw=dict(width_ratios=[5, 3.1], wspace=0.2)
    )
    w = 0.26
    for ax, rows, ylab, top, title in (
        (a, cells, "LongBench score", 62, "Qwen2.5-7B, LongBench (8K prompts)"),
        (b, gsm, "GSM8K accuracy (%)", 118, "Long reasoning, llama.cpp q4_0"),
    ):
        for i, (_, ref, x1, x2) in enumerate(rows):
            if ax is a:
                bef, aft = ref + x1, ref + x2
            else:
                bef, aft = x1, x2
            ax.bar(i - w, ref, w, color=C_REF, **EDGE)
            ax.bar(i, bef, w, color=C_BEFORE, hatch="////", **EDGE)
            ax.bar(i + w, aft, w, color=C_AFTER, hatch="\\\\\\\\", **EDGE)
            if ax is b:  # the collapsed bar has zero height: say so
                ax.text(
                    i - 0.035,
                    0.03 * top,
                    "0.0",
                    ha="right",
                    va="bottom",
                    fontsize=5.6,
                    zorder=8,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"),
                )
            ax.text(i + w, aft + 0.012 * top, f"{aft:.1f}", ha="center", va="bottom", fontsize=5.6)
            if i:
                ax.axvline(i - 0.5, color="black", lw=0.5, zorder=2)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], linespacing=0.95)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_ylim(0, top)
        ax.set_title(title, pad=2.5)
        style(ax, ylab)
    legend(
        fig,
        [
            patch(C_REF, None, "16-bit cache"),
            patch(C_BEFORE, "////", "Low-bit K cache, no repair"),
            patch(C_AFTER, "\\\\\\\\", "SpinTare"),
        ],
        3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.175, top=0.8)
    save(fig, "fig_eval_recovery")


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


# ------------------------------------------------------------------------------------------ 3
# exemption gate (column)
def fig_gate() -> None:
    cols = ("2K decode", "2K prefill", "32K decode")
    data = (("Qwen2.5-7B", (0.787, 0.770, 0.826)), ("Qwen3-1.7B", (0.484, 0.496, 0.523)))
    fig, ax = plt.subplots(figsize=(3.03, 1.5))
    w = 0.34
    for i in range(3):
        for j, (_, vals) in enumerate(data):
            x = i + (j - 0.5) * w
            ax.bar(x, vals[i], w, color=(C_MID, C_AFTER)[j], hatch=("////", "\\\\\\\\")[j], **EDGE)
            ax.text(
                x,
                vals[i] / 2,
                f"{vals[i]:.2f}",
                ha="center",
                va="center",
                fontsize=5.8,
                fontweight="bold",
                color="white" if j else "black",
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.12", fc=(C_MID, C_AFTER)[j], ec="none"),
            )
        if i:
            ax.axvline(i - 0.5, color="black", lw=0.5, zorder=2)
    ax.axhline(1.0, color="black", lw=0.7, ls=(0, (4, 2)), zorder=4)
    ax.text(
        2.48,
        1.075,
        "rotary centring without the gate = 1",
        ha="right",
        va="bottom",
        fontsize=5.8,
        style="italic",
    )
    ax.set_xticks(range(3))
    ax.set_xticklabels(cols)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1.2)
    style(ax, "KL with gate / without")
    legend(
        ax,
        [patch(C_MID, "////", data[0][0]), patch(C_AFTER, "\\\\\\\\", data[1][0])],
        2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(left=0.135, right=0.99, bottom=0.135, top=0.855)
    save(fig, "fig_eval_gate_col")


if __name__ == "__main__":
    fig_recovery()
    fig_vllm()
    fig_gate()
