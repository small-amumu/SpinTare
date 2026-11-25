"""Data figures (two-column width), read from the registered aggregate files only.

fig_length_kl     : KL vs position, 5 collapsing models, arms none / static mean / rotary centring / + exemption gate (LC1, sim)
fig_residual      : (a) left-over offset ratio of the static mean vs position, 12 models; (b) predicted vs measured (LC1)
fig_sink_gate     : (a) gate / no-gate KL per model and position bucket (LC1, sim); (b) engine ratios (SG1, llama.cpp)
fig_decode_cost   : decode overhead of the add-back, table path vs position kernel (OV2, RTX 5090 D)
Usage: python make_figs.py   (writes PDF and PNG next to this script)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator, FormatStrFormatter, LogLocator, NullFormatter

RES = Path(__file__).resolve().parents[3] / "results" / "spintare"  # frozen inputs of the figures
# figures are written next to this file unless FIG_OUT names another directory
OUT = Path(os.environ.get("FIG_OUT", Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)
LC1 = json.loads((RES / "long_context_lc1" / "res_lc1_agg.json").read_text(encoding="utf-8"))
OV2 = json.loads((RES / "decode_overhead" / "res_ov2_time_agg.json").read_text(encoding="utf-8"))
SG1 = json.loads((RES / "decode_overhead" / "res_sg1_acc_agg.json").read_text(encoding="utf-8"))
SG2 = json.loads((RES / "decode_overhead" / "res_sg2_time_agg.json").read_text(encoding="utf-8"))

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
# Okabe-Ito, colour-blind safe; one colour per arm in every figure
C = {
    "none": "#7F7F7F",
    "static": "#E69F00",
    "rotary": "#0072B2",
    "gate": "#009E73",
    "red": "#D55E00",
    "kqbias": "#9DB9D6",
    "kqpos": "#0072B2",
}
NAME = {
    "q3_0p6b": "Qwen3-0.6B",
    "q25_1p5b": "Qwen2.5-1.5B",
    "llama3b": "Llama-3.2-3B",
    "q25_7b": "Qwen2.5-7B",
    "r1q_7b": "R1-Distill-Qwen-7B",
    "llama31_8b": "Llama-3.1-8B",
    "mistral7b": "Mistral-7B",
    "q3_8b": "Qwen3-8B",
    "granite8b": "Granite-3.1-8B",
    "olmo7b": "OLMo-2-7B",
    "yi6b": "Yi-1.5-6B",
    "ds7b": "DeepSeek-7B",
    "qwen25_7b": "Qwen2.5-7B",
    "qwen3_1p7b": "Qwen3-1.7B",
}
X = np.array([1024, 3584, 7680, 15872, 32256]) / 1024  # position buckets, in K tokens
CAL = 2.0  # calibration window, K tokens


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def calib(ax) -> None:
    ax.axvspan(0.8, CAL, color="#DDEBE6", zorder=0, lw=0)
    ax.set_xlim(0.8, 40)


# ---------------------------------------------------------------- fig_length_kl
def fig_length_kl() -> None:
    models = [m for m in ("q25_7b", "r1q_7b", "q25_1p5b", "q3_0p6b", "q3_8b") if m in LC1["models"]]
    fig, axes = plt.subplots(1, len(models), figsize=(6.3, 1.55), sharex=True)
    for ax, m in zip(axes, models):
        kl = LC1["models"][m]["kl"]
        calib(ax)
        for arm, key, lab in (
            ("none", "A_nr", "No repair"),
            ("static", "B2_nr", "Static mean"),
            ("rotary", "D_nr", "Rotary centring"),
            ("gate", "E_nr", "+ exemption gate"),
        ):
            ax.plot(X, kl[key], marker="o", color=C[arm], label=lab, zorder=3)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(NAME[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 32])
        ax.set_xticklabels(["1K", "4K", "32K"])
        ax.xaxis.set_minor_locator(FixedLocator([8, 16]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        lo, hi = ax.get_ylim()
        if (
            hi / lo < 20
        ):  # less than about one decade: label 1-2-5 ticks instead of powers of ten only
            t = [v for v in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10) if lo <= v <= hi]
            ax.yaxis.set_major_locator(FixedLocator(t))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.grid(True, which="major")
        ax.set_xlabel("Position")
    axes[0].set_ylabel("Output KL")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_length_kl")


# ---------------------------------------------------------------- fig_residual
def fig_residual() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 1.9), gridspec_kw={"width_ratios": [1.35, 1]})
    calib(a)
    rho = []
    for m, d in LC1["models"].items():
        r = d["rho"]["meas_B2"]
        rho.append(r)
        a.plot(X, r, color=C["static"], alpha=0.35, lw=0.9, ls="-" if d["in_range"] else "--")
    med = np.median(np.array(rho), axis=0)
    a.plot(X, med, color=C["red"], lw=2.0, marker="o", label="Median of 12 models", zorder=4)
    a.axhline(0.5, color="black", lw=0.7, ls=":")
    a.set_xscale("log", base=2)
    a.set_xticks([1, 4, 8, 16, 32])
    a.set_xticklabels(["1K", "4K", "8K", "16K", "32K"])
    a.set_xlabel("Position")
    a.set_ylabel("Left-over offset ratio")
    a.set_title("(a) Static mean: offset returns past calibration")
    a.text(
        1.05, 0.04, "calibration", fontsize=6, color="#3B7A65", transform=a.get_xaxis_transform()
    )
    a.grid(True)
    a.legend(loc="upper left", frameon=False)
    xs, ys = [], []
    for d in LC1["models"].values():
        xs += d["rho"]["pred_B2"]
        ys += d["rho"]["meas_B2"]
    xs, ys = np.array(xs), np.array(ys)
    b.scatter(
        xs, ys, s=9, color=C["static"], edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3
    )
    lim = [0, max(xs.max(), ys.max()) * 1.05]
    b.plot(lim, lim, color="black", lw=0.7, ls="--")
    n_ok = LC1["verdicts"]["all"]["L1_fit"]["n_spearman_ge_0.8"]
    b.text(
        0.04,
        0.93,
        f"Spearman $\\geq$ 0.8 in {n_ok}/12 models",
        transform=b.transAxes,
        fontsize=6.5,
        color=C["red"],
        fontweight="bold",
        va="top",
    )
    b.set_xlim(lim)
    b.set_ylim(lim)
    b.set_xlabel("Predicted (calibration + positions)")
    b.set_ylabel("Measured")
    b.set_title("(b) Predictable from frequencies")
    b.grid(True)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_residual")


# ---------------------------------------------------------------- fig_sink_gate
def fig_sink_gate() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.25), gridspec_kw={"width_ratios": [1.45, 1]})
    order = sorted(LC1["models"], key=lambda m: np.mean(LC1["models"][m]["L4_E_over_D_nr"]))
    M = np.array([LC1["models"][m]["L4_E_over_D_nr"] for m in order])
    im = a.imshow(
        M, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=0.1, vcenter=1.0, vmax=1.5)
    )
    a.set_yticks(range(len(order)))
    a.set_yticklabels(
        [NAME[m] + ("" if LC1["models"][m]["in_range"] else " †") for m in order], fontsize=6
    )
    a.set_xticks(range(5))
    a.set_xticklabels(["1K", "4K", "8K", "16K", "32K"])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5.2,
                color="white" if M[i, j] < 0.35 else "black",
            )
    a.set_xlabel("Position")
    a.set_title("(a) KL with / without the gate (sim.)")
    cb = fig.colorbar(im, ax=a, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    groups = [("S4", "2K decode"), ("S5_prefill", "2K prefill"), ("S6_32k", "32K decode")]
    w = 0.36
    for k, (mod, col) in enumerate((("qwen25_7b", C["rotary"]), ("qwen3_1p7b", C["gate"]))):
        recs = [SG1["models"][mod][g] for g, _ in groups]
        vals = [r["ratio_of_means"] for r in recs]
        # error bar: two-sided 95 % interval of the per-chunk relative change, drawn around the
        # ratio of means
        err = np.array(
            [
                [r["mean_rel"] - r["ci95_two_sided"][0], r["ci95_two_sided"][1] - r["mean_rel"]]
                for r in recs
            ]
        ).T
        xs = np.arange(len(groups)) + (k - 0.5) * w
        b.bar(
            xs,
            vals,
            w,
            yerr=err,
            capsize=1.5,
            error_kw={"elinewidth": 0.6},
            color=col,
            label=NAME[mod],
            edgecolor="white",
            linewidth=0.5,
        )
        for x, v, e in zip(xs, vals, err[1]):
            if np.isfinite(v):
                b.text(
                    x,
                    v + e + 0.015,
                    f"{v:.2f}$\\times$",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=C["red"],
                    fontweight="bold",
                )
    b.axhline(1.0, color="black", lw=0.8, ls="--")
    b.set_xticks(range(len(groups)))
    b.set_xticklabels([g[1] for g in groups])
    b.set_ylim(0, 1.32)
    b.set_ylabel("KL gate / no gate")
    b.set_title("(b) In llama.cpp (K4V4)")
    b.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    b.grid(True, axis="y")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_sink_gate")


# ---------------------------------------------------------------- fig_decode_cost
def fig_decode_cost() -> None:
    cfgs = [
        ("qwen25_7b", "k4v4"),
        ("qwen25_7b", "k4v8"),
        ("qwen3_1p7b", "k4v4"),
        ("qwen3_1p7b", "k4v8"),
    ]
    depths = ["0", "8192", "32768"]
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 1.75), sharey=True)
    for ax, (mod, fmt) in zip(axes, cfgs):
        m = OV2["models"][mod]
        w = 0.38
        for k, arm in enumerate(("kqbias", "kqpos")):
            name = f"{fmt}mu_{arm}"
            vals = [m["overhead_pct"][name][d] for d in depths]
            rounds = [m["overhead_per_round_pct"][name][d] for d in depths]
            err = np.array([[v - min(r), max(r) - v] for v, r in zip(vals, rounds)]).T
            xs = np.arange(3) + (k - 0.5) * w
            ax.bar(
                xs,
                vals,
                w,
                yerr=err,
                capsize=1.5,
                color=C[arm],
                edgecolor="white",
                linewidth=0.5,
                error_kw={"elinewidth": 0.6},
                label="Table (kqbias)" if arm == "kqbias" else "Position kernel",
            )
        # exemption gate: its paired cost over the position kernel, measured in its own session
        # (SG2), stacked on top
        g = SG2["models"][mod]["gate_cost_pp"]
        gv = [g[f"{fmt}@{d}"]["median"] for d in depths]
        gerr = np.array(
            [
                [v - min(g[f"{fmt}@{d}"]["rounds"]), max(g[f"{fmt}@{d}"]["rounds"]) - v]
                for v, d in zip(gv, depths)
            ]
        ).T
        base = [m["overhead_pct"][f"{fmt}mu_kqpos"][d] for d in depths]
        xs = np.arange(3) + 0.5 * w
        ax.bar(
            xs,
            gv,
            w,
            bottom=base,
            yerr=gerr,
            capsize=1.5,
            color=C["gate"],
            edgecolor="white",
            linewidth=0.5,
            hatch="////",
            error_kw={"elinewidth": 0.6},
            label="+ exemption gate",
        )
        for j, d in enumerate(depths):
            a_, b_ = m["overhead_pct"][f"{fmt}mu_kqbias"][d], m["overhead_pct"][f"{fmt}mu_kqpos"][d]
            ax.text(
                j + 0.05,
                a_ + 0.45,
                f"$-${(1 - b_ / a_) * 100:.0f}%",
                ha="right",
                fontsize=5.4,
                color=C["red"],
                fontweight="bold",
            )
            ax.text(
                j + 0.08,
                b_ + gv[j] + 0.45,
                f"+{gv[j]:.1f}",
                ha="left",
                fontsize=5.4,
                color="#00664A",
                fontweight="bold",
            )
        ax.set_xticks(range(3))
        ax.set_xticklabels(["0", "8K", "32K"])
        ax.set_title(f"{NAME[mod]} {fmt.upper()}", fontsize=7)
        ax.set_xlabel("Context depth")
        ax.grid(True, axis="y")
    axes[0].set_ylabel("Decode slowdown (%)")
    axes[0].set_ylim(0, 16.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_decode_cost")


if __name__ == "__main__":
    fig_length_kl()
    fig_residual()
    fig_sink_gate()
    fig_decode_cost()
    print("written to", OUT)
