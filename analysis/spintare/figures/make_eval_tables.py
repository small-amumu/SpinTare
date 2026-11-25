# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Evaluation bar figures and LaTeX tables in the style of fig_eval_vllm (make_figs_eval_bars.py).

Five results are figures (vLLM bars, gate heatmap, length-KL panels, position heatmap, Pareto); every
other evaluation result is a numeric table or a bar chart in the vLLM-figure style. Data loading is reused from make_eval_figs.py (same stored results, no new runs). Outputs:
fig_evalb_*.pdf/png, tab_eval_*.tex, tables_preview.tex.
"""

from __future__ import annotations

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script dir
import make_eval_figs as D  # data + helpers; its rcParams are overridden below

OUT = D.OUT
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.2,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.6,
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
    }
)
C_REF, C_BEFORE, C_AFTER = "#EEF2FB", "#F4C27A", "#3A9D8F"  # 16-bit / no repair / SpinTare
C_MID, C_DARK, C_GREY = (
    "#AFC4E8",
    "#27457A",
    "#D9D9D9",
)  # static mean / rotary w/o gate / SmoothAttn.
EDGE = dict(edgecolor="black", linewidth=0.55, zorder=3)
H_BEFORE, H_MID, H_AFTER, H_DARK, H_GREY = "////", "xxxx", "\\\\\\\\", "....", "++"


def style(ax, ylabel: str | None = None) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.45, color="#B5B5B5", zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2)
    ax.tick_params(pad=1.5)


def patch(color: str, hatch: str | None, label: str) -> mpl.patches.Patch:
    return mpl.patches.Patch(
        facecolor=color, edgecolor="black", linewidth=0.55, hatch=hatch, label=label
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


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ figure: engine KL by length |
# reasoning accuracy
def fig_reasoning_bars() -> None:
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(6.3, 1.85), gridspec_kw=dict(width_ratios=[3.4, 2.6], wspace=0.24)
    )
    r1 = D.E2["q8_r1q7-q8_0"]
    lens = ["2048", "4096", "8192", "16384", "32768", "65536"]
    arms = [("q4_0_B", C_BEFORE, H_BEFORE), ("pm", C_MID, H_MID), ("abbias", C_DARK, H_DARK)]
    w = 0.26
    for i, L in enumerate(lens):
        for j, (arm, col, h) in enumerate(arms):
            v = r1[arm][L][0]
            a.bar(i + (j - 1) * w, v, w, color=col, hatch=h, **EDGE)
    a.set_yscale("log")
    a.set_ylim(3e-3, 30)
    a.set_xticks(range(len(lens)))
    a.set_xticklabels(["2K", "4K", "8K", "16K", "32K", "64K"])
    a.set_xlim(-0.5, len(lens) - 0.5)
    a.set_xlabel("Context length", labelpad=1)
    a.set_title("(a) Output KL to the 16-bit cache (llama.cpp)", pad=3)
    style(a, "KL (nats, log scale)")
    # (b) accuracy on the reasoning items
    arms_b = [
        ("16bit", C_REF, None),
        ("plain_q4_0", C_BEFORE, H_BEFORE),
        ("static_postmean", C_MID, H_MID),
        ("smoothattention", C_GREY, H_GREY),
        ("spintare", C_AFTER, H_AFTER),
    ]
    w2 = 0.16
    for i, task in enumerate(["gsm8k", "math500"]):
        for j, (arm, col, h) in enumerate(arms_b):
            rec = D.reasoning_records(arm, task)
            v = 100 * sum(r["ok"] for r in rec) / len(rec)
            x = i + (j - 2) * w2
            b.bar(x, v, w2, color=col, hatch=h, **EDGE)
            b.text(x, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=5.0, rotation=90)
    b.set_ylim(0, 118)
    b.set_yticks([0, 20, 40, 60, 80, 100])
    b.set_xticks([0, 1])
    b.set_xticklabels(["GSM8K (200)", "MATH-500 (500)"])
    b.set_xlim(-0.5, 1.5)
    b.set_title("(b) Accuracy, R1-Distill-Qwen-7B (q4_0 K)", pad=3)
    style(b, "Accuracy (%)")
    legend(
        fig,
        [
            patch(C_REF, None, "16-bit K"),
            patch(C_BEFORE, H_BEFORE, "No repair"),
            patch(C_MID, H_MID, "Static mean (2K calib.)"),
            patch(C_GREY, H_GREY, "SmoothAttention"),
            patch(C_DARK, H_DARK, "SpinTare w/o gate"),
            patch(C_AFTER, H_AFTER, "SpinTare"),
        ],
        6,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
    )
    save(fig, "fig_evalb_reasoning_bars")


# ------------------------------------------------------------------ figure: b_top2 bins (cliff as
# bars)
def fig_cliff_bars() -> None:
    tau = D.T1["thresholds_fit_on_FIT"]["b_top2"]
    edges = [-math.inf, 0.0, 1.0, 1.5, tau, 2.5, 3.0, math.inf]
    labels = ["<0", "0-1", "1-1.5", f"1.5-{tau:.2f}", f"{tau:.2f}-2.5", "2.5-3", ">=3"]
    cells = D.T1["cells"]
    counts = np.zeros((len(labels), 3))
    for c in cells:
        k = next(i for i in range(len(labels)) if edges[i] < c["b_top2"] <= edges[i + 1])
        counts[k, c["y"]] += 1
    n = counts.sum(1)
    share = 100 * counts / n[:, None]
    fig, ax = plt.subplots(figsize=(3.03, 1.6))
    bottom = np.zeros(len(labels))
    for y, col, h, lab in [
        (2, C_BEFORE, H_BEFORE, "Collapse"),
        (1, C_MID, H_MID, "Damaged"),
        (0, C_AFTER, H_AFTER, "Safe"),
    ]:
        ax.bar(
            range(len(labels)),
            share[:, y],
            0.68,
            bottom=bottom,
            color=col,
            hatch=h,
            label=lab,
            **EDGE,
        )
        bottom += share[:, y]
    for i, k in enumerate(n):
        ax.text(i, 101.5, f"n={int(k)}", ha="center", va="bottom", fontsize=5.4)
    ax.axvline(3.5, color="black", lw=0.9, ls=(0, (4, 2)), zorder=4)
    ax.text(
        3.55,
        60,
        r"$\tau_b$",
        fontsize=7,
        ha="left",
        va="center",
        zorder=8,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=5.6)
    ax.set_xlabel(r"$b_{\mathrm{top2}}$ (bits), computed without running the task", labelpad=1)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    style(ax, "Share of cells (%)")
    legend(
        ax,
        [
            patch(C_BEFORE, H_BEFORE, "Collapse"),
            patch(C_MID, H_MID, "Damaged"),
            patch(C_AFTER, H_AFTER, "Safe"),
        ],
        3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    save(fig, "fig_evalb_btop2_bins")
    D.NUMBERS["btop2_bins"] = {
        lab: {
            "n": int(n[i]),
            "collapse": int(counts[i, 2]),
            "damaged": int(counts[i, 1]),
            "safe": int(counts[i, 0]),
        }
        for i, lab in enumerate(labels)
    }


# ------------------------------------------------------------------ figure: component waterfall as
# bars
def fig_waterfall_bars() -> None:
    models = [
        "q25_1p5b",
        "q25_7b",
        "q3_0p6b",
        "r1q_7b",
        "llama31_8b",
        "llama3b",
        "mistral7b",
        "granite8b",
    ]
    arms = [
        ("A_nr", C_BEFORE, H_BEFORE, "No repair"),
        ("B2_nr", C_MID, H_MID, "Static mean (2K calib.)"),
        ("D_nr", C_DARK, H_DARK, "SpinTare w/o gate"),
        ("E_nr", C_AFTER, H_AFTER, "SpinTare"),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    w = 0.2
    for i, m in enumerate(models):
        k = D.LC1[m]["kl"]
        base = k["A_nr"][4]
        for j, (arm, col, h, _) in enumerate(arms):
            v = k[arm][4]
            x = i + (j - 1.5) * w
            ax.bar(x, v, w, color=col, hatch=h, **EDGE)
            if (
                arm in ("B2_nr", "D_nr") and v > base
            ):  # the step leaves the cache worse than no repair
                ax.text(
                    x,
                    v * 1.12,
                    r"$\uparrow$",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#B03A2E",
                )
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 80)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([D.NAME[m] for m in models], fontsize=6.2)
    ax.set_xlim(-0.5, len(models) - 0.5)
    for x0, x1, lab in [
        (-0.5, 3.5, "configuration check selects centring"),
        (3.5, 7.5, "low-offset models, centring forced"),
    ]:
        ax.text((x0 + x1) / 2, 35, lab, ha="center", va="center", fontsize=6.2, style="italic")
    style(ax, "Output KL at 32K (log)")
    handles = [patch(c, h, l) for _, c, h, l in arms]
    handles.append(
        mpl.lines.Line2D(
            [], [], color="#B03A2E", marker=r"$\uparrow$", ls="", ms=6, label="worse than no repair"
        )
    )
    legend(fig, handles, 5, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    save(fig, "fig_evalb_waterfall_bars")


def fig_waterfall_bars_col() -> None:
    """Single-column (3.03 in) waterfall: top panel centring-selected models, bottom panel
    low-offset models."""
    groups = [
        (["q25_1p5b", "q25_7b", "q3_0p6b", "r1q_7b"], "(a) Configuration check selects centring"),
        (
            ["llama31_8b", "llama3b", "mistral7b", "granite8b"],
            "(b) Low-offset models, centring forced",
        ),
    ]
    arms = [
        ("A_nr", C_BEFORE, H_BEFORE, "No repair"),
        ("B2_nr", C_MID, H_MID, "Static mean (2K)"),
        ("D_nr", C_DARK, H_DARK, "SpinTare w/o gate"),
        ("E_nr", C_AFTER, H_AFTER, "SpinTare"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(3.03, 3.1), gridspec_kw=dict(hspace=0.55, top=0.84))
    w = 0.2
    for ax, (models, title) in zip(axes, groups):
        for i, m in enumerate(models):
            k = D.LC1[m]["kl"]
            base = k["A_nr"][4]
            for j, (arm, col, h, _) in enumerate(arms):
                v = k[arm][4]
                x = i + (j - 1.5) * w
                ax.bar(x, v, w, color=col, hatch=h, **EDGE)
                if arm in ("B2_nr", "D_nr") and v > base:
                    ax.text(
                        x,
                        v * 1.12,
                        r"$\uparrow$",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#B03A2E",
                    )
        ax.set_yscale("log")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([D.NAME[m] for m in models], fontsize=6)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_title(title, fontsize=6.8, pad=2)
        ax.tick_params(labelsize=6)
        style(ax, "KL at 32K")
    axes[0].set_ylim(4e-3, 30)
    axes[1].set_ylim(2e-3, 0.3)
    handles = [patch(c, h, l) for _, c, h, l in arms]
    handles.append(
        mpl.lines.Line2D(
            [], [], color="#B03A2E", marker=r"$\uparrow$", ls="", ms=6, label="worse than no repair"
        )
    )
    lg = fig.legend(
        handles=handles,
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=6,
        handlelength=1.4,
        handleheight=0.7,
        columnspacing=0.8,
        borderpad=0.3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    lg.get_frame().set_linewidth(0.5)
    save(fig, "fig_evalb_waterfall_bars_col")


# ------------------------------------------------------------------ LaTeX tables
def tex_num(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def table(
    name: str, caption: str, colspec: str, header: str, rows: list[str], note: str = ""
) -> str:
    body = "\n".join(rows)
    t = (
        f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{tab:{name}}}\n\\footnotesize\n"
        f"\\setlength{{\\tabcolsep}}{{3pt}}\n\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )
    if note:
        t += f"\n\\par\\smallskip\\parbox{{\\linewidth}}{{\\scriptsize {note}}}\n"
    t += "\\end{table}\n"
    (OUT / f"tab_eval_{name}.tex").write_text(t, encoding="utf-8")
    return t


def tab_length() -> str:
    labs = {
        "q8_0": "8-bit K",
        "q4_0": "No repair",
        "pm": "Static mean (2K)",
        "abbias": "SpinTare w/o gate",
        "abmu": "SpinTare w/o gate",
    }
    Ls = ["2048", "4096", "8192", "16384", "32768", "65536", "131072"]
    rows = []
    for m, mn in [
        ("q8_r1q7-q8_0", "R1-Qwen-7B"),
        ("qwen25_7b-q8_0", "Qwen2.5-7B"),
        ("qwen3_0p6b-q8_0", "Qwen3-0.6B"),
        ("llama31_8b-q8_0", "Llama-3.1-8B"),
    ]:
        first = True
        for arm in ["q8_0", "q4_0", "pm", "abbias", "abmu"]:
            if arm not in D.E2[m]:
                continue
            cells = [tex_num(D.E2[m][arm][L][0]) if L in D.E2[m][arm] else "--" for L in Ls]
            rows.append(f"{mn if first else ''} & {labs[arm]} & " + " & ".join(cells) + " \\\\")
            first = False
        rows.append("\\addlinespace[2pt]")
    rows.pop()
    return table(
        "length_engine",
        "Output KL to the 16-bit cache by context length in \\texttt{llama.cpp} "
        "(\\texttt{q4\\_0} K, \\texttt{q8\\_0} V, PG19).",
        "llrrrrrrr",
        "Model & K cache & 2K & 4K & 8K & 16K & 32K & 64K & 128K",
        rows,
        "SpinTare w/o gate = rotary centring with exact add-back but no exemption gate; full SpinTare was "
        "not part of this sweep (its 64K point is queued). -- = not run.",
    )


def tab_waterfall() -> str:
    rows = []
    for m in [
        "q25_1p5b",
        "q25_7b",
        "q3_0p6b",
        "r1q_7b",
        "llama31_8b",
        "llama3b",
        "mistral7b",
        "granite8b",
    ]:
        k = D.LC1[m]["kl"]
        a, s, d, e = (k[x][4] for x in ["A_nr", "B2_nr", "D_nr", "E_nr"])

        def cell(v: float, ref: float) -> str:
            r = v / ref
            mark = "$\\uparrow$" if r > 1 else ""
            return f"{tex_num(v)} ({r:.2g}$\\times${mark})"

        rows.append(f"{D.NAME[m]} & {tex_num(a)} & {cell(s, a)} & {cell(d, a)} & {cell(e, d)} \\\\")
        if m == "r1q_7b":
            rows.append("\\midrule")
    return table(
        "waterfall",
        "Component ablation at 32K: output KL with each step "
        "(simulation, \\texttt{q4\\_0} without rotation, exact values).",
        "lrrrr",
        "Model & No repair & + Static mean (vs.\\ none) & Rotary, no gate (vs.\\ none) & "
        "+ Exemption gate (vs.\\ no gate)",
        rows,
        "$\\uparrow$: the step raises KL. Rows above the line are the models whose configuration check "
        "selects centring.",
    )


def tab_rho() -> str:
    rows = []
    for m in D.ORDER:
        f = D.LC1[m]["fit"]["B2"]
        r = D.LC1[m]["rho"]
        k = D.LC1[m]["kl"]
        rows.append(
            f"{D.NAME[m]} & {f['spearman_pred']:.2f} & {f['median_abs_log10_err_pred']:.3f} & "
            f"{r['meas_B2'][4]:.2f} & {tex_num(k['B2_nr'][4] / k['E_nr'][4])} \\\\"
        )
        if m == "r1q_7b":
            rows.append("\\midrule")
    return table(
        "rho",
        "Residual energy $\\rho(t)$ of the 2K static mean: closed-form prediction against "
        "measurement, and the static-mean penalty at 32K.",
        "lrrrr",
        "Model & Spearman (pred., meas.) & Median $|\\log_{10}$ err.$|$ & Measured $\\rho$ at 32K & "
        "KL static / KL SpinTare",
        rows,
        "Pooled over models and positions, $\\rho$ and the penalty have Spearman "
        f"{D.NUMBERS.get('rho_vs_static_penalty_spearman', float('nan')):.2f}: $\\rho$ explains where the "
        "static mean drifts but not how much it costs.",
    )


def tab_row() -> str:
    rg = D.NUMBERS["row_gate"]
    rows = []
    for m in rg["D/E"]:
        de = rg["D/E"][m]
        xb = rg["X_bos/X_z"][m]
        rows.append(
            f"{D.NAME[m]} & {de[1]:.2f} [{de[0]:.2f}, {de[2]:.2f}] & "
            f"{xb[1]:.2f} [{xb[0]:.2f}, {xb[2]:.2f}] \\\\"
        )
        if m == "q3_8b":
            rows.append("\\midrule")
    return table(
        "row_gate",
        "SpinTare exemption gate at 32K (simulation, 8 texts): median [min, max] KL ratios.",
        "lcc",
        "Model & SpinTare w/o gate / SpinTare (same bits) & Row 0 exact / SpinTare's gated rows "
        "exact",
        rows,
        "Above 1: the alternative is worse than SpinTare's exemption gate. Rows above the line are "
        "offset-dominated.",
    )


def tab_detection() -> str:
    # raw set names of the T1 rescore; their mapping to the fitting / prospective / blind /
    # newly-released sets of z_analyses.py is not yet reconciled (z_analyses: LOFO BA 0.967 with 2
    # misses; T1: 0.983 with 1)
    names = {"FIT": "FIT", "PRED-P": "PRED-P", "N13": "N13", "N1": "N1", "N4": "N4 (injected)"}
    rows = []
    for s in ["FIT", "PRED-P", "N13", "N1", "N4"]:
        v = D.T1["sets"][s]
        r = v["rules"]["b_top2"]
        ba = "--" if r["ba"] != r["ba"] else f"{r['ba']:.3f}"
        rows.append(
            f"{names[s]} & {v['n_cells']} & {v['n_damaged']} & {r['misclear']} ({r['misclear_collapse']}) "
            f"& {r['false_alarm']} & {ba} \\\\"
        )
    return table(
        "detection_sets",
        "SpinTare configuration check ($b_{\\mathrm{top2}} \\le \\tau_b$, frozen) "
        "on every model set.",
        "lrrrrr",
        "Set & Cells & Damaged & Missed (collapse) & False alarms & Bal.\\ acc.",
        rows,
        "T1 rescore (res\\_t1\\_score.json). N4 holds only damaged cells, so balanced accuracy is "
        "undefined. "
        "The mapping to the z\\_analyses sets and the FIT difference (LOFO 0.967, 2 misses) are not "
        "yet reconciled.",
    )


def tab_failure() -> str:
    names = {a: l for a, l, _ in D.ARMS_R}
    rows = []
    for task, tn in [("gsm8k", "GSM8K"), ("math500", "MATH-500")]:
        first = True
        for arm, _, _ in D.ARMS_R:
            v = D.NUMBERS["failure_causes"][f"{arm}/{task}"]
            n = v["n"]
            rows.append(
                f"{tn if first else ''} & {names[arm]} & {100 * v['correct'] / n:.1f} & {100 * v['wrong'] / n:.1f} "
                f"& {100 * v['loop'] / n:.1f} & {100 * v['cap'] / n:.1f} & {v['mean_gen']} \\\\"
            )
            first = False
        rows.append("\\addlinespace[2pt]")
    rows.pop()
    return table(
        "failure_causes",
        "Where long reasoning fails (R1-Distill-Qwen-7B, \\texttt{llama.cpp}, \\% of items).",
        "llrrrrr",
        "Task & K cache & Correct & Wrong, ended & Loop & Length cap & Mean gen.\\ tokens",
        rows,
        "Loop and length cap count wrong items only.",
    )


def tab_pareto() -> str:
    pr = D.NUMBERS["pareto_reasoning"]
    names = {
        "16bit": "16-bit K",
        "spintare": "SpinTare",
        "static_postmean": "Static mean",
        "plain_q4_0": "No repair / SmoothAttn.",
    }
    rows = []
    for task, tn in [("gsm8k", "GSM8K"), ("math500", "MATH-500")]:
        first = True
        for arm in ["16bit", "spintare", "static_postmean", "plain_q4_0"]:
            v = pr[f"{arm}/{task}"]
            sl = v["slope_per_doubling"]
            seg = " & ".join(
                "--" if sl[k] is None else f"{sl[k]:.1f}"
                for k in ["1024-2048", "2048-4096", "4096-8192"]
            )
            rows.append(
                f"{tn if first else ''} & {names[arm]} & {seg} & {v['8192'][0]:.1f} & {v['8192'][1]:.1f} \\\\"
            )
            first = False
        rows.append("\\addlinespace[2pt]")
    rows.pop()
    return table(
        "pareto_slopes",
        "Accuracy gained per doubling of KV memory along the budget sweep of the Pareto figure.",
        "llrrrrr",
        "Task & K cache & 1K$\\to$2K & 2K$\\to$4K & 4K$\\to$8K & MiB at 8K & Acc.\\ at 8K",
        rows,
        "Memory counts generated tokens only; values truncate the same runs at each budget.",
    )


def tables_preview(parts: list[str]) -> None:
    doc = (
        "\\documentclass[10pt]{article}\n\\usepackage[margin=1.6cm]{geometry}\n\\usepackage{times}\n"
        "\\usepackage{booktabs}\n\\usepackage{amsmath,amssymb}\n\\usepackage{hyperref}\n\\usepackage{cleveref}\n"
        "\\newcommand{\\fitcol}[1]{#1}\n\\begin{document}\n\\setcounter{table}{4}\n"
        + "\n\\clearpage\n".join(parts)
        + "\n\\end{document}\n"
    )
    (OUT / "tables_preview.tex").write_text(
        doc,
        encoding="utf-8",
    )


if __name__ == "__main__":
    # rebuild the numbers the tables use (same data path as make_eval_figs.py, without redrawing its
    # figures)
    plt.switch_backend("agg")
    D.save = lambda fig, name: plt.close(fig)  # numbers only; superseded figures are not redrawn
    D.fig_rho()
    D.fig_row_gate()
    D.tab_fail_causes()
    D.fig_pareto_reasoning()
    plt.close("all")
    mpl.rcParams.update({"font.family": "serif"})
    fig_reasoning_bars()
    fig_cliff_bars()
    fig_waterfall_bars()
    fig_waterfall_bars_col()  # replaces the earlier waterfall table
    tables_preview(
        [tab_length(), tab_rho(), tab_row(), tab_detection(), tab_failure(), tab_pareto()]
    )
    print("tables written")
