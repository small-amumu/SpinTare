"""Evaluation data figures and tables for SpinTare, recomputed from stored results only (no new
runs).

Outputs (next to this script): fig5_*.pdf/png, tab5_*.md, ch5_numbers.json.
Style follows make_figs.py (Okabe-Ito, 7.5 pt, print size).
"""

from __future__ import annotations

import json
import os
import math
from pathlib import Path
from statistics import median

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

# figures are written next to this file unless FIG_OUT names another directory
OUT = Path(os.environ.get("FIG_OUT", Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)
RES = Path(__file__).resolve().parents[3] / "results" / "spintare"  # frozen inputs of the figures

LC1 = json.loads((RES / "long_context_lc1" / "res_lc1_agg.json").read_text(encoding="utf-8"))[
    "models"
]
C23 = json.loads((RES / "exemption_gate_c23" / "res_c23_agg.json").read_text(encoding="utf-8"))
E2 = json.loads((RES / "long_context_llamacpp" / "e2_final.json").read_text(encoding="utf-8"))["D"]
T1 = json.loads((RES / "detection_gate" / "res_t1_score.json").read_text(encoding="utf-8"))
GP = json.loads((RES / "detection_gate" / "gp_score.json").read_text(encoding="utf-8"))
PROS = RES / "prospective"
R1 = RES / "reasoning_r1"  # per-item records of the R1-Distill-Qwen-7B reasoning cells

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
C = {
    "none": "#7F7F7F",
    "static": "#E69F00",
    "static_full": "#CC79A7",
    "rotary": "#0072B2",
    "gate": "#009E73",
    "red": "#D55E00",
    "sa": "#56B4E9",
    "f16": "#000000",
}
NAME = {
    "q3_0p6b": "Qwen3-0.6B",
    "q25_1p5b": "Qwen2.5-1.5B",
    "llama3b": "Llama-3.2-3B",
    "q25_7b": "Qwen2.5-7B",
    "r1q_7b": "R1-Qwen-7B",
    "llama31_8b": "Llama-3.1-8B",
    "mistral7b": "Mistral-7B",
    "q3_8b": "Qwen3-8B",
    "granite8b": "Granite-3.1-8B",
    "olmo7b": "OLMo-2-7B",
    "yi6b": "Yi-1.5-6B",
    "ds7b": "DeepSeek-7B",
}
# Offset-dominated models first (configuration check selects centring), then the rest.
ORDER = [
    "q25_1p5b",
    "q25_7b",
    "q3_0p6b",
    "q3_8b",
    "r1q_7b",
    "llama31_8b",
    "llama3b",
    "mistral7b",
    "granite8b",
    "olmo7b",
    "yi6b",
    "ds7b",
]
BUCKETS_K = ["1K", "3.5K", "7.7K", "15.9K", "32K"]
NUMBERS: dict = {}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ teaser: mechanism | outcome
def reasoning_records(arm: str, task: str) -> list[dict]:
    d = R1 / arm / task
    files = sorted(
        (p for p in d.glob("*.json") if not p.name.startswith("meta") and "journal" not in p.name),
        key=lambda p: int(p.stem.rsplit("_s", 1)[1].split("n")[0]),
    )  # shard start, e.g. _s250n250
    recs: list[dict] = []
    for p in files:
        recs += json.loads(p.read_text(encoding="utf-8"))["records"]
    expect = {"gsm8k": 200, "math500": 500}[task]
    assert len(recs) == expect, (arm, task, len(recs))
    return recs


ARMS_R = [
    ("16bit", "16-bit", C["f16"]),
    ("plain_q4_0", "No repair", C["none"]),
    ("static_postmean", "Static mean", C["static"]),
    ("smoothattention", "SmoothAttn.", C["sa"]),
    ("spintare", "SpinTare", C["gate"]),
]


def fig_teaser() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 1.75), gridspec_kw={"width_ratios": [1.0, 1.25]})
    r1 = E2["q8_r1q7-q8_0"]
    for arm, lab, col, ls in [
        ("q4_0", "No repair", C["none"], "-"),
        ("pm", "Static mean (2K)", C["static"], "-"),
        ("abbias", "Rotary add-back", C["rotary"], "-"),
    ]:
        pts = sorted((int(k), v[0]) for k, v in r1[arm].items())
        a.plot(
            [p[0] / 1024 for p in pts], [p[1] for p in pts], ls, color=col, marker="o", label=lab
        )
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xticks([2, 4, 8, 16, 32, 64])
    a.set_xticklabels(["2K", "4K", "8K", "16K", "32K", "64K"])
    a.axvspan(1.6, 2.0, color="#DDEBE6", lw=0, zorder=0)
    a.set_xlabel("Context length")
    a.set_ylabel("Output KL to 16-bit")
    a.set_title("(a) Static mean crosses no repair", loc="left")
    a.legend(frameon=False, loc="center right", fontsize=6)
    a.grid(True, which="major", axis="y")
    # (b) accuracy bars with loop rate on top
    x = np.arange(2)
    w = 0.16
    for i, (arm, lab, col) in enumerate(ARMS_R):
        accs, loops = [], []
        for task in ["gsm8k", "math500"]:
            rec = reasoning_records(arm, task)
            accs.append(100 * sum(r["ok"] for r in rec) / len(rec))
            loops.append(100 * sum(r["loop"] for r in rec) / len(rec))
        bars = b.bar(x + (i - 2) * w, accs, w, color=col, label=lab)
        for bar, lp in zip(bars, loops):
            if lp >= 5:
                b.text(
                    bar.get_x() + w / 2,
                    bar.get_height() + 1.5,
                    f"{lp:.0f}",
                    ha="center",
                    fontsize=5.2,
                    color=C["red"],
                )
    b.set_xticks(x)
    b.set_xticklabels(["GSM8K (200)", "MATH-500"])
    b.set_ylabel("Accuracy (%)")
    b.set_ylim(0, 105)
    b.set_title("(b) R1-Distill-Qwen-7B, q4_0 K (red: loop %)", loc="left")
    b.legend(
        frameon=False,
        ncol=5,
        fontsize=5.6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        handlelength=1.0,
        columnspacing=0.8,
    )
    save(fig, "fig5_1_teaser")


# ------------------------------------------------------------------ RQ1: position x arm heatmap
def fig_pos_heatmap() -> None:
    arms = [
        ("B2_nr", "Static, 2K calib."),
        ("BL_nr", "Static, full 32K calib."),
        ("D_nr", "Rotary, no gate"),
        ("E_nr", "SpinTare"),
    ]
    mats = []
    for arm, _ in arms:
        mats.append(
            np.array(
                [
                    [math.log10(LC1[m]["kl"][arm][j] / LC1[m]["kl"]["A_nr"][j]) for j in range(5)]
                    for m in ORDER
                ]
            )
        )
    vmax = max(abs(m).max() for m in mats)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.3), sharey=True)
    for ax, (arm, lab), M in zip(axes, arms, mats):
        im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels(BUCKETS_K, rotation=45)
        ax.set_title(lab, fontsize=7)
        ax.axhline(4.5, color="k", lw=0.6)
        for i in range(M.shape[0]):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=4.6,
                    color="white" if abs(M[i, j]) > 0.55 * vmax else "black",
                )
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(range(len(ORDER)))
    axes[0].set_yticklabels([NAME[m] for m in ORDER])
    fig.supxlabel(
        "Scored position bucket (2K calibration window = first bucket)", fontsize=7, y=-0.06
    )
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label(r"log$_{10}$(KL$_{arm}$ / KL$_{no\ repair}$)", fontsize=6.5)
    save(fig, "fig5_2_position_heatmap")
    NUMBERS["heatmap_32k"] = {
        arm: {m: round(float(M[i, 4]), 3) for i, m in enumerate(ORDER)}
        for (arm, _), M in zip(arms, mats)
    }


def fig_pos_heatmap_col() -> None:
    """Single-column (3.03 in) version of the position heatmap: 2 x 2 panels, fonts >= 6 pt."""
    arms = [
        ("B2_nr", "Static, 2K calib."),
        ("BL_nr", "Static, full 32K calib."),
        ("D_nr", "Rotary, no gate"),
        ("E_nr", "SpinTare"),
    ]
    mats = [
        np.array(
            [
                [math.log10(LC1[m]["kl"][arm][j] / LC1[m]["kl"]["A_nr"][j]) for j in range(5)]
                for m in ORDER
            ]
        )
        for arm, _ in arms
    ]
    vmax = max(abs(M).max() for M in mats)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(
        2, 2, figsize=(3.03, 3.75), sharey=True, gridspec_kw=dict(hspace=0.42, wspace=0.06)
    )
    short = ["1K", "3.5K", "7.7K", "16K", "32K"]
    for ax, (arm, lab), M in zip(axes.flat, arms, mats):
        im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels(short, fontsize=6, rotation=45)
        ax.set_title(lab, fontsize=6.5, pad=2)
        ax.axhline(4.5, color="k", lw=0.6)
        ax.tick_params(length=1.5, pad=1)
        for i in range(M.shape[0]):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:+.1f}".replace("+0.0", "0.0").replace("-0.0", "0.0"),
                    ha="center",
                    va="center",
                    fontsize=4.8,
                    color="white" if abs(M[i, j]) > 0.55 * vmax else "black",
                )
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([NAME[m] for m in ORDER], fontsize=6)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.035, pad=0.09, aspect=40)
    cb.set_label(
        r"$\log_{10}(\mathrm{KL}_{\mathrm{arm}}/\mathrm{KL}_{\mathrm{no\ repair}})$", fontsize=6
    )
    cb.ax.tick_params(labelsize=6, length=1.5)
    save(fig, "fig5_2_position_heatmap_col")


# ------------------------------------------------------------------ RQ1: rho structure -> behaviour
def fig_rho() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 1.9))
    xs, ys, ks, kl = [], [], [], []
    for m in ORDER:
        r = LC1[m]["rho"]
        a.scatter(
            r["pred_B2"],
            r["meas_B2"],
            s=9,
            color=C["gate"] if m in ORDER[:5] else C["none"],
            alpha=0.85,
            lw=0,
        )
        for j in range(5):
            ks.append(r["meas_B2"][j])
            kl.append(LC1[m]["kl"]["B2_nr"][j] / LC1[m]["kl"]["E_nr"][j])
            xs.append(r["pred_B2"][j])
            ys.append(r["meas_B2"][j])
        b.plot(
            r["meas_B2"],
            [LC1[m]["kl"]["B2_nr"][j] / LC1[m]["kl"]["E_nr"][j] for j in range(5)],
            "-o",
            ms=2.5,
            lw=0.7,
            color=C["gate"] if m in ORDER[:5] else C["none"],
            alpha=0.85,
        )
    lim = [0, max(max(xs), max(ys)) * 1.05]
    a.plot(lim, lim, "k--", lw=0.6)
    a.set_xlabel(r"Predicted $\rho(t)$, Eq. (rho)")
    a.set_ylabel(r"Measured $\rho(t)$")
    sp = [LC1[m]["fit"]["B2"]["spearman_pred"] for m in ORDER]
    a.set_title(
        f"(a) Residual energy, 12 models (Spearman {min(sp):.2f}-{max(sp):.2f})",
        loc="left",
        fontsize=7,
    )
    b.set_yscale("log")
    b.axhline(1, color="k", lw=0.5)
    b.axvline(1, color=C["red"], lw=0.6, ls=":")
    b.set_xlabel(r"Measured $\rho(t)$ (5 position buckets per model)")
    b.set_ylabel("KL static / KL SpinTare")
    b.set_title(
        f"(b) Static penalty vs residual energy (pooled Spearman {spearman(ks, kl):.2f})",
        loc="left",
        fontsize=7,
    )
    b.plot([], [], "-o", color=C["gate"], ms=2.5, label="centring selected")
    b.plot([], [], "-o", color=C["none"], ms=2.5, label="not selected")
    b.legend(frameon=False, fontsize=6)
    save(fig, "fig5_3_rho_structure")
    NUMBERS["rho_vs_static_penalty_spearman"] = round(spearman(ks, kl), 3)
    NUMBERS["rho_pred_spearman_per_model"] = {
        m: round(LC1[m]["fit"]["B2"]["spearman_pred"], 3) for m in ORDER
    }


def spearman(a: list[float], b: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(s):
            r[i] = float(k)
        return r

    ra, rb = rank(a), rank(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


# ------------------------------------------------------------------ RQ2: row gate vs no gate vs
# BOS-only
def fig_row_gate() -> None:
    """(a) same-bits ungated rotary vs gated (D/E); (b) exact rows: row 0 only vs all gated rows
    (X_bos/X_z).

    E_bos is not used: that arm codes row 0 as raw q4_0 whatever the gate says, which collapses in
    the Qwen2.5 models (REGISTRATION_C23.md, outcome note on H5); X_bos/X_z is the clean first-row
    test.
    """
    texts = list(C23["texts"])
    models = [m for m in ORDER if m in C23["texts"][texts[0]]["per_model"]]
    nq = sum(m in ORDER[:5] for m in models)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.0), sharey=False)
    out = {}
    for ax, (num, den, lab, col, title) in zip(
        axes,
        [
            ("D", "E", "Rotary, no gate", C["rotary"], "(a) Same bits: no gate / exemption gate"),
            (
                "X_bos",
                "X_z",
                "Row 0 only exact",
                C["static"],
                "(b) Exact rows: row 0 only / all gated rows",
            ),
        ],
    ):
        x = np.arange(len(models))
        meds, lo, hi = [], [], []
        for m in models:
            v = [
                C23["texts"][t]["per_model"][m]["kl"][num]
                / C23["texts"][t]["per_model"][m]["kl"][den]
                for t in texts
            ]
            meds.append(median(v))
            lo.append(min(v))
            hi.append(max(v))
        out[f"{num}/{den}"] = {
            m: [round(lo[i], 3), round(meds[i], 3), round(hi[i], 3)] for i, m in enumerate(models)
        }
        ax.errorbar(
            x,
            meds,
            yerr=[np.array(meds) - lo, np.array(hi) - meds],
            fmt="o",
            color=col,
            ms=3.5,
            capsize=1.5,
            lw=0.8,
        )
        ax.set_yscale("log")
        ax.axhline(1, color=C["gate"], lw=1.0)
        ax.axvline(nq - 0.5, color="k", lw=0.5, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([NAME[m] for m in models], rotation=35, ha="right", fontsize=6)
        ax.set_title(title, loc="left", fontsize=7)
    axes[0].set_ylabel("KL ratio (median,\nrange over 8 texts)")
    axes[0].text(
        0.02,
        0.95,
        "offset-dominated | low-offset",
        transform=axes[0].transAxes,
        fontsize=5.8,
        va="top",
    )
    save(fig, "fig5_4_row_gate")
    NUMBERS["row_gate"] = out


# ------------------------------------------------------------------ RQ3: b_top2 cliff
def task_loss_cells() -> list[dict]:
    gk = {(c["tag"], c["cfg"]): c for c in GP["cells"]}
    labs: dict = {}
    for d in ["predp", "n1", "n4", "n13"]:
        labs.update(json.loads((PROS / d / "labels.json").read_text(encoding="utf-8")))
    rows = []
    for c in T1["cells"]:
        key = (c["tag"], c["cfg"])
        if c["set"] == "FIT":
            loss = gk[key].get("r")
        else:
            lab = labs.get(f"{c['tag']}|{c['cfg']}")
            loss = None if lab is None or lab.get("rel") is None else -100 * lab["rel"]
        rows.append(
            {
                "set": c["set"],
                "tag": c["tag"],
                "cfg": c["cfg"],
                "y": c["y"],
                "b_top2": c["b_top2"],
                "loss": loss,
            }
        )
    return rows


def fig_cliff() -> None:
    tau = T1["thresholds_fit_on_FIT"]["b_top2"]
    rows = [r for r in task_loss_cells() if r["loss"] is not None]
    rows.sort(key=lambda r: (r["b_top2"] <= tau, -r["loss"]))
    miss_ids: list[str] = []
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    mk = {"FIT": "o", "PRED-P": "s", "N1": "^", "N4": "v", "N13": "D"}
    col = {0: C["gate"], 1: C["static"], 2: C["red"]}
    for r in rows:
        yv = min(max(r["loss"], -5), 100)
        ax.scatter(r["b_top2"], yv, marker=mk[r["set"]], s=12, color=col[r["y"]], alpha=0.85, lw=0)
        if r["b_top2"] > tau and r["y"] >= 1:
            miss_ids.append(f"{len(miss_ids) + 1}: {r['set']} {r['tag']}/{r['cfg']}")
            ax.annotate(
                str(len(miss_ids)),
                (r["b_top2"], yv),
                fontsize=5.5,
                xytext=(3, 1),
                textcoords="offset points",
            )
    ax.axvline(tau, color="k", lw=0.8, ls="--")
    ax.text(tau + 0.05, 70, rf"$\tau_b$={tau:.3f}", fontsize=6)
    ax.set_xlabel(r"$b_{\mathrm{top2}}$ (bits, no task run)")
    ax.set_ylabel("Task loss vs 16-bit (%, clipped at 100)")
    for s, m in mk.items():
        ax.scatter([], [], marker=m, color="#555555", s=12, label=s)
    for y, lab in [(0, "safe"), (1, "damaged"), (2, "collapse")]:
        ax.scatter([], [], marker="s", color=col[y], s=12, label=lab)
    ax.legend(frameon=False, fontsize=5.3, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("Collapse cliff below the frozen threshold", loc="left", fontsize=7)
    save(fig, "fig5_5_btop2_cliff")
    below = [r for r in rows if r["b_top2"] <= tau]
    above = [r for r in rows if r["b_top2"] > tau]
    NUMBERS["cliff"] = {
        "n_cells_with_task_loss": len(rows),
        "tau_b": tau,
        "below_tau_median_loss": round(median(r["loss"] for r in below), 2),
        "above_tau_median_loss": round(median(r["loss"] for r in above), 2),
        "collapse_above_tau": sum(r["y"] == 2 for r in above),
        "safe_below_tau": sum(r["y"] == 0 for r in below),
        "n_below": len(below),
        "n_above": len(above),
        "numbered_misses": miss_ids,
        "misses_above_tau": [
            f"{r['set']}:{r['tag']}/{r['cfg']} y={r['y']} loss={r['loss']:.1f}"
            for r in above
            if r["y"] >= 1
        ],
        "per_set_b_top2": {k: v["rules"]["b_top2"] for k, v in T1["sets"].items()},
    }


# ------------------------------------------------------------------ RQ4: failure causes (reasoning)
def tab_fail_causes() -> None:
    lines = [
        "| Arm | Task | n | Correct | Wrong, ended | Loop | Length cap, no loop | Mean gen tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.6), sharey=True)
    cats = [
        ("Correct", C["gate"]),
        ("Wrong, ended", C["static"]),
        ("Loop", C["red"]),
        ("Length cap", C["none"]),
    ]
    for ax, task, tname in zip(axes, ["gsm8k", "math500"], ["GSM8K (200)", "MATH-500 (500)"]):
        left = np.zeros(len(ARMS_R))
        vals = []
        for arm, lab, _ in ARMS_R:
            rec = reasoning_records(arm, task)
            n = len(rec)
            ok = sum(r["ok"] for r in rec)
            lp = sum((not r["ok"]) and r["loop"] for r in rec)
            cap = sum((not r["ok"]) and (not r["loop"]) and r["truncated"] for r in rec)
            wr = n - ok - lp - cap
            g = sum(r["n_gen"] for r in rec) / n
            vals.append([100 * ok / n, 100 * wr / n, 100 * lp / n, 100 * cap / n])
            lines.append(
                f"| {lab} | {task} | {n} | {100 * ok / n:.1f} | {100 * wr / n:.1f} | {100 * lp / n:.1f} | "
                f"{100 * cap / n:.1f} | {g:.0f} |"
            )
            out[f"{arm}/{task}"] = {
                "correct": ok,
                "wrong": wr,
                "loop": lp,
                "cap": cap,
                "n": n,
                "mean_gen": round(g),
            }
        V = np.array(vals)
        for k, (cname, ccol) in enumerate(cats):
            ax.barh(range(len(ARMS_R)), V[:, k], left=left, color=ccol, label=cname, height=0.65)
            left += V[:, k]
        ax.set_title(tname, loc="left", fontsize=7)
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.set_xlabel("Share of items (%)")
    axes[0].set_yticks(range(len(ARMS_R)))
    axes[0].set_yticklabels([a[1] for a in ARMS_R])
    axes[1].legend(
        frameon=False, fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(-0.1, -0.35)
    )
    save(fig, "fig5_6_failure_causes")
    (OUT / "tab5_failure_causes.md").write_text(
        "# Reasoning failure causes (R1-Distill-Qwen-7B, llama.cpp, CH4X items)\n\n"
        "Loop and length cap are counted only among wrong items; a looping item that also hit the cap "
        "counts as Loop.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    NUMBERS["failure_causes"] = out


# ------------------------------------------------------------------ Pareto: KV memory per
# generation vs accuracy
# R1-Distill-Qwen-7B: 28 layers, 4 KV heads, head_dim 128. V is q8_0 in every arm
# (REGISTRATION_CH4X).
R1_ELEMS = 28 * 4 * 128  # K (or V) elements per token
K_BITS = {
    "16bit": 16.0,
    "plain_q4_0": 4.5,
    "static_postmean": 4.5,
    "smoothattention": 4.5,
    "spintare": 4.5 + 32 / 128,
}  # SpinTare stores one F32 gate flag per row (layer, KV head)
V_BITS = 8.5
BUDGETS_FINE = sorted({int(round(b)) for b in np.geomspace(256, 8192, 15)})  # 15 log-spaced budgets
MILESTONES = [1024, 2048, 4096, 8192]


def pareto_curve(
    rec: list[dict], bpt: float, budgets: list[int]
) -> tuple[list[float], list[float]]:
    """Accuracy and mean KV memory (MiB) when the recorded runs are cut at each budget."""
    n = len(rec)
    xs, ys = [], []
    for B in budgets:
        ys.append(100 * sum(r["ok"] and r["n_gen"] <= B for r in rec) / n)
        xs.append(sum(min(r["n_gen"], B) for r in rec) / n * bpt / 2**20)
    return xs, ys


def fig_pareto_reasoning() -> None:
    """Accuracy vs KV-cache memory per generation, sweeping the generation budget (Not All Bits,
    Fig. 1 style).

    Each budget B truncates the same runs: an item counts as correct only if it was correct and
    finished within B tokens; its memory is min(n_gen, B) tokens. Prompt tokens are not counted.
    Per-item seeds make the first B tokens those of the recorded run; -np 4 batching noise is not
    modelled. No repair and SmoothAttention are 0% at every budget and are drawn as one line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=True)
    arms = [
        ("16bit", "16-bit K", C["f16"], "o"),
        ("spintare", "SpinTare (4-bit K)", C["gate"], "s"),
        ("static_postmean", "Static mean (4-bit K)", C["static"], "s"),
        ("plain_q4_0", "No repair / SmoothAttn. (4-bit K, 0%)", C["none"], "s"),
    ]
    out: dict = {}
    for ax, task, tname in zip(
        axes, ["gsm8k", "math500"], ["(a) GSM8K (200 items)", "(b) MATH-500 (500 items)"]
    ):
        for arm, lab, col, m in arms:
            rec = reasoning_records(arm, task)
            bpt = R1_ELEMS * (K_BITS[arm] + V_BITS) / 8  # bytes per cached token
            xs, ys = pareto_curve(rec, bpt, BUDGETS_FINE)
            ls = "--" if arm == "plain_q4_0" else "-"
            ax.plot(
                xs,
                ys,
                ls,
                color=col,
                lw=1.2,
                marker=m,
                ms=3.0,
                mec="white",
                mew=0.4,
                label=lab,
                zorder=3,
            )
            mx, my = pareto_curve(rec, bpt, MILESTONES)
            out[f"{arm}/{task}"] = {
                str(B): [round(x, 2), round(y, 1)] for B, x, y in zip(MILESTONES, mx, my)
            }
            # accuracy gained per doubling of memory, between milestone budgets
            out[f"{arm}/{task}"]["slope_per_doubling"] = {
                f"{a}-{b}": round((my[j + 1] - my[j]) / math.log2(mx[j + 1] / mx[j]), 1)
                if mx[j + 1] > mx[j]
                else None
                for j, (a, b) in enumerate(zip(MILESTONES[:-1], MILESTONES[1:]))
            }
        ax.set_xscale("log")
        ax.set_xlim(8, 200)
        ax.set_ylim(-3, 100)
        ax.set_xlabel("KV-cache memory per generation (MiB)")
        ax.set_title(tname, loc="left", fontsize=7)
        ax.grid(True, which="both", axis="x", alpha=0.25)
        ax.grid(True, which="major", axis="y", alpha=0.35)
    axes[0].set_ylabel("Accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h,
        l,
        frameon=False,
        ncol=4,
        fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        handlelength=2.0,
        columnspacing=1.2,
    )
    save(fig, "fig5_7_pareto_reasoning")
    NUMBERS["pareto_reasoning"] = out


def x_at_accuracy(xs: list[float], ys: list[float], target: float) -> float | None:
    """Memory (log-interpolated) at which a monotone budget curve first reaches `target`
    accuracy."""
    for j in range(1, len(xs)):
        if ys[j - 1] < target <= ys[j]:
            f = (target - ys[j - 1]) / (ys[j] - ys[j - 1])
            return math.exp(math.log(xs[j - 1]) + f * (math.log(xs[j]) - math.log(xs[j - 1])))
    return None


def fig_pareto_combined() -> None:
    """One panel: GSM8K (solid, circles) and MATH-500 (dashed, triangles); colour = K cache arm.

    Same data and truncation rule as fig_pareto_reasoning. Only R1-Distill-Qwen-7B has every arm on
    the current build, so no other model is drawn.
    """
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    arms = [
        ("16bit", "16-bit K", C["f16"]),
        ("spintare", "SpinTare (4-bit K)", C["gate"]),
        ("static_postmean", "Static mean (4-bit K)", C["static"]),
        ("plain_q4_0", "No repair / SmoothAttn. (4-bit K)", C["none"]),
    ]
    tasks = [("gsm8k", "GSM8K", "-", "o"), ("math500", "MATH-500", "--", "^")]
    curves: dict = {}
    for task, tn, ls, mk in tasks:
        for arm, lab, col in arms:
            rec = reasoning_records(arm, task)
            bpt = R1_ELEMS * (K_BITS[arm] + V_BITS) / 8
            xs, ys = pareto_curve(rec, bpt, BUDGETS_FINE)
            curves[(arm, task)] = (xs, ys)
            ax.plot(
                xs, ys, ls, color=col, lw=1.3, marker=mk, ms=3.2, mec="white", mew=0.4, zorder=3
            )
            if arm != "plain_q4_0":  # end label: accuracy at the 8K budget
                ax.annotate(
                    f"{ys[-1]:.1f}",
                    (xs[-1], ys[-1]),
                    xytext=(3, 0),
                    textcoords="offset points",
                    fontsize=6,
                    color=col,
                    va="center",
                )
    # memory saving at equal accuracy (16-bit vs SpinTare)
    for task, target, dy in [("gsm8k", 80, 4), ("math500", 60, 4)]:
        x16 = x_at_accuracy(*curves[("16bit", task)], target)
        xsp = x_at_accuracy(*curves[("spintare", task)], target)
        ax.annotate(
            "",
            xy=(xsp, target),
            xytext=(x16, target),
            arrowprops=dict(arrowstyle="<->", color=C["gate"], lw=0.9, shrinkA=0, shrinkB=0),
        )
        lab = f"{x16 / xsp:.1f}$\\times$ less memory"
        if task == "gsm8k":  # free space above the middle of the arrow
            ax.text(
                math.sqrt(xsp * x16),
                target + 2,
                lab,
                ha="center",
                va="bottom",
                fontsize=6,
                color=C["gate"],
                fontweight="bold",
            )
        else:  # right of the 16-bit end, below its curve
            ax.text(
                x16 * 1.05,
                target - 1.5,
                lab,
                ha="left",
                va="top",
                fontsize=6,
                color=C["gate"],
                fontweight="bold",
            )
        NUMBERS.setdefault("pareto_memory_ratio", {})[f"{task}@{target}%"] = round(x16 / xsp, 2)
    # static mean plateau and the collapsed arms
    xs_, ys_ = curves[("static_postmean", "math500")]
    ax.annotate(
        "static mean plateaus:\nloops beyond the 2K window",
        (xs_[-3], ys_[-3]),
        xytext=(18, -30),
        textcoords="offset points",
        fontsize=6,
        color=C["static"],
        arrowprops=dict(arrowstyle="->", color=C["static"], lw=0.6),
    )
    xn = curves[("plain_q4_0", "math500")][0][-1]
    ax.annotate(
        f"no repair / SmoothAttn.: 0% at every budget;\nloops fill it, up to {xn:.0f} MiB",
        (xn, 0),
        xytext=(-4, 12),
        textcoords="offset points",
        ha="right",
        fontsize=6,
        color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.6),
    )
    ax.set_xscale("log")
    ax.set_xlim(12, 190)
    ax.set_ylim(-3, 102)
    ax.set_xticks([15, 20, 30, 50, 70, 100, 150])
    ax.set_xticklabels(["15", "20", "30", "50", "70", "100", "150"])
    ax.minorticks_off()
    ax.set_xlabel("KV-cache memory per generation (MiB, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, which="major", alpha=0.35)
    h_arm = [mpl.lines.Line2D([], [], color=col, lw=1.6, label=lab) for _, lab, col in arms]
    h_task = [
        mpl.lines.Line2D([], [], color="#333333", ls=ls, marker=mk, ms=3.5, lw=1.2, label=tn)
        for _, tn, ls, mk in tasks
    ]
    ax.legend(handles=h_arm + h_task, frameon=False, fontsize=6.2, loc="upper left", ncol=1)
    ax.set_title(
        "R1-Distill-Qwen-7B, llama.cpp, V q8_0; generation budget 256-8192 tokens (15 points)",
        loc="left",
        fontsize=7,
    )
    save(fig, "fig5_7_pareto_reasoning")


def fig_pareto_combined_col() -> None:
    """Single-column (3.03 in), flat version of the combined Pareto plot; benchmarks labelled in the
    plot."""
    fig, ax = plt.subplots(figsize=(3.03, 1.72))
    arms = [
        ("16bit", "16-bit K", C["f16"]),
        ("spintare", "SpinTare", C["gate"]),
        ("static_postmean", "Static mean", C["static"]),
        ("plain_q4_0", "No repair / SmoothAttn.", C["none"]),
    ]
    tasks = [("gsm8k", "GSM8K", "-", "o"), ("math500", "MATH-500", "--", "^")]
    curves: dict = {}
    for task, _, ls, mk in tasks:
        for arm, _, col in arms:
            rec = reasoning_records(arm, task)
            bpt = R1_ELEMS * (K_BITS[arm] + V_BITS) / 8
            xs, ys = pareto_curve(rec, bpt, BUDGETS_FINE)
            curves[(arm, task)] = (xs, ys)
            ax.plot(
                xs, ys, ls, color=col, lw=1.0, marker=mk, ms=2.2, mec="white", mew=0.3, zorder=3
            )
            if arm != "plain_q4_0":
                # static labels sit below their end points, clear of the arrows and the 16-bit
                # curves
                dx, dy, ha = {
                    ("static_postmean", "gsm8k"): (0, -7, "center"),
                    ("static_postmean", "math500"): (2, -4, "left"),
                }.get((arm, task), (2, 0, "left"))
                ax.annotate(
                    f"{ys[-1]:.1f}",
                    (xs[-1], ys[-1]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=6,
                    color=col,
                    va="center",
                    ha=ha,
                )
    for task, target in [("gsm8k", 80), ("math500", 60)]:
        x16 = x_at_accuracy(*curves[("16bit", task)], target)
        xsp = x_at_accuracy(*curves[("spintare", task)], target)
        ax.annotate(
            "",
            xy=(xsp, target),
            xytext=(x16, target),
            arrowprops=dict(arrowstyle="<->", color=C["gate"], lw=0.8, shrinkA=0, shrinkB=0),
        )
        lab = f"{x16 / xsp:.1f}$\\times$"
        ax.text(
            math.sqrt(xsp * x16),
            target + 2,
            lab,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=C["gate"],
            fontweight="bold",
        )
    ax.set_xscale("log")
    ax.set_xlim(12, 190)
    ax.set_ylim(-3, 102)
    ax.set_xticks([15, 20, 30, 50, 100, 150])
    ax.set_xticklabels(["15", "20", "30", "50", "100", "150"])
    ax.minorticks_off()
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.set_xlabel("KV memory per generation (MiB)", fontsize=6.8, labelpad=1.5)
    ax.set_ylabel("Accuracy (%)", fontsize=6.8, labelpad=1.5)
    ax.grid(True, which="major", alpha=0.3, lw=0.4)
    ax.set_yticks([0, 25, 50, 75, 100])
    # name the two line styles in the plot instead of the legend
    xg = x_at_accuracy(*curves[("spintare", "gsm8k")], 55)
    ax.annotate(
        "GSM8K (solid)",
        xy=(xg, 55),
        xytext=(12.8, 86),
        fontsize=6.3,
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.6, shrinkA=1, shrinkB=1),
    )
    xm = x_at_accuracy(*curves[("16bit", "math500")], 30)
    ax.annotate(
        "MATH-500 (dashed)",
        xy=(xm, 30),
        xytext=(96, 12),
        fontsize=6.3,
        color="#333333",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.6, shrinkA=1, shrinkB=1),
    )
    h_arm = [mpl.lines.Line2D([], [], color=col, lw=1.3, label=lab) for _, lab, col in arms]
    ax.legend(
        handles=h_arm,
        frameon=False,
        fontsize=6,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.45, -0.2),
        handlelength=1.8,
        columnspacing=1.2,
        labelspacing=0.3,
    )
    save(fig, "fig5_7_pareto_reasoning_col")


# ------------------------------------------------------------------ RQ4: component waterfall (KL)
def tab_waterfall() -> None:
    steps = [
        ("A_nr", "q4_0, no repair"),
        ("B2_nr", "+ static mean (2K)"),
        ("D_nr", "rotary add-back, no gate"),
        ("E_nr", "+ exemption gate (SpinTare)"),
    ]
    lines = [
        "# Component waterfall, output KL at 32K (LC1 simulation, q4_0 no rotation, values exact)",
        "",
        "Ratio = KL / KL of previous row; cum = KL / KL no repair. Ratio > 1 means the step hurts.",
        "",
    ]
    for m in [
        "q25_7b",
        "q3_0p6b",
        "r1q_7b",
        "q25_1p5b",
        "llama31_8b",
        "mistral7b",
        "llama3b",
        "granite8b",
    ]:
        lines += [
            f"## {NAME[m]}",
            "",
            "| Step | KL 32K | Ratio vs prev | Cum vs no repair |",
            "|---|---|---|---|",
        ]
        prev = None
        base = LC1[m]["kl"]["A_nr"][4]
        for arm, lab in steps:
            v = LC1[m]["kl"][arm][4]
            # the rotary row replaces the static row, so its 'prev' is no repair, not static
            ref = base if arm == "D_nr" else prev
            r = "-" if ref is None else f"{v / ref:.3g}" + (" (up)" if v > ref else "")
            lines.append(f"| {lab} | {v:.4g} | {r} | {v / base:.3f} |")
            prev = v
        lines.append("")
    (OUT / "tab5_waterfall_kl.md").write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ RQ1: length table (engine)
def tab_length() -> None:
    lines = [
        "# Engine KL by context length (llama.cpp, q4_0 K / q8_0 V, PG19; E2)",
        "",
        "| Model | Arm | 2K | 4K | 8K | 16K | 32K | 64K | 128K |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    labs = {
        "q4_0": "No repair",
        "pm": "Static mean 2K",
        "abbias": "Rotary add-back (bias)",
        "abmu": "Rotary add-back (mu)",
        "q8_0": "8-bit K",
    }
    for m, mn in [
        ("q8_r1q7-q8_0", "R1-Qwen-7B"),
        ("qwen25_7b-q8_0", "Qwen2.5-7B"),
        ("qwen3_0p6b-q8_0", "Qwen3-0.6B"),
        ("llama31_8b-q8_0", "Llama-3.1-8B"),
    ]:
        for arm in ["q8_0", "q4_0", "pm", "abbias", "abmu"]:
            if arm not in E2[m]:
                continue
            row = [
                f"{E2[m][arm][L][0]:.4g}" if L in E2[m][arm] else ""
                for L in ["2048", "4096", "8192", "16384", "32768", "65536", "131072"]
            ]
            lines.append(f"| {mn} | {labs[arm]} | " + " | ".join(row) + " |")
    lines += [
        "",
        "Empty cells were not run in E2. abbias/abmu carry no exemption gate (not full SpinTare).",
    ]
    (OUT / "tab5_length_engine.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    # 2026-11-22: only the position heatmap and the Pareto plot stay as figures; the rest are tables
    # or
    # vLLM-style bars in make_ch5_bars.py. The other fig_* functions remain for their numbers.
    fig_pos_heatmap()
    fig_pos_heatmap_col()
    tab_fail_causes()
    tab_waterfall()
    tab_length()
    fig_pareto_reasoning()
    fig_pareto_combined()
    fig_pareto_combined_col()
    (OUT / "ch5_numbers.json").write_text(
        json.dumps(NUMBERS, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in NUMBERS.items() if k in ("cliff", "rho_vs_static_penalty_spearman")},
            indent=1,
        )
    )
