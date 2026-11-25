# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
"""Mechanism figures in the style of the reference figures (scatter + layered confidence halos;
per-layer curves + layered bands).

fig_keys_scatter : one KV head of one layer, post-RoPE keys projected on (offset direction, top
residual direction
                   orthogonal to it); keys before / after subtracting the static post-RoPE mean,
                   layered 0.3-3 sigma halos, centroids, the first (sink) token marked. Three cases:
                   Qwen2.5-7B L0 (offset >> spread, broken x axis), Qwen3-0.6B L5 (sink becomes the
                   longest row), Llama-3.1-8B L31 (small offset). Data:
                   results/spintare/figure_inputs/keys3d/*.npz (dump_keys.py, 200 tokens shown, mean
                   over 512).
fig_layer_bits   : effective bits per layer (engine coordinates, q4_0) vs relative depth; collapsing vs nearly
                   lossless models (mean line, +-1 std band, min-max band, thin lines per model);
                   two example models with the worst layer marked. Data:
                   results/spintare/figure_inputs/layers.csv.
Descriptive figures; no measurement is made here.
"""

from __future__ import annotations

import csv
from collections import defaultdict
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Ellipse

# keys3d/*.npz from dump_keys.py, layers.csv from export_layers.py (both next to this file)
SRC = Path(__file__).resolve().parents[3] / "results" / "spintare" / "figure_inputs"
# figures are written next to this file unless FIG_OUT names another directory
OUT = Path(os.environ.get("FIG_OUT", Path(__file__).resolve().parent))
OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.45,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
# saturated colours (dark shade for edges / text)
BLUE, BLUE_D = "#2D4BFF", "#1325A8"
GREEN, GREEN_D = "#1FA83A", "#0D6620"
ORANGE, ORANGE_D = "#FF5A1F", "#B83100"
PURPLE = "#8E24AA"
BLACK, GRAY = "#111111", "#9A9A9A"


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)


def halo(ax, pts: np.ndarray, color: str, dark: str) -> None:
    """Layered confidence halo: nested 0.3-3 sigma ellipses, each lightly filled, so the core is
    darkest."""
    mu = pts.mean(0)
    val, vec = np.linalg.eigh(np.cov(pts.T))
    ang = np.degrees(np.arctan2(vec[1, -1], vec[0, -1]))
    sd = np.sqrt(np.maximum(val[::-1], 1e-12))
    for nsig in np.linspace(3.0, 0.3, 10):
        w, h = 2 * nsig * sd
        ax.add_patch(Ellipse(mu, w, h, angle=ang, facecolor=color, alpha=0.09, lw=0, zorder=1))
    for nsig, ls, a in ((3.0, (0, (3, 2)), 0.55), (2.0, "-", 0.8)):
        w, h = 2 * nsig * sd
        ax.add_patch(
            Ellipse(
                mu,
                w,
                h,
                angle=ang,
                facecolor="none",
                edgecolor=dark,
                lw=0.6,
                ls=ls,
                alpha=a,
                zorder=2,
            )
        )
    ax.scatter(*mu, s=22, color="white", edgecolor=dark, lw=0.9, zorder=6)


def dots(ax, pts: np.ndarray, color: str, dark: str, label: str | None) -> None:
    ax.scatter(*pts.T, s=6, color=color, edgecolor=dark, lw=0.2, alpha=0.9, zorder=3, label=label)


def star(ax, p, color: str, label: str | None) -> None:
    ax.scatter(*p, s=48, marker="*", color=color, edgecolor="white", lw=0.6, zorder=7, label=label)


def load_case(f: str, head: int):
    d = np.load(SRC / "keys3d" / f)
    k, m = d["keys"][head].astype(np.float64), d["mean"][head].astype(np.float64)
    e1 = m / np.linalg.norm(m)
    r = k[1:] - k[1:].mean(0)
    r = r - np.outer(r @ e1, e1)
    e2 = np.linalg.svd(r, full_matrices=False)[2][0]
    P = lambda x: np.stack([x @ e1, x @ e2], -1)  # noqa: E731
    out = dict(
        before=P(k[1:]),
        after=P(k[1:] - m),
        s0=P(k[:1])[0],
        s1=P(k[:1] - m)[0],
        mu=P(m[None])[0],
        mu_norm=float(np.linalg.norm(m)),
        spread=float(k[1:].std(0).mean()),
        sink0=float(np.linalg.norm(k[0])),
        sink1=float(np.linalg.norm(k[0] - m)),
        rest1=float(np.linalg.norm(k[1:] - m, axis=1).max()),
    )
    return out


def best_head(f: str) -> int:
    d = np.load(SRC / "keys3d" / f)
    k, m = d["keys"], d["mean"]
    return int(
        np.argmax(
            [np.linalg.norm(m[i]) / (k[i][1:].std(0).mean() + 1e-9) for i in range(k.shape[0])]
        )
    )


def offset_arrow(ax, mu) -> None:
    ax.annotate(
        "",
        xy=(mu[0], mu[1]),
        xytext=(0, 0),
        zorder=5,
        arrowprops=dict(arrowstyle="-|>,head_width=0.14,head_length=0.32", color=PURPLE, lw=1.0),
    )


def sink_arrow(ax, s0, s1) -> None:
    ax.annotate(
        "",
        xy=s1,
        xytext=s0,
        zorder=5,
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.14,head_length=0.32",
            color=ORANGE_D,
            lw=0.8,
            ls=(0, (3, 1.5)),
        ),
    )


def note(
    ax, text: str, color: str = BLACK, y: float = 0.04, x: float = 0.97, ha: str = "right"
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="bottom",
        fontsize=6,
        color=color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        zorder=8,
    )


def sub(ax, offset_txt: str, sink_txt: str, x: float = 0.5) -> None:
    """Two subtitle lines between the panel title and the axes: offset (purple) over sink
    (orange)."""
    kw = dict(transform=ax.transAxes, ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax.text(x, 1.075, offset_txt, color=PURPLE, **kw)
    ax.text(x, 1.01, sink_txt, color=ORANGE_D, **kw)


def fig_keys_scatter() -> None:
    fig = plt.figure(figsize=(6.3, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1], wspace=0.32)
    ga = gs[0].subgridspec(1, 2, wspace=0.07)
    a_l, a_r = fig.add_subplot(ga[0]), fig.add_subplot(ga[1])
    b, c = fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # (a) Qwen2.5-7B L0: offset dwarfs the spread -> broken x axis, two equal-width windows
    f = "Qwen2.5-7B-Instruct_L0.npz"
    A = load_case(f, best_head(f))
    for ax in (a_l, a_r):
        halo(ax, A["before"], BLUE, BLUE_D)
        halo(ax, A["after"], GREEN, GREEN_D)
        dots(ax, A["before"], BLUE, BLUE_D, None)
        dots(ax, A["after"], GREEN, GREEN_D, None)
        star(ax, A["s0"], BLACK, None)
        star(ax, A["s1"], ORANGE, None)
        ax.axhline(0, color=GRAY, lw=0.5, zorder=0)
        ax.grid(True, zorder=0)
    half = 4.0
    cx = round(A["mu"][0])
    a_l.set_xlim(-half, half)
    a_r.set_xlim(cx - half, cx + half)
    ylo = min(A["before"][:, 1].min(), A["after"][:, 1].min()) - 4
    yhi = max(A["before"][:, 1].max(), A["after"][:, 1].max()) + 7
    for ax in (a_l, a_r):
        ax.set_ylim(ylo, yhi)
    a_l.set_xticks([-3, 0, 3])
    a_r.set_xticks([cx - 3, cx, cx + 3])
    a_r.spines["left"].set_visible(False)
    a_r.tick_params(axis="y", left=False, labelleft=False)
    kw = dict(
        marker=[(-1, -1.6), (1, 1.6)],
        markersize=6,
        linestyle="none",
        color="k",
        mec="k",
        mew=0.8,
        clip_on=False,
    )
    a_l.plot([1], [0], transform=a_l.transAxes, **kw)
    a_r.plot([0], [0], transform=a_r.transAxes, **kw)
    con = ConnectionPatch(
        xyA=(0.0, 0.0),
        coordsA=a_l.transData,
        xyB=(A["mu"][0], A["mu"][1]),
        coordsB=a_r.transData,
        arrowstyle="-|>,head_width=0.14,head_length=0.32",
        color=PURPLE,
        lw=1.0,
        zorder=5,
    )
    fig.add_artist(con)
    a_l.set_ylabel("Top residual direction")
    a_l.set_title("(a) Qwen2.5-7B, layer 0", x=1.04, pad=19)
    sub(
        a_l,
        f"offset $|\\mu|$ = {A['mu_norm']:.0f}, key spread {A['spread']:.2f}",
        f"sink {A['sink0']:.0f} $\\rightarrow$ {A['sink1']:.1f} (an ordinary row)",
        x=1.04,
    )
    a_l.text(
        1.04,
        -0.12,
        "Offset direction",
        transform=a_l.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
    )

    # (b) Qwen3-0.6B L5: short sink becomes the longest row after subtraction
    f = "Qwen3-0.6B_L5.npz"
    B = load_case(f, best_head(f))
    halo(b, B["before"], BLUE, BLUE_D)
    halo(b, B["after"], GREEN, GREEN_D)
    dots(b, B["before"], BLUE, BLUE_D, "Keys")
    dots(b, B["after"], GREEN, GREEN_D, "Keys $-$ static mean")
    offset_arrow(b, B["mu"])
    sink_arrow(b, B["s0"], B["s1"])
    star(b, B["s0"], BLACK, "Sink key (token 0)")
    star(b, B["s1"], ORANGE, "Sink key $-$ static mean")
    b.axhline(0, color=GRAY, lw=0.5, zorder=0)
    b.axvline(0, color=GRAY, lw=0.5, zorder=0)
    b.grid(True, zorder=0)
    b.set_xlim(B["s1"][0] - 22, B["before"][:, 0].max() + 22)
    yb = np.abs(np.r_[B["before"][:, 1], B["after"][:, 1]]).max()
    b.set_ylim(-1.12 * yb, 1.12 * yb)
    b.set_title("(b) Qwen3-0.6B, layer 5", pad=19)
    sub(
        b,
        f"offset $|\\mu|$ = {B['mu_norm']:.0f}",
        f"sink {B['sink0']:.0f} $\\rightarrow$ {B['sink1']:.0f} (other rows $\\leq$ {B['rest1']:.0f})",
    )
    b.set_xlabel("Offset direction")

    # (c) Llama-3.1-8B L31: offset small relative to the spread
    f = "Meta-Llama-3.1-8B-Instruct_L31.npz"
    C = load_case(f, best_head(f))
    halo(c, C["before"], BLUE, BLUE_D)
    halo(c, C["after"], GREEN, GREEN_D)
    dots(c, C["before"], BLUE, BLUE_D, None)
    dots(c, C["after"], GREEN, GREEN_D, None)
    offset_arrow(c, C["mu"])
    sink_arrow(c, C["s0"], C["s1"])
    star(c, C["s0"], BLACK, None)
    star(c, C["s1"], ORANGE, None)
    c.axhline(0, color=GRAY, lw=0.5, zorder=0)
    c.axvline(0, color=GRAY, lw=0.5, zorder=0)
    c.grid(True, zorder=0)
    yc = np.abs(np.r_[C["before"][:, 1], C["after"][:, 1]]).max()
    c.set_ylim(-1.25 * yc, 1.22 * yc)
    c.set_title("(c) Llama-3.1-8B, layer 31", pad=19)
    sub(
        c,
        f"offset $|\\mu|$ = {C['mu_norm']:.1f}",
        f"sink {C['sink0']:.1f} $\\rightarrow$ {C['sink1']:.1f} (other rows $\\leq$ {C['rest1']:.1f})",
    )
    c.set_xlabel("Offset direction")

    h_, l_ = b.get_legend_handles_labels()
    fig.legend(
        h_,
        l_,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.13),
        markerscale=1.3,
        handletextpad=0.3,
        columnspacing=1.4,
    )
    save(fig, "fig_keys_scatter")


def fig_layer_bits() -> None:
    rows: dict[str, list[tuple[int, float]]] = defaultdict(list)
    label: dict[str, int] = {}
    for r in csv.DictReader(open(SRC / "layers.csv", encoding="utf-8")):
        rows[r["tag"]].append((int(r["layer"]), float(r["b"])))
        label[r["tag"]] = int(r["worst_label"])
    grid = np.linspace(0, 1, 41)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.2), gridspec_kw={"width_ratios": [1.25, 1]})
    for lab, col, dark, name in (
        (0, BLUE, BLUE_D, "Nearly lossless models"),
        (2, ORANGE, ORANGE_D, "Collapsing models"),
    ):
        curves = []
        for t, v in rows.items():
            if label[t] != lab:
                continue
            v = sorted(v)
            L = len(v)
            x = np.array([i for i, _ in v]) / (L - 1)
            y = np.array([bb for _, bb in v])
            a.plot(x, y, color=col, lw=0.45, alpha=0.28, zorder=2)
            curves.append(np.interp(grid, x, y))
        Cv = np.array(curves)
        mu, sd = Cv.mean(0), Cv.std(0)
        a.fill_between(grid, Cv.min(0), Cv.max(0), color=col, alpha=0.10, lw=0, zorder=1)
        a.fill_between(grid, mu - sd, mu + sd, color=col, alpha=0.26, lw=0, zorder=1)
        a.plot(grid, mu, color=col, lw=4.5, alpha=0.18, solid_capstyle="round", zorder=3)  # glow
        a.plot(
            grid,
            mu,
            color=dark,
            lw=1.6,
            marker="o",
            markersize=3.0,
            markevery=4,
            mfc="white",
            mec=dark,
            mew=0.9,
            zorder=4,
            label=f"{name} (n={len(Cv)})",
        )
    a.axhline(1.0, color=BLACK, lw=0.7, ls=":", zorder=2)
    a.text(
        0.985, 1.08, "1 bit", transform=a.get_yaxis_transform(), ha="right", va="bottom", fontsize=6
    )
    a.set_xlabel("Relative layer depth")
    a.set_ylabel("Effective bits (q4_0)")
    a.set_title("(a) Damage sits in a few layers")
    a.grid(True, zorder=0)
    a.set_xlim(-0.02, 1.02)
    hh, ll = a.get_legend_handles_labels()
    a.legend(
        hh[::-1],
        ll[::-1],
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        loc="lower center",
        bbox_to_anchor=(0.6, 0.02),
    )

    ex = (
        ("llama31_8b", BLUE, BLUE_D, "Llama-3.1-8B"),
        ("qwen25_7b", ORANGE, ORANGE_D, "Qwen2.5-7B"),
    )
    for t, col, dark, name in ex:
        v = sorted(rows[t])
        x = np.array([i for i, _ in v])
        y = np.array([bb for _, bb in v])
        b.plot(x, y, color=col, lw=4.0, alpha=0.18, solid_capstyle="round", zorder=2)
        b.plot(
            x,
            y,
            color=dark,
            lw=1.3,
            marker="o",
            markersize=2.8,
            mfc="white",
            mec=dark,
            mew=0.8,
            zorder=3,
            label=name,
        )
        w = int(np.argmin(y))
        b.axvline(x[w], color=dark, lw=0.9, ls="--", zorder=1)
    b.annotate(
        "worst layer 0: $-$0.27 bits",
        xy=(0, -0.27),
        xytext=(4.5, 0.55),
        fontsize=6,
        color=ORANGE_D,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=ORANGE_D, lw=0.6),
        zorder=5,
    )
    b.annotate(
        "worst layer 31: 3.00 bits",
        xy=(31, 3.0),
        xytext=(23.5, 3.72),
        fontsize=6,
        color=BLUE_D,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="-", color=BLUE_D, lw=0.6),
        zorder=5,
    )
    b.set_ylim(-0.6, 4.1)
    b.set_xlabel("Layer")
    b.set_ylabel("Effective bits (q4_0)")
    b.set_title("(b) Two examples")
    b.grid(True, zorder=0)
    b.legend(
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        loc="center right",
        bbox_to_anchor=(1.0, 0.42),
    )
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_layer_bits")


if __name__ == "__main__":
    fig_keys_scatter()
    fig_layer_bits()
    print("written to", OUT)
