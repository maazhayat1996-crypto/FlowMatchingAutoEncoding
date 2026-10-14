"""
Visualise a saved FMPE run (see trainer.py).

    python plot_posterior.py runs/20260924-194028 [--cut 2500]

Writes PNGs into <run>/figures/:
    chi2.png          distribution of chi^2 (Eq. 3.15) over posterior samples
    observables.png   marginals of the five fitted observables T vs NuFIT 6.0
    params.png        1D marginals of all 23 seesaw parameters, posterior vs prior
    corner_scale.png  pairwise view of the five mass-sector parameters
    physics.png       derived-quantity correlations (masses, phases, 0nubb)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import simulate as S

# ---- palette (validated reference palette, light surface) ------------------ #
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"                       # categorical slots 1, 2
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
BAND = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "legend.frameon": False, "legend.fontsize": 8,
})


def hist(ax, data, bins, color, label, density=True):
    ax.hist(data, bins=bins, density=density, histtype="step", lw=1.6, color=color, label=label)


def plot_chi2(chi2, cut, out):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    lc = np.log10(np.clip(chi2, 1e-3, None))
    ax.hist(lc, bins=80, color=BLUE, alpha=0.85, lw=0)
    for v, name in ((45, "chi2 = 45"), (2500, "chi2 = 2500")):
        ax.axvline(np.log10(v), color=ORANGE, lw=1.4, ls="--")
        ax.text(np.log10(v), ax.get_ylim()[1] * 0.96, f" {name}", color=ORANGE, va="top", fontsize=8)
    n = len(chi2)
    ax.set_xlabel("log10 chi2  (Eq. 3.15)")
    ax.set_ylabel("posterior samples")
    ax.set_title(f"Accuracy of {n:,} posterior samples: "
                 f"{np.mean(chi2 < 2500)*100:.1f}% below 2500, {int(np.sum(chi2 < 45))} below 45")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_observables(T, Tcut, cut, out):
    keys = ["dm21_meV2", "dm31_meV2", "s2_12", "s2_13", "s2_23"]
    labels = ["log10(dm21^2 / meV^2)", "log10(dm31^2 / meV^2)", "sin^2 th12", "sin^2 th13", "sin^2 th23"]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    for j, (ax, k, lab) in enumerate(zip(axes, keys, labels)):
        best, lo, hi = S.NUFIT[k]
        if j < 2:
            best, lo, hi = np.log10(best), np.log10(lo), np.log10(hi)
        # window: 3-sigma band widened so the posterior spread is visible
        w = 10 * (hi - lo)
        xmin, xmax = best - w, best + w
        if j >= 2:
            xmin, xmax = max(0, xmin), min(1, xmax)
        bins = np.linspace(xmin, xmax, 60)
        ax.axvspan(lo, hi, color=BAND, lw=0, label="NuFIT 3-sigma" if j == 0 else None)
        ax.axvline(best, color=INK2, lw=1, ls=":", label="NuFIT best fit" if j == 0 else None)
        hist(ax, np.clip(T[:, j], xmin, xmax), bins, BLUE, "all posterior samples" if j == 0 else None)
        hist(ax, np.clip(Tcut[:, j], xmin, xmax), bins, ORANGE, f"chi2 < {cut:g}" if j == 0 else None)
        ax.set_xlabel(lab); ax.set_yticks([])
        ax.grid(False)
    axes[0].set_ylabel("density (a.u.)")
    fig.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Fitted observables: posterior conditioned on NuFIT 6.0 (values clipped to window)", y=1.10, fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)


def plot_params(G, Gcut, cut, out):
    low, high = S.prior_bounds()
    fig, axes = plt.subplots(5, 5, figsize=(13, 11))
    axes = axes.ravel()
    for j in range(S.DIM_G):
        ax = axes[j]
        bins = np.linspace(low[j], high[j], 40)
        ax.axhline(1.0 / (high[j] - low[j]), color=MUTED, lw=1, ls="--", label="prior" if j == 0 else None)
        hist(ax, G[:, j], bins, BLUE, "posterior" if j == 0 else None)
        hist(ax, Gcut[:, j], bins, ORANGE, f"chi2 < {cut:g}" if j == 0 else None)
        ax.set_title(S.G_NAMES[j], fontsize=9, color=INK2)
        ax.set_yticks([]); ax.grid(False)
    for ax in axes[S.DIM_G:]:
        ax.axis("off")
    fig.legend(loc="lower right", bbox_to_anchor=(0.97, 0.06), fontsize=10)
    fig.suptitle("Seesaw parameters G: 1D marginals of the posterior versus the flat prior", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=140); plt.close(fig)


def plot_corner(Gcut, cut, out):
    idx = [18, 19, 20, 21, 22]
    names = ["|M1|/M3", "|M2|/M3", "log10(v^2/M3 / meV)", "arg M1", "arg M2"]
    low, high = S.prior_bounds()
    k = len(idx)
    fig, axes = plt.subplots(k, k, figsize=(11, 11))
    for r in range(k):
        for c in range(k):
            ax = axes[r, c]
            i, j = idx[r], idx[c]
            if c > r:
                ax.axis("off"); continue
            if r == c:
                bins = np.linspace(low[i], high[i], 40)
                ax.hist(Gcut[:, i], bins=bins, color=BLUE, alpha=0.85, lw=0)
                ax.set_yticks([]); ax.grid(False)
            else:
                ax.hist2d(Gcut[:, j], Gcut[:, i], bins=40,
                          range=[[low[j], high[j]], [low[i], high[i]]], cmap=SEQ)
                ax.grid(False)
            if r == k - 1: ax.set_xlabel(names[c])
            else: ax.set_xticklabels([])
            if c == 0 and r > 0: ax.set_ylabel(names[r])
            elif c > 0: ax.set_yticklabels([])
    fig.suptitle(f"Mass-sector parameters, posterior samples with chi2 < {cut:g}  (n = {len(Gcut):,})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=140); plt.close(fig)


def plot_physics(q, cut, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    n = len(q["m1"])

    def h2(ax, x, y, xl, yl, rng=None, logx=False, logy=False):
        xx = np.log10(x) if logx else x
        yy = np.log10(y) if logy else y
        ax.hist2d(xx, yy, bins=60, range=rng, cmap=SEQ)
        ax.set_xlabel(("log10 " if logx else "") + xl); ax.set_ylabel(("log10 " if logy else "") + yl)
        ax.grid(False)

    ax = axes[0, 0]
    h2(ax, q["m1"], q["m_ee"], "m1 / meV", "m_ee / meV", logx=True, logy=True, rng=[[-3, 2], [-1, 2]])
    ax.axhline(np.log10(36), color=ORANGE, lw=1.4, ls="--"); ax.text(-2.9, np.log10(36) + 0.05, "KamLAND-Zen 36 meV", color=ORANGE, fontsize=8)
    ax.set_title("Lightest mass vs effective 0nubb mass")

    ax = axes[0, 1]
    h2(ax, q["sum_m"], q["m_ee"], "sum m_i / meV", "m_ee / meV", rng=[[40, 130], [0, 40]])
    ax.axvline(113, color=ORANGE, lw=1.4, ls="--"); ax.text(112, 38, "DESI+CMB 113 meV ", color=ORANGE, fontsize=8, ha="right")
    ax.set_title("Sum of masses vs effective 0nubb mass")

    ax = axes[0, 2]
    h2(ax, q["delta"] / np.pi, q["alpha21"] / np.pi, "delta_CP / pi", "alpha21 / pi", rng=[[0, 2], [0, 2]])
    ax.set_title("Dirac phase vs Majorana phase alpha21")

    ax = axes[1, 0]
    h2(ax, q["delta"] / np.pi, q["alpha31"] / np.pi, "delta_CP / pi", "alpha31 / pi", rng=[[0, 2], [0, 2]])
    ax.set_title("Dirac phase vs Majorana phase alpha31")

    ax = axes[1, 1]
    h2(ax, q["m1"], q["alpha21"] / np.pi, "m1 / meV", "alpha21 / pi", logx=True, rng=[[-3, 2], [0, 2]])
    ax.set_title("Lightest mass vs alpha21")

    ax = axes[1, 2]
    bins = np.linspace(-3, 2, 60)
    for k, col, lab in (("m1", BLUE, "m1"), ("m2", ORANGE, "m2"), ("m3", "#1baf7a", "m3")):
        hist(ax, np.log10(np.clip(q[k], 1e-3, None)), bins, col, lab)
    ax.set_xlabel("log10 (m_i / meV)"); ax.set_yticks([]); ax.grid(False); ax.legend()
    ax.set_title("Light neutrino mass spectrum")

    fig.suptitle(f"Derived quantities for posterior samples with chi2 < {cut:g}  (n = {n:,})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--cut", type=float, default=2500.0, help="chi2 cut for the 'accepted' subset")
    a = p.parse_args()

    G = np.load(a.run / "G_samples.npy")
    chi2 = np.load(a.run / "chi2.npy")
    outdir = a.run / "figures"; outdir.mkdir(exist_ok=True)

    sel = chi2 < a.cut
    Gcut = G[sel]
    print(f"{len(G):,} samples, {sel.sum():,} with chi2 < {a.cut:g}")

    T = S.observables_T(G); Tcut = T[sel]
    q = S.derived_quantities(Gcut)

    plot_chi2(chi2, a.cut, outdir / "chi2.png")
    plot_observables(T, Tcut, a.cut, outdir / "observables.png")
    plot_params(G, Gcut, a.cut, outdir / "params.png")
    plot_corner(Gcut, a.cut, outdir / "corner_scale.png")
    plot_physics(q, a.cut, outdir / "physics.png")
    print("figures written to", outdir)


if __name__ == "__main__":
    main()
