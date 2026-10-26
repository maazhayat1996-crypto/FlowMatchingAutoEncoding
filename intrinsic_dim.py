"""
Intrinsic dimension of the accurate posterior samples in observable space.

    python intrinsic_dim.py RUN_DIR [--cut 45]

Uses RUN_DIR/G_samples.npy + chi2.npy if present (any --cut), otherwise
RUN_DIR/G_chi2_lt_45.npy (--cut <= 45).  Writes
    RUN_DIR/intrinsic_dim_chi2lt<cut>.json
    RUN_DIR/figures/intrinsic_dim_chi2lt<cut>.png

The nine observables of the paper's autoencoder (Eq. 4.1) are embedded in four
ways, because the estimate depends on the metric:
    natural      log10 m_i, sin^2 theta_ij, each phase as (cos, sin)   [12 dims]
    free         log10 m1 and the three phases only                    [ 7 dims]
    standardised natural, each column scaled to unit variance          [12 dims]
    paper        Eqs. 4.4-4.6: min-max scaled log masses, phases / pi  [ 9 dims]
In natural units the five observables fixed by the chi^2 cut are thin
(NuFIT-width) directions, so at resolvable scales the estimate counts the
unconstrained ones; standardising stretches the thin directions to unit size.

Estimators: TwoNN (Facco et al. 2017), the Levina-Bickel MLE (k nearest
neighbours, MacKay-Ghahramani average), the local slope of the correlation
integral (Grassberger-Procaccia) as a function of scale, and TwoNN on random
subsets of N/f points (larger f probes larger scales).  Two nulls with the
same N and marginals calibrate them: the free quantities shuffled
independently (true ID 4) and the paper's relations alpha21 = 0,
alpha31 = 2 delta imposed exactly (true ID 2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

import plot_posterior as PP          # palette and rcParams shared with the other figures
import simulate as S

import matplotlib.pyplot as plt

AQUA = "#1baf7a"                     # categorical slot 3
MLE_KS = (5, 10, 20)
DECIMATION = (1, 2, 4, 8)


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #

def knn_dists(X: np.ndarray, k: int) -> np.ndarray:
    """Distances to the k nearest neighbours of every point (self excluded)."""
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return d[:, 1:]


def twonn(X: np.ndarray, discard: float = 0.1) -> float:
    """Facco et al. 2017: fit -log(1 - F(mu)) = d log(mu) through the origin, mu = r2 / r1."""
    r = knn_dists(X, 2)
    ok = r[:, 0] > 0
    mu = np.sort(r[ok, 1] / r[ok, 0])
    n = len(mu)
    keep = int((1 - discard) * n)
    x = np.log(mu[:keep])
    y = -np.log(1 - np.arange(1, keep + 1) / n)
    return float(np.sum(x * y) / np.sum(x * x))


def mle_levina_bickel(X: np.ndarray, k: int) -> float:
    """Levina & Bickel 2004 with the MacKay-Ghahramani average of inverse local estimates."""
    T = knn_dists(X, k)
    T = np.where(T > 0, T, np.nan)
    inv = np.nanmean(np.log(T[:, -1:] / T[:, :-1]), axis=1)
    return float(1.0 / np.nanmean(inv))


def correlation_dimension(X: np.ndarray, rng: np.random.Generator, max_points: int = 4000,
                          quantiles: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Local slope d log C / d log r of the correlation integral, at pair-distance
    quantiles q (so C(r_q) = q).  Uses at most `max_points` points (O(n^2) pairs).
    """
    if len(X) > max_points:
        X = X[rng.choice(len(X), max_points, replace=False)]
    d = pdist(X)
    d = d[d > 0]
    # stop at the 20% quantile: beyond it distances approach the size of the set
    # (phases live on unit circles, chord <= 2) and the slope no longer measures dimension
    q = np.geomspace(1e-3, 0.2, 14) if quantiles is None else quantiles
    r = np.quantile(d, q)
    return r, np.gradient(np.log(q), np.log(r))


def decimated_twonn(X: np.ndarray, rng: np.random.Generator) -> list[dict]:
    """TwoNN on random subsets of N/f points, with the median 1st-neighbour distance as scale."""
    out = []
    for f in DECIMATION:
        ids, scales = [], []
        for _ in range(f):
            Xs = X[rng.choice(len(X), len(X) // f, replace=False)]
            ids.append(twonn(Xs))
            scales.append(float(np.median(knn_dists(Xs, 1))))
        out.append(dict(f=f, scale=float(np.median(scales)), id=float(np.mean(ids))))
    return out


def estimate(X: np.ndarray, rng: np.random.Generator) -> dict:
    r, slope = correlation_dimension(X, rng)
    dec = decimated_twonn(X, rng)
    est = dict(twonn=twonn(X), **{f"mle_k{k}": mle_levina_bickel(X, k) for k in MLE_KS})
    point = [est["twonn"], *(est[f"mle_k{k}"] for k in MLE_KS), *(d["id"] for d in dec)]
    return dict(n=len(X), ambient=X.shape[1], **est, decimated=dec,
                corr_dim=dict(r=r.tolist(), slope=slope.tolist()),
                nn_median=float(np.median(knn_dists(X, 1))),
                range=[float(min(point)), float(max(point))])


# --------------------------------------------------------------------------- #
# Representations
# --------------------------------------------------------------------------- #

def observables(G: np.ndarray) -> dict[str, np.ndarray]:
    q = S.derived_quantities(G)
    return dict(lm1=np.log10(q["m1"]), lm2=np.log10(q["m2"]), lm3=np.log10(q["m3"]),
                s12=q["s2_12"], s13=q["s2_13"], s23=q["s2_23"],
                delta=q["delta"], a21=q["alpha21"], a31=q["alpha31"])


def _circle(phi: np.ndarray) -> list[np.ndarray]:
    return [np.cos(phi), np.sin(phi)]


def natural(o: dict) -> np.ndarray:
    return np.column_stack([o["lm1"], o["lm2"], o["lm3"], o["s12"], o["s13"], o["s23"],
                            *_circle(o["delta"]), *_circle(o["a21"]), *_circle(o["a31"])])


def free(o: dict) -> np.ndarray:
    return np.column_stack([o["lm1"], *_circle(o["delta"]), *_circle(o["a21"]), *_circle(o["a31"])])


def standardised(o: dict) -> np.ndarray:
    X = natural(o)
    return (X - X.mean(0)) / X.std(0)


def paper(o: dict) -> np.ndarray:
    """Eqs. 4.4-4.6 of arXiv:2608.15042, with m_min/m_max taken from the sample."""
    cols = []
    for k in ("lm1", "lm2", "lm3"):
        ell = np.log10(1 + 10 ** o[k])
        cols.append(-1 + 2 * (ell - ell.min()) / (ell.max() - ell.min()))
    cols += [o["s12"], o["s13"], o["s23"]]
    cols += [np.angle(np.exp(1j * o[k])) / np.pi for k in ("delta", "a21", "a31")]
    return np.column_stack(cols)


def null_shuffled(o: dict, rng: np.random.Generator) -> dict:
    """Free quantities permuted independently: same marginals, no relations (ID 4)."""
    return dict(o, **{k: rng.permutation(o[k]) for k in ("lm1", "delta", "a21", "a31")})


def null_paper_relations(o: dict, rng: np.random.Generator) -> dict:
    """alpha21 = 0 and alpha31 = 2 delta imposed on shuffled (m1, delta): ID 2."""
    d = rng.permutation(o["delta"])
    return dict(o, lm1=rng.permutation(o["lm1"]), delta=d,
                a21=np.zeros_like(d), a31=np.mod(2 * d, 2 * np.pi))


def relation_stats(o: dict) -> dict:
    """Circular statistics for the paper's two claimed relations (0 = absent, 1 = exact)."""
    return dict(alpha21_resultant_length=float(abs(np.mean(np.exp(1j * o["a21"])))),
                coherence_2delta_minus_alpha31=float(abs(np.mean(np.exp(1j * (2 * o["delta"] - o["a31"]))))))


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

ROWS = [  # key, label, colour
    ("natural", "data, natural units", PP.BLUE),
    ("free", "data, free quantities only", PP.BLUE),
    ("standardised", "data, standardised", PP.BLUE),
    ("paper", "data, paper's encoding", PP.BLUE),
    ("null_shuffled", "null: no relations (true 4)", PP.ORANGE),
    ("null_paper", "null: paper's relations (true 2)", AQUA),
]


def plot(res: dict, cut: float, out: Path) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.1, 1]))

    # (a) local dimension versus scale, data against the two nulls (shared natural units)
    for key, label, color, ls, dy in (("free", "data", PP.BLUE, "-", 10),
                                      ("null_shuffled", "null, no relations (true 4)", PP.ORANGE, "--", -14),
                                      ("null_paper", "null, paper's relations (true 2)", AQUA, "--", 10)):
        r = np.array(res[key]["corr_dim"]["r"]); s = np.array(res[key]["corr_dim"]["slope"])
        ax.plot(r, s, ls, color=color, lw=2, marker="o", ms=4, label=label)
        ax.annotate(label, (r[0], s[0]), xytext=(0, dy), textcoords="offset points",
                    va="center", fontsize=8, color=PP.INK2)
    ax.axvline(res["free"]["nn_median"], color=PP.MUTED, lw=1, ls=":")
    ax.text(res["free"]["nn_median"], 0.3, " median nearest-\n neighbour distance",
            color=PP.MUTED, fontsize=7.5, va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0, 6.5)
    ax.set_yticks(range(0, 7))
    ax.set_xlabel("pair distance r  (log10 m1 and phases on the unit circle)")
    ax.set_ylabel("local dimension  d log C / d log r")
    ax.set_title("Dimension versus scale (correlation integral)")
    ax.legend(loc="upper right")

    # (b) all estimators per representation: dot = TwoNN, whisker = spread over estimators/scales
    y = np.arange(len(ROWS))[::-1]
    for yi, (key, label, color) in zip(y, ROWS):
        e = res[key]
        lo, hi = e["range"]
        bx.plot([lo, hi], [yi, yi], color=color, lw=2, solid_capstyle="round")
        bx.plot([e["twonn"]], [yi], "o", color=color, ms=8, mec=PP.SURFACE, mew=2)
        bx.text(hi + 0.15, yi, f"{e['twonn']:.2f}", va="center", fontsize=8, color=PP.INK2)
    for v in (2, 4):
        bx.axvline(v, color=PP.MUTED, lw=1, ls=":")
    bx.set_yticks(y)
    bx.set_yticklabels([f"{lbl}  [{res[k]['ambient']}D]" for k, lbl, _ in ROWS])
    bx.set_xlim(0, 10)
    bx.set_xlabel("intrinsic dimension")
    bx.set_title("Estimates by representation\n(dot: TwoNN, line: range over all estimators)", fontsize=10)
    bx.grid(axis="y", visible=False)

    fig.suptitle(f"Intrinsic dimension of posterior samples with chi2 < {cut:g}  "
                 f"(n = {res['natural']['n']:,})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def load(run: Path, cut: float) -> np.ndarray:
    if (run / "G_samples.npy").exists() and (run / "chi2.npy").exists():
        G, c = np.load(run / "G_samples.npy"), np.load(run / "chi2.npy")
        return G[c < cut]
    if cut <= 45 and (run / "G_chi2_lt_45.npy").exists():
        G = np.load(run / "G_chi2_lt_45.npy")
        return G[S.chi2(G) < cut]
    raise SystemExit(f"{run}: need G_samples.npy + chi2.npy, or G_chi2_lt_45.npy for --cut <= 45")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("--cut", type=float, default=45.0, help="chi2 cut defining the sample set")
    p.add_argument("--max-n", type=int, default=20_000, help="random subsample above this size")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    G = load(a.run, a.cut)
    if len(G) > a.max_n:
        G = G[rng.choice(len(G), a.max_n, replace=False)]
    o = observables(G)

    res = {key: estimate(fn(o), rng) for key, fn in
           (("natural", natural), ("free", free), ("standardised", standardised), ("paper", paper))}
    res["null_shuffled"] = estimate(free(null_shuffled(o, rng)), rng)
    res["null_paper"] = estimate(free(null_paper_relations(o, rng)), rng)
    res["relations"] = relation_stats(o)
    res["cut"], res["seed"] = a.cut, a.seed

    for key, label, _ in ROWS:
        e = res[key]
        print(f"{label:<34} [{e['ambient']:>2}D]  TwoNN {e['twonn']:5.2f}   MLE "
              + " ".join(f"{e[f'mle_k{k}']:5.2f}" for k in MLE_KS)
              + f"   range {e['range'][0]:.2f}-{e['range'][1]:.2f}")
    rel = res["relations"]
    print(f"alpha21 resultant length {rel['alpha21_resultant_length']:.3f}, "
          f"coherence of 2 delta - alpha31 {rel['coherence_2delta_minus_alpha31']:.3f}  (1 = exact relation)")

    tag = f"chi2lt{a.cut:g}"
    (a.run / f"intrinsic_dim_{tag}.json").write_text(json.dumps(res, indent=2))
    (a.run / "figures").mkdir(exist_ok=True)
    plot(res, a.cut, a.run / "figures" / f"intrinsic_dim_{tag}.png")
    print(f"wrote {a.run}/intrinsic_dim_{tag}.json and figures/intrinsic_dim_{tag}.png")


if __name__ == "__main__":
    main()
