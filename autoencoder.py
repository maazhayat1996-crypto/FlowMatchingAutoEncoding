"""
Correlation discovery with an autoencoder, following Sec. 4 of arXiv:2608.15042.

    python autoencoder.py RUN_DIR [--latents 6,5,4,3,2] [--steps 50000]

1. Observables A = {m1, m2, m3, sin^2 th12, sin^2 th13, sin^2 th23, delta,
   alpha21, alpha31} (Eq. 4.1) of the chi^2 < 2500 samples
   (RUN_DIR/G_chi2_lt_2500.npy), encoded as in Eqs. 4.4-4.6 (--encoding paper:
   masses as min-max scaled log10(1 + m/meV), angles raw, phases/pi in (-1, 1])
   or with each phase as (cos, sin) (--encoding circular).
2. The Table 2 autoencoder is trained for every latent dimension in --latents
   (Adam + OneCycleLR max 1e-3, batch 64, 50k steps, 90/10 split, MSE).
3. The chi^2 < 45 samples (RUN_DIR/G_chi2_lt_45.npy) are encoded with the
   2-dim model, clustered with HDBSCAN, and every observable plus Sum m_i and
   m_ee is fitted by a quadratic in (z1, z2) within each cluster (Table 4).
4. Control (not in the paper): the same analysis on a copy of the chi^2 < 45
   set in which m1, delta, alpha21 and alpha31 are shuffled independently
   across samples (m2, m3, Sum m_i and m_ee recomputed), which has no
   correlations among the free quantities by construction.  R^2 values that
   the control reproduces are properties of the method, not of the physics.

Outputs in RUN_DIR/autoencoder/ (models, histories, summary.json, analysis.json)
and RUN_DIR/figures/ae_*.png.  The paper does not define its ResBlock; here it
is x + Linear(tanh(Linear(x))).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.cluster import HDBSCAN

import plot_posterior as PP          # palette and rcParams shared with the other figures
import simulate as S

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

AQUA, YELLOW, MAGENTA = "#1baf7a", "#eda100", "#e87ba4"    # categorical slots 3-5
CLUSTER_COLORS = [PP.BLUE, PP.ORANGE, AQUA, YELLOW, MAGENTA, "#008300", "#6250d6", "#e34948"]
FEATURES = ["m1", "m2", "m3", "s2_12", "s2_13", "s2_23", "delta", "alpha21", "alpha31"]
TARGETS = FEATURES + ["sum_m", "m_ee"]                      # Table 4 rows
LABELS = dict(m1="m1 [meV]", m2="m2 [meV]", m3="m3 [meV]", s2_12="sin^2 th12",
              s2_13="sin^2 th13", s2_23="sin^2 th23", delta="delta_CP [rad]",
              alpha21="alpha21 [rad]", alpha31="alpha31 [rad]", sum_m="Sum m_i [meV]",
              m_ee="m_ee [meV]")
PHASES = ("delta", "alpha21", "alpha31")


# --------------------------------------------------------------------------- #
# Observables and encodings
# --------------------------------------------------------------------------- #

def wrap(phi: np.ndarray) -> np.ndarray:
    """Phase to (-pi, pi], the paper's convention (Table 3)."""
    return np.angle(np.exp(1j * phi))


def observables(G: np.ndarray, chunk: int = 200_000) -> dict[str, np.ndarray]:
    """The quantities of Eq. 4.1 plus Sum m_i and m_ee, masses in meV, phases in (-pi, pi]."""
    parts = [S.derived_quantities(G[i:i + chunk]) for i in range(0, len(G), chunk)]
    o = {k: np.concatenate([p[k] for p in parts]) for k in FEATURES + ["sum_m", "m_ee"]}
    for k in PHASES:
        o[k] = wrap(o[k])
    return o


def mee(o: dict) -> np.ndarray:
    """Effective 0nubb mass from masses, angles and phases (PDG parametrisation)."""
    c12s, c13s = 1 - o["s2_12"], 1 - o["s2_13"]
    return np.abs(c12s * c13s * o["m1"] + o["s2_12"] * c13s * o["m2"] * np.exp(1j * o["alpha21"])
                  + o["s2_13"] * o["m3"] * np.exp(1j * (o["alpha31"] - 2 * o["delta"])))


def shuffled_control(o: dict, rng: np.random.Generator) -> dict:
    """Free quantities permuted independently; dependent masses, Sum m_i and m_ee recomputed."""
    dm21, dm31 = o["m2"] ** 2 - o["m1"] ** 2, o["m3"] ** 2 - o["m1"] ** 2
    c = dict(o)
    for k in ("m1",) + PHASES:
        c[k] = rng.permutation(o[k])
    c["m2"], c["m3"] = np.sqrt(c["m1"] ** 2 + dm21), np.sqrt(c["m1"] ** 2 + dm31)
    c["sum_m"] = c["m1"] + c["m2"] + c["m3"]
    c["m_ee"] = mee(c)
    return c


class Encoding:
    """Eqs. 4.4-4.6, with mass ranges fixed on the training set and reused when applying."""

    def __init__(self, kind: str, train: dict | None = None, state: dict | None = None):
        self.kind = kind
        if state is not None:
            self.lo, self.hi = np.array(state["lo"]), np.array(state["hi"])
        else:
            ell = np.column_stack([np.log10(1 + train[k]) for k in ("m1", "m2", "m3")])
            self.lo, self.hi = ell.min(0), ell.max(0)

    def state(self) -> dict:
        return dict(kind=self.kind, lo=self.lo.tolist(), hi=self.hi.tolist())

    def __call__(self, o: dict) -> np.ndarray:
        ell = np.column_stack([np.log10(1 + o[k]) for k in ("m1", "m2", "m3")])
        cols = [-1 + 2 * (ell - self.lo) / (self.hi - self.lo),
                np.column_stack([o["s2_12"], o["s2_13"], o["s2_23"]])]
        if self.kind == "paper":
            cols.append(np.column_stack([o[k] / np.pi for k in PHASES]))
        else:
            cols += [np.column_stack([np.cos(o[k]), np.sin(o[k])]) for k in PHASES]
        return np.hstack(cols).astype(np.float32)

    def columns(self) -> list[str]:
        base = ["m1", "m2", "m3", "s2_12", "s2_13", "s2_23"]
        if self.kind == "paper":
            return base + list(PHASES)
        return base + [f"{t}({k})" for k in PHASES for t in ("cos", "sin")]


# --------------------------------------------------------------------------- #
# Network (Table 2)
# --------------------------------------------------------------------------- #

class ResBlock(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, d))

    def forward(self, x):
        return x + self.f(x)


class Autoencoder(nn.Module):
    def __init__(self, d_in: int, latent: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.LayerNorm(32),
            *[ResBlock(32) for _ in range(4)], nn.Linear(32, latent))
        self.decoder = nn.Sequential(
            nn.Linear(latent, 32), nn.Tanh(), nn.LayerNorm(32),
            *[ResBlock(32) for _ in range(4)],
            nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, d_in), nn.Tanh())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train(X: np.ndarray, latent: int, steps: int, batch: int, lr: float, seed: int,
          device: str) -> tuple[Autoencoder, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = len(X) // 10
    Xv = torch.as_tensor(X[perm[:n_val]], device=device)
    Xt = torch.as_tensor(X[perm[n_val:]], device=device)
    model = Autoencoder(X.shape[1], latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    hist, t0 = dict(step=[], train=[], val=[]), time.time()
    run = 0.0
    for step in range(1, steps + 1):
        xb = Xt[torch.randint(len(Xt), (batch,), device=device)]
        loss = torch.mean((model(xb) - xb) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run += loss.item()
        if step % 1000 == 0:                       # paper: every 1000 steps on 10 batches
            with torch.no_grad():
                idx = torch.randint(len(Xv), (10 * batch,), device=device)
                val = torch.mean((model(Xv[idx]) - Xv[idx]) ** 2).item()
            hist["step"].append(step); hist["train"].append(run / 1000); hist["val"].append(val)
            run = 0.0
    model.eval()
    with torch.no_grad():                          # full validation set at the end
        hist["val_full"] = torch.mean((model(Xv) - Xv) ** 2).item()
    hist["seconds"] = time.time() - t0
    return model, hist


@torch.no_grad()
def encode(model: Autoencoder, X: np.ndarray, device: str) -> np.ndarray:
    return model.encoder(torch.as_tensor(X, device=device)).cpu().numpy()


@torch.no_grad()
def feature_r2(model: Autoencoder, X: np.ndarray, device: str) -> np.ndarray:
    """Per-feature reconstruction R^2 = 1 - MSE / variance on the given set."""
    Xr = model(torch.as_tensor(X, device=device)).cpu().numpy()
    return 1 - ((Xr - X) ** 2).mean(0) / X.var(0)


# --------------------------------------------------------------------------- #
# Latent-space analysis (Sec. 4.3.2)
# --------------------------------------------------------------------------- #

def quadratic_r2(z: np.ndarray, y: np.ndarray) -> float:
    """R^2 of y ~ a z1^2 + b z2^2 + c z1 z2 + d z1 + e z2 + f (Eqs. 4.7-4.10)."""
    D = np.column_stack([z[:, 0] ** 2, z[:, 1] ** 2, z[:, 0] * z[:, 1], z[:, 0], z[:, 1],
                         np.ones(len(z))])
    coef, *_ = np.linalg.lstsq(D, y, rcond=None)
    return float(1 - np.sum((y - D @ coef) ** 2) / np.sum((y - y.mean()) ** 2))


def analyse(z: np.ndarray, o: dict, min_cluster_size: int) -> dict:
    labels = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(z)
    clusters = []
    for c in sorted(set(labels) - {-1}):
        sel = labels == c
        clusters.append(dict(cluster=int(c), n=int(sel.sum()),
                             r2={k: quadratic_r2(z[sel], o[k][sel]) for k in TARGETS},
                             ranges={k: [float(o[k][sel].min()), float(o[k][sel].max())]
                                     for k in TARGETS}))
    n_in = sum(c["n"] for c in clusters)
    weighted = {k: (sum(c["n"] * c["r2"][k] for c in clusters) / n_in if n_in else float("nan"))
                for k in TARGETS}
    return dict(labels=labels.tolist(), n_noise=int(np.sum(labels == -1)), clusters=clusters,
                r2_weighted=weighted, r2_all=({k: quadratic_r2(z, o[k]) for k in TARGETS}))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_training(summary: dict, cols: list[str], out: Path) -> None:
    lat = sorted(int(k) for k in summary)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1, 1.4]))
    v = [summary[str(l)]["val_full"] for l in lat]
    ax.plot(lat, v, "-o", color=PP.BLUE, lw=2, ms=6)
    for l, vv in zip(lat, v):
        ax.annotate(f"{vv:.1e}", (l, vv), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=PP.INK2)
    ax.set_yscale("log"); ax.set_xticks(lat)
    ax.set_xlabel("latent dimension"); ax.set_ylabel("validation MSE (chi2 < 2500)")
    ax.set_title("Reconstruction error versus latent dimension")

    R = np.array([summary[str(l)]["feature_r2_chi2lt45"] for l in lat])
    im = bx.imshow(R.T, cmap=PP.SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            # features nearly fixed by the cut (e.g. sin^2 th13) have tiny variance,
            # so R^2 can be hugely negative: show those as "<0"
            bx.text(i, j, f"{R[i, j]:.2f}" if R[i, j] >= 0 else "<0", ha="center", va="center", fontsize=7,
                    color=PP.SURFACE if R[i, j] > 0.6 else PP.INK)
    bx.set_xticks(range(len(lat)), [str(l) for l in lat]); bx.set_yticks(range(len(cols)), cols)
    bx.set_xlabel("latent dimension"); bx.grid(False)
    bx.set_title("Per-feature reconstruction R^2 on the chi2 < 45 samples")
    fig.colorbar(im, ax=bx, fraction=0.04)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_latent(z: np.ndarray, o: dict, ana: dict, title: str, out: Path) -> None:
    panels = ["cluster", "m1", "m_ee", "sum_m", "delta", "alpha21", "alpha31", "s2_12",
              "s2_13", "s2_23"]
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.6))
    labels = np.array(ana["labels"])
    for ax, key in zip(axes.ravel(), panels):
        if key == "cluster":
            ax.scatter(*z[labels == -1].T, s=5, color=PP.GRID, label="noise")
            for c in sorted(set(labels) - {-1}):
                pts = z[labels == c]
                ax.scatter(*pts.T, s=6, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)])
                ax.text(*np.median(pts, 0), str(c), fontsize=10, weight="bold", color=PP.INK,
                        ha="center", va="center")
            ax.set_title(f"HDBSCAN clusters ({len(set(labels) - {-1})}, noise {ana['n_noise']})")
        else:
            cyclic = key in PHASES
            kw = dict(cmap="twilight", vmin=-np.pi, vmax=np.pi) if cyclic else \
                dict(cmap=PP.SEQ, norm=LogNorm()) if key == "m1" else dict(cmap=PP.SEQ)
            sc = ax.scatter(*z.T, c=o[key], s=5, **kw)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(LABELS[key])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=140); plt.close(fig)


def plot_r2_vs_control(data: dict, control: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(TARGETS))[::-1]
    for ana, color, label, dy in ((data, PP.BLUE, "chi2 < 45 samples", 0.12),
                                  (control, PP.ORANGE, "shuffled control (no correlations)", -0.12)):
        r = [ana["r2_weighted"][k] for k in TARGETS]
        ax.scatter(r, y + dy, color=color, s=40, label=label, zorder=3)
    for yi, k in zip(y, TARGETS):
        ax.plot([control["r2_weighted"][k], data["r2_weighted"][k]], [yi - 0.12, yi + 0.12],
                color=PP.GRID, lw=1, zorder=1)
    ax.axvline(0.8, color=PP.MUTED, ls=":", lw=1)
    ax.text(0.8, y[0] + 0.6, " paper's R^2 cut", color=PP.MUTED, fontsize=8)
    ax.set_yticks(y, [LABELS[k] for k in TARGETS]); ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("quadratic-fit R^2 in latent space (cluster-size weighted mean)")
    ax.set_title("Is the latent structure physics or method?  Data versus shuffled control")
    ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("--train-file", default="G_chi2_lt_2500.npy")
    p.add_argument("--apply-file", default="G_chi2_lt_45.npy")
    p.add_argument("--encoding", choices=["paper", "circular"], default="paper")
    p.add_argument("--latents", default="6,5,4,3,2")
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="OneCycleLR maximum")
    p.add_argument("--min-cluster-size", type=int, default=50, help="HDBSCAN (paper: not given)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default=None, help="output subdirectory (default: the encoding)")
    a = p.parse_args()
    latents = [int(x) for x in a.latents.split(",")]
    out = a.run / "autoencoder" / (a.tag or a.encoding)
    out.mkdir(parents=True, exist_ok=True)
    (a.run / "figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)

    o_tr = observables(np.load(a.run / a.train_file))
    o_ap = observables(np.load(a.run / a.apply_file))
    enc = Encoding(a.encoding, train=o_tr)
    X_tr, X_ap = enc(o_tr), enc(o_ap)
    o_ctl = shuffled_control(o_ap, rng)
    X_ctl = enc(o_ctl)
    print(f"train: {len(X_tr):,} samples ({a.train_file}), apply: {len(X_ap):,} ({a.apply_file}), "
          f"encoding {a.encoding} ({X_tr.shape[1]} features)", flush=True)

    summary = {}
    for latent in latents:
        model, hist = train(X_tr, latent, a.steps, a.batch, a.lr, a.seed, a.device)
        hist["feature_r2_chi2lt45"] = feature_r2(model, X_ap, a.device).tolist()
        torch.save(dict(state_dict=model.state_dict(), latent=latent, d_in=X_tr.shape[1],
                        encoding=enc.state(), args={k: str(v) for k, v in vars(a).items()}),
                   out / f"model_l{latent}.pt")
        summary[str(latent)] = hist
        print(f"latent {latent}: val MSE {hist['val_full']:.2e} ({hist['seconds']:.0f} s); "
              f"feature R^2 on chi2<45: "
              + " ".join(f"{c}={r:.2f}" for c, r in zip(enc.columns(), hist["feature_r2_chi2lt45"])),
              flush=True)
        if latent == 2:
            z, z_ctl = encode(model, X_ap, a.device), encode(model, X_ctl, a.device)
            ana, ana_ctl = analyse(z, o_ap, a.min_cluster_size), analyse(z_ctl, o_ctl, a.min_cluster_size)
            np.save(out / "z_chi2lt45.npy", z)
            (out / "analysis.json").write_text(json.dumps(dict(data=ana, control=ana_ctl), indent=2))
            tag = a.tag or a.encoding
            plot_latent(z, o_ap, ana, f"Latent space (l = 2, {a.encoding} encoding) of the "
                        f"{len(z):,} chi2 < 45 samples", a.run / "figures" / f"ae_latent_{tag}.png")
            plot_latent(z_ctl, o_ctl, ana_ctl, "Shuffled control: same autoencoder, free quantities "
                        "permuted independently", a.run / "figures" / f"ae_latent_control_{tag}.png")
            plot_r2_vs_control(ana, ana_ctl, a.run / "figures" / f"ae_r2_vs_control_{tag}.png")
            for name, an in (("data", ana), ("control", ana_ctl)):
                print(f"[{name}] {len(an['clusters'])} clusters "
                      f"(sizes {[c['n'] for c in an['clusters']]}, noise {an['n_noise']}); "
                      "weighted R^2: " + " ".join(f"{k}={v:.2f}" for k, v in an["r2_weighted"].items()),
                      flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_training(summary, enc.columns(), a.run / "figures" / f"ae_training_{a.tag or a.encoding}.png")
    print(f"wrote {out} and figures/ae_*_{a.tag or a.encoding}.png")


if __name__ == "__main__":
    main()
