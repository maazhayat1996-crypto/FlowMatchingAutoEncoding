"""
Can an autoencoder recover the dimension of the Casas-Ibarra matrix R?

    python fibre_autoencoder.py RUN_DIR [--device mps]

At fixed light-neutrino data (m_hat, U: 9 numbers) and heavy-mass ratios
(|M1|/M3, |M2|/M3), the seesaw parameters G still vary along a "fibre":
    raw G (23 parameters):  12 directions = R (6) + 6 redundant ones
        (3 charged-lepton rephasings, 2 phases of M1, M2 absorbable into Y,
         1 trade-off between the Yukawa normalisation and v^2/M3)
    gauge-fixed Yukawas:     6 directions = dim R (complex orthogonal 3x3)
Gauge fixing: Yhat = sqrt(v^2/M3) e^{-i arg M_k / 2} Y_k (so m_nu = Yhat^T
diag(1/r) Yhat with r_k = |M_k|/M3), columns rephased so row 1 is real and
positive, rows 2, 3 sign-fixed by Re Yhat_ke > 0: 15 real numbers.

The test is a conditional autoencoder (Albert et al. 2022, "explicit"
conditioning): the encoder sees (x, c), the decoder sees (z, c), with x the
parameters and c the 14-feature encoding of (m_hat, U, log r1, log r2).  The
bottleneck z only has to carry what c does not determine, so the validation
error should stop improving at latent dimension = fibre dimension: 12 for raw
G, 6 for gauge-fixed Yukawas.

Ground truth, computed independently of any network:
  * rank of the Jacobian d(c)/d(x) at sample points (fibre = dim x - rank),
  * R = sqrt(D) Yhat U m_hat^{-1/2} from Casas-Ibarra (checks R^T R = 1) and
    the TwoNN intrinsic dimension of the R samples.

Outputs: RUN_DIR/fibre_ae/summary.json, RUN_DIR/figures/fibre_ae.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import intrinsic_dim as ID
import plot_posterior as PP          # palette and rcParams shared with the other figures
import simulate as S

import matplotlib.pyplot as plt

EXPECTED = dict(gauge=6, raw=12)
PHASES = ("delta", "alpha21", "alpha31")


# --------------------------------------------------------------------------- #
# Parameters, gauge fixing, conditioning features
# --------------------------------------------------------------------------- #

def gauge_fixed_yukawa(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Yhat (N, 3, 3) complex, r (N, 3)) with m_nu = P (Yhat^T diag(1/r) Yhat) P, P a rephasing."""
    Y, _, s = S.unpack(G)
    phi = np.stack([G[:, 21], G[:, 22], np.zeros(len(G))], axis=1)
    Yh = np.sqrt(s)[:, None, None] * np.exp(-0.5j * phi)[:, :, None] * Y
    Yh = Yh * np.exp(-1j * np.angle(Yh[:, 0, :]))[:, None, :]          # row 1 real, positive
    sign = np.where(Yh[:, 1:, 0].real < 0, -1.0, 1.0)
    Yh[:, 1:, :] *= sign[:, :, None]
    r = np.stack([G[:, 18], G[:, 19], np.ones(len(G))], axis=1)
    return Yh, r


def yukawa_vector(Yh: np.ndarray) -> np.ndarray:
    """The 15 real gauge-fixed coordinates: Re Yhat_1a, Re/Im Yhat_2a, Yhat_3a."""
    return np.column_stack([Yh[:, 0].real, Yh[:, 1:].real.reshape(-1, 6), Yh[:, 1:].imag.reshape(-1, 6)])


def raw_vector(G: np.ndarray) -> np.ndarray:
    """Raw G without r1, r2 (they are in the conditioning), phases on the unit circle: 23 features."""
    return np.column_stack([G[:, :18], G[:, 20], np.cos(G[:, 21]), np.sin(G[:, 21]),
                            np.cos(G[:, 22]), np.sin(G[:, 22])])


def conditioning(masses: np.ndarray, U: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Smooth 14-feature encoding of (m_hat, U) and log10 r1, r2 (rank 11)."""
    q = S.pmns_parameters(U)
    return np.column_stack([np.log10(masses), q["s2_12"], q["s2_13"], q["s2_23"],
                            *[f(q[k]) for k in PHASES for f in (np.cos, np.sin)],
                            np.log10(r[:, :2])])


def conditioning_from_yukawa(Yh: np.ndarray, r: np.ndarray) -> np.ndarray:
    masses, U = S.takagi(np.einsum("nka,nk,nkb->nab", Yh, 1 / r, Yh))
    return conditioning(masses, U, r)


def conditioning_from_G(G: np.ndarray) -> np.ndarray:
    masses, U = S.takagi(S.light_mass_matrix(G))
    return conditioning(masses, U, np.stack([G[:, 18], G[:, 19], np.ones(len(G))], axis=1))


def casas_ibarra_R(Yh: np.ndarray, r: np.ndarray) -> np.ndarray:
    """R = sqrt(diag(1/r)) Yhat U m_hat^{-1/2}; complex orthogonal by construction."""
    masses, U = S.takagi(np.einsum("nka,nk,nkb->nab", Yh, 1 / r, Yh))
    return np.sqrt(1 / r)[:, :, None] * Yh @ U / np.sqrt(masses)[:, None, :]


def jacobian_spectra(G: np.ndarray, kind: str, h: float = 1e-6) -> np.ndarray:
    """Normalised singular values of d(conditioning)/d(parameters) at each point (central differences)."""
    out = []
    for g in G:
        if kind == "raw":
            P = g + h * np.vstack([np.eye(23), -np.eye(23)])
            C = conditioning_from_G(P)
        else:
            Yh, r = gauge_fixed_yukawa(g[None])
            v0 = np.concatenate([yukawa_vector(Yh)[0], r[0, :2]])
            P = v0 + h * np.vstack([np.eye(17), -np.eye(17)])
            Yp = np.zeros((len(P), 3, 3), complex)
            Yp[:, 0] = P[:, :3]
            Yp[:, 1:] = (P[:, 3:9] + 1j * P[:, 9:15]).reshape(-1, 2, 3)
            rp = np.column_stack([P[:, 15], P[:, 16], np.ones(len(P))])
            C = conditioning_from_yukawa(Yp, rp)
        n = len(P) // 2
        sv = np.linalg.svd((C[:n] - C[n:]).T / (2 * h), compute_uv=False)
        out.append(sv / sv[0])
    return np.array(out)


# --------------------------------------------------------------------------- #
# Conditional autoencoder
# --------------------------------------------------------------------------- #

def mlp(d_in: int, d_out: int, width: int, depth: int) -> nn.Sequential:
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, d_out))


class ConditionalAE(nn.Module):
    def __init__(self, dx: int, dc: int, latent: int, width: int, depth: int):
        super().__init__()
        self.latent = latent
        self.encoder = mlp(dx + dc, latent, width, depth) if latent > 0 else None
        self.decoder = mlp(latent + dc, dx, width, depth)

    def forward(self, x, c):
        if self.latent == 0:                      # decoder sees only c: E[x | c]
            return self.decoder(c)
        return self.decoder(torch.cat([self.encoder(torch.cat([x, c], 1)), c], 1))


def fit(x: np.ndarray, c: np.ndarray, latent: int, a: argparse.Namespace) -> dict:
    """Train on 90%, return the validation fraction of variance unexplained (FVU)."""
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(x))
    n_val = len(x) // 10
    mu, sd = x[perm[n_val:]].mean(0), x[perm[n_val:]].std(0)
    cm, cs = c[perm[n_val:]].mean(0), c[perm[n_val:]].std(0)
    X = torch.as_tensor((x - mu) / sd, dtype=torch.float32, device=a.device)
    C = torch.as_tensor((c - cm) / cs, dtype=torch.float32, device=a.device)
    tr = torch.as_tensor(perm[n_val:], device=a.device)
    va = torch.as_tensor(perm[:n_val], device=a.device)
    model = ConditionalAE(x.shape[1], c.shape[1], latent, a.width, a.depth).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps)
    t0 = time.time()
    for _ in range(a.steps):
        idx = tr[torch.randint(len(tr), (a.batch,), device=a.device)]
        loss = torch.mean((model(X[idx], C[idx]) - X[idx]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        err = torch.zeros(x.shape[1], device=a.device)
        for i in range(0, len(va), 65536):
            j = va[i:i + 65536]
            err += ((model(X[j], C[j]) - X[j]) ** 2).sum(0)
        fvu_feat = (err / len(va)).cpu().numpy()            # features are standardised
    return dict(latent=latent, fvu=float(fvu_feat.mean()), fvu_features=fvu_feat.tolist(),
                seconds=time.time() - t0)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def plot(res: dict, spectra: dict, out: Path) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for kind, color, label in (("gauge", PP.BLUE, "gauge-fixed Yukawas (15 features)"),
                               ("raw", PP.ORANGE, "raw parameters G (23 features)")):
        rows = res["sweep"][kind]
        l = [r["latent"] for r in rows]; f = [r["fvu"] for r in rows]
        ax.plot(l, f, "-o", color=color, lw=2, ms=6, label=label)
        e = EXPECTED[kind]
        ax.axvline(e, color=color, lw=1, ls=":")
        ax.text(e + 0.15, max(f) * 0.6, f"fibre dim {e}", color=PP.INK2, fontsize=8, rotation=90, va="top")
    ax.set_yscale("log")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("validation error (fraction of variance unexplained)")
    ax.set_title("Conditional autoencoder: decoder also sees (m_hat, U, M ratios)")
    ax.legend(loc="upper right")

    for kind, color, n_par in (("gauge", PP.BLUE, 17), ("raw", PP.ORANGE, 23)):
        sv = np.median(spectra[kind], axis=0)
        idx = np.arange(1, len(sv) + 1)
        bx.plot(idx, sv, "-o", color=color, lw=2, ms=5,
                label=f"{'gauge-fixed' if kind == 'gauge' else 'raw'}: {n_par} parameters")
    bx.axvline(11.5, color=PP.MUTED, lw=1, ls=":")
    bx.text(11.6, 1e-3, " rank 11 = 9 (m_hat, U) + 2 ratios", color=PP.MUTED, fontsize=8)
    bx.set_yscale("log"); bx.set_ylim(1e-18, 3)
    bx.set_xlabel("singular value index")
    bx.set_ylabel("singular value / largest (median over points)")
    bx.set_title("Ground truth: Jacobian of observables w.r.t. parameters")
    bx.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("--file", default="G_chi2_lt_2500.npy")
    p.add_argument("--max-n", type=int, default=1_000_000)
    p.add_argument("--latents-gauge", default="0,1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--latents-raw", default="0,4,8,10,11,12,13,14,16")
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3, help="OneCycleLR maximum")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    out = a.run / "fibre_ae"
    out.mkdir(parents=True, exist_ok=True)
    (a.run / "figures").mkdir(exist_ok=True)

    G = np.load(a.run / a.file)
    if len(G) > a.max_n:
        G = G[rng.choice(len(G), a.max_n, replace=False)]
    Yh, r = gauge_fixed_yukawa(G)
    c = np.concatenate([conditioning_from_yukawa(Yh[i:i + 200_000], r[i:i + 200_000])
                        for i in range(0, len(G), 200_000)])
    feats = dict(gauge=yukawa_vector(Yh), raw=raw_vector(G))

    # ground truth
    pts = G[rng.choice(len(G), 200, replace=False)]
    spectra = {k: jacobian_spectra(pts, k) for k in ("gauge", "raw")}
    ranks = {k: int(np.median(np.sum(spectra[k] > 1e-6, axis=1))) for k in spectra}
    R = casas_ibarra_R(Yh[:20_000], r[:20_000])
    orth = float(np.abs(np.swapaxes(R, 1, 2) @ R - np.eye(3)).max())
    R_feat = np.column_stack([R.real.reshape(-1, 9), R.imag.reshape(-1, 9)])
    id_R = ID.twonn(R_feat)
    print(f"{len(G):,} samples ({a.file}); Jacobian rank: gauge {ranks['gauge']} of 17 -> fibre "
          f"{17 - ranks['gauge']}, raw {ranks['raw']} of 23 -> fibre {23 - ranks['raw']}; "
          f"Casas-Ibarra max|R^T R - 1| = {orth:.1e}, TwoNN ID of R = {id_R:.2f}", flush=True)

    res = dict(n=len(G), file=a.file, jacobian_rank=ranks, fibre_dim_truth=dict(gauge=17 - ranks["gauge"],
               raw=23 - ranks["raw"]), R_orthogonality=orth, twonn_R=id_R,
               args={k: str(v) for k, v in vars(a).items()}, sweep={})
    for kind, lat in (("gauge", a.latents_gauge), ("raw", a.latents_raw)):
        res["sweep"][kind] = []
        for latent in [int(v) for v in lat.split(",")]:
            row = fit(feats[kind], c, latent, a)
            res["sweep"][kind].append(row)
            print(f"[{kind}] latent {latent:2d}: validation FVU {row['fvu']:.2e}  ({row['seconds']:.0f} s)",
                  flush=True)
            (out / "summary.json").write_text(json.dumps(res, indent=2))
    plot(res, spectra, a.run / "figures" / "fibre_ae.png")
    print(f"wrote {out}/summary.json and figures/fibre_ae.png")


if __name__ == "__main__":
    main()
