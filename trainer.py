"""
Flow-matching posterior estimation for the Type-I seesaw, following
arXiv:2608.15042 (Sec. 3).  Forward model lives in simulate.py.

Usage
-----
    python trainer.py --quick            # ~1 min smoke test on CPU
    python trainer.py                    # full run (1e6 sims, paper-like settings)
    python trainer.py --n-sims 10000000 --n-post 3000000   # paper scale

Outputs (in --out, default ./runs/<timestamp>):
    estimator.pt      trained vector-field estimator (torch.save)
    posterior.pt      the sbi posterior object (pickle via torch.save)
    G_samples.npy     posterior samples of G conditioned on the NuFIT best fit
    chi2.npy          chi^2 (Eq. 3.15) of every posterior sample
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from sbi.inference import FMPE
from sbi.neural_nets import posterior_flow_nn
from sbi.utils import BoxUniform

import simulate as S


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    p.add_argument("--n-sims", type=int, default=1_000_000)
    p.add_argument("--n-post", type=int, default=100_000, help="posterior samples to draw")
    p.add_argument("--max-epochs", type=int, default=2**31 - 1)
    p.add_argument("--stop-after", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    if a.quick:
        a.n_sims, a.n_post, a.max_epochs, a.stop_after = 20_000, 2_000, 3, 2
    if a.out is None:
        a.out = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    return a


def main() -> None:
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    print(f"device={a.device}  n_sims={a.n_sims}  out={a.out}")

    # ---- prior on G (Eq. 3.2) -------------------------------------------- #
    low, high = S.prior_bounds()
    prior = BoxUniform(low=torch.as_tensor(low, dtype=torch.float32),
                       high=torch.as_tensor(high, dtype=torch.float32),
                       device=a.device)               # sbi requires prior on the training device

    # ---- simulate (Eq. 3.16) ---------------------------------------------- #
    t0 = time.time()
    theta_np = S.sample_prior(a.n_sims, rng)                    # (N, 23) float64
    x_np = S.simulate_chunked(theta_np)                         # (N, 11)
    theta = torch.as_tensor(theta_np, dtype=torch.float32)
    x = torch.as_tensor(x_np, dtype=torch.float32)
    print(f"simulated {a.n_sims} points in {time.time()-t0:.1f} s")

    # ---- vector-field network (paper: transformer, no z-scoring of G) ----- #
    vf_builder = posterior_flow_nn(
        model="transformer",
        hidden_features=100, num_layers=5,
        z_score_theta="none",        # G already lives in a bounded box
        z_score_x="independent",     # Eq. 3.17: standardise each label
    )

    # ---- train ------------------------------------------------------------ #
    trainer = FMPE(prior=prior, vf_estimator=vf_builder, device=a.device)
    t0 = time.time()
    estimator = trainer.append_simulations(theta, x).train(
        learning_rate=a.lr,
        training_batch_size=a.batch,
        validation_fraction=0.1,
        stop_after_epochs=a.stop_after,
        max_num_epochs=a.max_epochs,
        show_train_summary=True,
    )
    print(f"trained in {(time.time()-t0)/60:.1f} min")
    torch.save(estimator, a.out / "estimator.pt")

    # ---- posterior conditioned on NuFIT best fit -------------------------- #
    posterior = trainer.build_posterior(estimator)
    torch.save(posterior, a.out / "posterior.pt")
    L_obs = torch.as_tensor(S.labels_from_nufit_bestfit(), dtype=torch.float32)  # (1, 11)

    t0 = time.time()
    G = posterior.sample((a.n_post,), x=L_obs).cpu().numpy().astype(np.float64)
    print(f"drew {a.n_post} posterior samples in {time.time()-t0:.1f} s")

    # ---- evaluate (Eq. 3.15) ---------------------------------------------- #
    low, high = S.prior_bounds()
    inside = np.all((G >= low) & (G <= high), axis=1)
    G_in = G[inside]
    chi2 = S.chi2(G_in) if len(G_in) else np.empty(0)
    np.save(a.out / "G_samples.npy", G_in)
    np.save(a.out / "chi2.npy", chi2)

    print(f"samples inside prior box: {inside.mean()*100:.1f}%")
    if len(chi2):
        q = S.derived_quantities(G_in)
        print(f"chi2: min={chi2.min():.1f}  median={np.median(chi2):.1f}  "
              f"frac<2500={np.mean(chi2 < 2500):.3f}  frac<45={np.mean(chi2 < 45):.4f}")
        print(f"medians: s2_12={np.median(q['s2_12']):.3f}  s2_13={np.median(q['s2_13']):.4f}  "
              f"s2_23={np.median(q['s2_23']):.3f}  dm21={np.median(q['dm21']):.1f} meV^2  "
              f"dm31={np.median(q['dm31']):.0f} meV^2")
    print(f"saved to {a.out}")


if __name__ == "__main__":
    main()
