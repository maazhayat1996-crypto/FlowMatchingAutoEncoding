"""
Draw posterior samples from a trained FMPE run (see trainer.py) at the NuFIT 6.0
best fit, keep those inside the prior box and compute chi^2 (Eq. 3.15).

    python sample_posterior.py RUN_DIR --n 2000000 --seed 3 --chunk 200000 \
        --device cuda --tag 3

Writes RUN_DIR/samples/<tag>_G.npy, <tag>_chi2.npy and <tag>_meta.json.
Run `merge_samples.py RUN_DIR` afterwards to combine all tags into
RUN_DIR/G_samples.npy / chi2.npy for plot_posterior.py.  Different seeds give
independent draws, so this is meant to run as a Slurm job array with
--seed $SLURM_ARRAY_TASK_ID.

The functions here are also used by trainer.py for its own draws.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sbi.inference import VectorFieldPosterior
from sbi.utils import BoxUniform

import simulate as S

CHI2_CUTS = (45, 100, 2500)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def box_prior(device: str) -> BoxUniform:
    low, high = S.prior_bounds()
    return BoxUniform(low=torch.as_tensor(low, dtype=torch.float32),
                      high=torch.as_tensor(high, dtype=torch.float32), device=device)


def load_estimator(run: Path, device: str = "cpu") -> torch.nn.Module:
    """The trained vector-field estimator saved by trainer.py, moved to `device`."""
    est = torch.load(run / "estimator.pt", map_location="cpu", weights_only=False)
    return est.to(device).eval()


def build_posterior(estimator: torch.nn.Module, device: str, ode_atol: float | None = None,
                    ode_rtol: float | None = None) -> VectorFieldPosterior:
    """
    ODE-sampling posterior on `device`.  sbi 0.27 takes the ODE solver settings
    only at construction (zuko's adaptive solver: atol/rtol, no step count);
    None keeps sbi's defaults (atol=1e-6, rtol=1e-5).
    """
    ode_kwargs = {k: v for k, v in (("atol", ode_atol), ("rtol", ode_rtol)) if v is not None}
    return VectorFieldPosterior(estimator.to(device), box_prior(device), device=device,
                                sample_with="ode", neural_ode_kwargs=ode_kwargs or None)


def conditioning_point(labels: str) -> np.ndarray:
    """NuFIT 6.0 best fit in the label space the network was trained on."""
    return S.labels_from_nufit_bestfit() if labels == "L" else S.labels_T_from_nufit_bestfit()


def draw(posterior: VectorFieldPosterior, x_obs: np.ndarray, n: int, chunk: int = 200_000,
         seed: int | None = None, deadline: float | None = None,
         log: Callable[[str], None] = lambda s: print(s, flush=True)) -> dict:
    """
    Draw n samples at x_obs in chunks (accumulated on the CPU), keep the ones in
    the prior box and evaluate T and chi^2 on them.

    Sampling uses reject_outside_prior=False and filters afterwards, which gives
    the same distribution as sbi's rejection sampling but lets us report the
    fraction inside the box.  If `deadline` (a time.time() value) would be
    overrun by another chunk, stop early; the result then has n_drawn < n.
    """
    if seed is not None:
        torch.manual_seed(seed)
    low, high = S.prior_bounds()
    x = torch.as_tensor(x_obs, dtype=torch.float32)
    Gs, Ts, cs = [], [], []
    n_drawn, t0, dt = 0, time.time(), 0.0
    while n_drawn < n:
        if deadline is not None and time.time() + 1.2 * dt > deadline:
            log(f"[sample] stopping early to respect the time budget: {n_drawn}/{n} drawn")
            break
        m = min(chunk, n - n_drawn)
        t1 = time.time()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")    # "samples outside prior support"
            G = posterior.sample((m,), x=x, reject_outside_prior=False,
                                 show_progress_bars=False)
        G = G.cpu().numpy().astype(np.float64)
        G = G[np.all((G >= low) & (G <= high), axis=1)]
        T = S.observables_T(G) if len(G) else np.empty((0, 5))
        Gs.append(G); Ts.append(T); cs.append(S.chi2_from_T(T))
        n_drawn += m
        dt = time.time() - t1
        log(f"[sample] {n_drawn}/{n} drawn, {sum(map(len, Gs))} inside box, "
            f"{dt:.1f} s for this chunk")
    G, T, c = (np.concatenate(Gs) if Gs else np.empty((0, S.DIM_G)),
               np.concatenate(Ts) if Ts else np.empty((0, 5)),
               np.concatenate(cs) if cs else np.empty(0))
    return dict(G=G, T=T, chi2=c, n_requested=n, n_drawn=n_drawn, seconds=time.time() - t0)


def stats(chi2: np.ndarray, T: np.ndarray | None = None, n_drawn: int | None = None) -> dict:
    """Summary numbers for a sample set (JSON-serialisable)."""
    out: dict = {"n_inside": int(len(chi2))}
    if n_drawn is not None:
        out["n_drawn"] = int(n_drawn)
        out["frac_inside_prior"] = float(len(chi2) / n_drawn) if n_drawn else float("nan")
    if len(chi2):
        out["chi2_min"] = float(chi2.min())
        out["chi2_median"] = float(np.median(chi2))
    for cut in CHI2_CUTS:
        out[f"n_chi2_lt_{cut}"] = int(np.sum(chi2 < cut))
    if T is not None and len(T):
        med = np.median(T, axis=0)
        out["median_observables"] = {
            "dm21_meV2": float(10 ** med[0]), "dm31_meV2": float(10 ** med[1]),
            "s2_12": float(med[2]), "s2_13": float(med[3]), "s2_23": float(med[4]),
        }
    return out


def format_stats(st: dict) -> str:
    s = f"{st['n_inside']:,} samples inside the prior box"
    if "frac_inside_prior" in st:
        s += f" ({100 * st['frac_inside_prior']:.1f}% of {st['n_drawn']:,})"
    if "chi2_min" in st:
        s += f"\n  chi2: min={st['chi2_min']:.2f}  median={st['chi2_median']:.1f}"
    s += "\n  counts: " + "  ".join(f"chi2<{c}: {st[f'n_chi2_lt_{c}']:,}" for c in CHI2_CUTS)
    if "median_observables" in st:
        m = st["median_observables"]
        s += (f"\n  medians: s2_12={m['s2_12']:.3f}  s2_13={m['s2_13']:.4f}  "
              f"s2_23={m['s2_23']:.3f}  dm21={m['dm21_meV2']:.1f} meV^2  "
              f"dm31={m['dm31_meV2']:.0f} meV^2")
    return s


def save_tag(run: Path, tag: str, res: dict, meta: dict) -> None:
    d = run / "samples"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{tag}_G.npy", res["G"])
    np.save(d / f"{tag}_chi2.npy", res["chi2"])
    meta = dict(meta, tag=tag, n_requested=res["n_requested"],
                seconds=res["seconds"], **stats(res["chi2"], res["T"], res["n_drawn"]))
    (d / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))


def read_config(run: Path) -> dict:
    p = run / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}     # pre-config runs: L mode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path, help="run directory written by trainer.py")
    p.add_argument("--n", type=int, default=2_000_000, help="samples to draw (before the box cut)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--chunk", type=int, default=200_000, help="samples per ODE solve")
    p.add_argument("--device", default=default_device())
    p.add_argument("--tag", default=None, help="output prefix in RUN/samples (default: s<seed>)")
    p.add_argument("--ode-atol", type=float, default=None)
    p.add_argument("--ode-rtol", type=float, default=None)
    a = p.parse_args()
    tag = a.tag if a.tag is not None else f"s{a.seed}"

    cfg = read_config(a.run)
    labels = cfg.get("labels", "L")
    print(f"run={a.run}  labels={labels}  label_noise={cfg.get('label_noise', False)}  "
          f"device={a.device}  n={a.n}  seed={a.seed}  tag={tag}", flush=True)

    posterior = build_posterior(load_estimator(a.run), a.device, a.ode_atol, a.ode_rtol)
    res = draw(posterior, conditioning_point(labels), a.n, a.chunk, seed=a.seed)
    save_tag(a.run, tag, res, dict(seed=a.seed, device=a.device, ode_atol=a.ode_atol,
                                   ode_rtol=a.ode_rtol))
    print(format_stats(stats(res["chi2"], res["T"], res["n_drawn"])), flush=True)
    print(f"wrote {a.run / 'samples'}/{tag}_*.npy in {res['seconds']:.1f} s", flush=True)
    if a.device.startswith("cuda"):
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
              f"for --chunk {a.chunk}", flush=True)


if __name__ == "__main__":
    main()
