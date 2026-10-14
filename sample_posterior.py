"""
Draw posterior samples from a trained FMPE run (see trainer.py) at the NuFIT 6.0
best fit, keep those inside the prior box and compute chi^2 (Eq. 3.15).

    python sample_posterior.py RUN_DIR --n 300000 --seed 3 --chunk 200000 \
        --device cuda --tag 3
    python sample_posterior.py RUN_DIR --benchmark      # samples/s per chunk size

Writes RUN_DIR/samples/<tag>_G.npy, <tag>_chi2.npy and <tag>_meta.json, and
appends one JSON line per call to RUN_DIR/samples/log.jsonl.  Run
`merge_samples.py RUN_DIR` afterwards to combine all tags into
RUN_DIR/G_samples.npy / chi2.npy for plot_posterior.py.  Different seeds give
independent draws, so this is meant to run as a Slurm job array with
--seed $SLURM_ARRAY_TASK_ID.  Under Slurm, sampling stops 5 minutes before the
job's end time and saves what it has.

The functions here are also used by trainer.py for its own draws.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sbi.inference import VectorFieldPosterior
from sbi.samplers.ode_solvers.zuko_ode import ZukoNeuralODE
from sbi.utils import BoxUniform

import simulate as S

CHI2_CUTS = (45, 100, 2500)
BENCHMARK_CHUNKS = (50_000, 200_000, 500_000)

# sbi's ODE solver defaults, read from the installed version rather than copied.
_ODE_SIG = inspect.signature(ZukoNeuralODE.__init__).parameters
ODE_ATOL_DEFAULT = _ODE_SIG["atol"].default
ODE_RTOL_DEFAULT = _ODE_SIG["rtol"].default


def log_stdout(msg: str) -> None:
    print(msg, flush=True)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gpu_name(device: str) -> str | None:
    return torch.cuda.get_device_name() if device.startswith("cuda") else None


# --------------------------------------------------------------------------- #
# Slurm wall time
# --------------------------------------------------------------------------- #

def _slurm_seconds(s: str) -> float | None:
    """Parse Slurm's [D-]HH:MM:SS / MM:SS duration format."""
    s = s.strip()
    if not s or not s[0].isdigit():
        return None                                   # UNLIMITED, INVALID, ...
    days, _, hms = s.rpartition("-")
    parts = [float(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return (float(days or 0) * 24 + parts[0]) * 3600 + parts[1] * 60 + parts[2]


def slurm_end_time() -> float | None:
    """
    time.time() at which the current Slurm job will be killed, or None outside
    Slurm.  Tries $SLURM_JOB_END_TIME, then `scontrol show job` (EndTime, local
    time), then `squeue -o %L` (time left).
    """
    if os.environ.get("SLURM_JOB_END_TIME"):
        return float(os.environ["SLURM_JOB_END_TIME"])
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    if shutil.which("scontrol"):
        with contextlib.suppress(Exception):
            out = subprocess.run(["scontrol", "show", "job", job], capture_output=True,
                                 text=True, timeout=60).stdout
            for tok in out.split():
                if tok.startswith("EndTime="):
                    return time.mktime(time.strptime(tok[len("EndTime="):], "%Y-%m-%dT%H:%M:%S"))
    if shutil.which("squeue"):
        with contextlib.suppress(Exception):
            out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True,
                                 text=True, timeout=60).stdout
            secs = _slurm_seconds(out.splitlines()[0]) if out.strip() else None
            if secs:
                return time.time() + secs
    return None


# --------------------------------------------------------------------------- #
# Posterior
# --------------------------------------------------------------------------- #

def box_prior(device: str) -> BoxUniform:
    low, high = S.prior_bounds()
    return BoxUniform(low=torch.as_tensor(low, dtype=torch.float32),
                      high=torch.as_tensor(high, dtype=torch.float32), device=device)


def load_estimator(run: Path, device: str = "cpu") -> torch.nn.Module:
    """The trained vector-field estimator saved by trainer.py, moved to `device`."""
    est = torch.load(run / "estimator.pt", map_location="cpu", weights_only=False)
    return est.to(device).eval()


def build_posterior(estimator: torch.nn.Module, device: str,
                    ode_atol: float = ODE_ATOL_DEFAULT,
                    ode_rtol: float = ODE_RTOL_DEFAULT) -> VectorFieldPosterior:
    """
    ODE-sampling posterior on `device`.  sbi 0.27 takes the ODE solver settings
    only here, at construction: zuko's adaptive solver has atol/rtol and no
    step count.
    """
    return VectorFieldPosterior(estimator.to(device), box_prior(device), device=device,
                                sample_with="ode",
                                neural_ode_kwargs=dict(atol=ode_atol, rtol=ode_rtol))


def conditioning_point(labels: str = "L") -> np.ndarray:
    """NuFIT 6.0 best fit in the label space the network was trained on."""
    return S.labels_from_nufit_bestfit() if labels == "L" else S.labels_T_from_nufit_bestfit()


def draw(posterior: VectorFieldPosterior, x_obs: np.ndarray, n: int, chunk: int = 200_000,
         seed: int | None = None, deadline: float | None = None,
         log: Callable[[str], None] = log_stdout) -> dict:
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
        log(f"[sample] {n_drawn:,}/{n:,} drawn, {sum(map(len, Gs)):,} inside box, "
            f"{m / dt:.0f} samples/s in this chunk")
    G, T, c = (np.concatenate(Gs) if Gs else np.empty((0, S.DIM_G)),
               np.concatenate(Ts) if Ts else np.empty((0, 5)),
               np.concatenate(cs) if cs else np.empty(0))
    secs = time.time() - t0
    return dict(G=G, T=T, chi2=c, n_requested=n, n_drawn=n_drawn, seconds=secs,
                samples_per_s=n_drawn / secs if secs > 0 else float("nan"))


def benchmark(posterior: VectorFieldPosterior, x_obs: np.ndarray, device: str,
              chunks: tuple[int, ...] = BENCHMARK_CHUNKS, seed: int = 0,
              max_s: float = 600.0, log: Callable[[str], None] = log_stdout) -> list[dict]:
    """
    Time one chunk of each size (after a small warm-up) and report samples/s.
    A size is skipped if the rate measured so far predicts it would push the
    total past `max_s` seconds; a CUDA out-of-memory error ends the benchmark.
    """
    cuda = device.startswith("cuda")
    quiet = lambda s: None  # noqa: E731
    # Warm-up (CUDA init, kernel selection).  Its rate, from a small batch, is a
    # pessimistic estimate used to decide whether the first size fits in max_s.
    warm = draw(posterior, x_obs, min(2000, min(chunks)), seed=seed, log=quiet)
    rate = warm["samples_per_s"]
    log(f"[benchmark] warm-up {warm['n_drawn']:,} samples: {rate:.1f} samples/s")
    results: list[dict] = []
    t_start = time.time()
    for size in chunks:
        if time.time() - t_start + size / rate > max_s:
            log(f"[benchmark] chunk {size:>7,}: skipped, predicted {size / rate:.0f} s "
                f"exceeds the {max_s:.0f} s benchmark budget")
            results.append(dict(chunk=size, skipped=True))
            continue
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        try:
            res = draw(posterior, x_obs, size, chunk=size, seed=seed, log=quiet)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log(f"[benchmark] chunk {size:>7,}: CUDA out of memory; use a smaller --chunk")
            results.append(dict(chunk=size, oom=True))
            break
        rate = res["samples_per_s"]
        mem = torch.cuda.max_memory_allocated() / 2**30 if cuda else None
        r = dict(chunk=size, seconds=res["seconds"], samples_per_s=rate,
                 frac_inside_prior=len(res["chi2"]) / size, peak_gpu_mem_gib=mem)
        results.append(r)
        log(f"[benchmark] chunk {size:>7,}: {rate:8.1f} samples/s  ({res['seconds']:.1f} s, "
            f"{100 * r['frac_inside_prior']:.1f}% inside box"
            + (f", peak GPU memory {mem:.1f} GiB)" if mem is not None else ")"))
    return results


# --------------------------------------------------------------------------- #
# Statistics and output
# --------------------------------------------------------------------------- #

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
        if n_drawn:
            out[f"frac_chi2_lt_{cut}"] = float(np.sum(chi2 < cut) / n_drawn)
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
        s += f" ({100 * st['frac_inside_prior']:.1f}% of {st['n_drawn']:,} drawn)"
    if "chi2_min" in st:
        s += f"\n  chi2: min={st['chi2_min']:.2f}  median={st['chi2_median']:.1f}"
    s += "\n  counts: " + "  ".join(
        f"chi2<{c}: {st[f'n_chi2_lt_{c}']:,}"
        + (f" ({100 * st[f'frac_chi2_lt_{c}']:.3f}%)" if f"frac_chi2_lt_{c}" in st else "")
        for c in CHI2_CUTS)
    if "median_observables" in st:
        m = st["median_observables"]
        s += (f"\n  medians: s2_12={m['s2_12']:.3f}  s2_13={m['s2_13']:.4f}  "
              f"s2_23={m['s2_23']:.3f}  dm21={m['dm21_meV2']:.1f} meV^2  "
              f"dm31={m['dm31_meV2']:.0f} meV^2")
    return s


def append_log(run: Path, entry: dict) -> None:
    d = run / "samples"
    d.mkdir(parents=True, exist_ok=True)
    entry = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                 slurm_job_id=os.environ.get("SLURM_JOB_ID"), **entry)
    with open(d / "log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_tag(run: Path, tag: str, res: dict, meta: dict) -> dict:
    """Write samples/<tag>_*.npy and _meta.json, log the call; return the stats."""
    d = run / "samples"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{tag}_G.npy", res["G"])
    np.save(d / f"{tag}_chi2.npy", res["chi2"])
    st = stats(res["chi2"], res["T"], res["n_drawn"])
    meta = dict(meta, tag=tag, n_requested=res["n_requested"], seconds=res["seconds"],
                samples_per_s=res["samples_per_s"], **st)
    (d / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    append_log(run, meta)
    return st


def read_config(run: Path) -> dict:
    p = run / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}     # pre-config runs: L mode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path, help="run directory written by trainer.py")
    p.add_argument("--n", type=int, default=300_000,
                   help="samples to draw (before the box cut); with --benchmark, caps the chunk sizes")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--chunk", type=int, default=200_000, help="samples per ODE solve")
    p.add_argument("--device", default=default_device())
    p.add_argument("--tag", default=None, help="output prefix in RUN/samples (default: s<seed>)")
    p.add_argument("--ode-atol", type=float, default=ODE_ATOL_DEFAULT, help="(sbi default)")
    p.add_argument("--ode-rtol", type=float, default=ODE_RTOL_DEFAULT, help="(sbi default)")
    p.add_argument("--benchmark", action="store_true",
                   help=f"time one chunk of each size {BENCHMARK_CHUNKS} (capped at --n), then exit")
    p.add_argument("--benchmark-max-s", type=float, default=600.0,
                   help="skip benchmark sizes predicted to exceed this many seconds in total")
    a = p.parse_args()
    tag = a.tag if a.tag is not None else f"s{a.seed}"

    cfg = read_config(a.run)
    labels = cfg.get("labels", "L")
    log_stdout(f"run={a.run}  labels={labels}  device={a.device}  gpu={gpu_name(a.device)}  "
               f"torch={torch.__version__}  ode atol={a.ode_atol} rtol={a.ode_rtol}")
    posterior = build_posterior(load_estimator(a.run), a.device, a.ode_atol, a.ode_rtol)
    x_obs = conditioning_point(labels)
    common = dict(device=a.device, gpu=gpu_name(a.device), seed=a.seed,
                  ode_atol=a.ode_atol, ode_rtol=a.ode_rtol)

    if a.benchmark:
        n_given = any(s == "--n" or s.startswith("--n=") for s in sys.argv[1:])
        chunks = tuple(sorted({min(c, a.n) for c in BENCHMARK_CHUNKS})) if n_given else BENCHMARK_CHUNKS
        res = benchmark(posterior, x_obs, a.device, chunks, a.seed, a.benchmark_max_s)
        append_log(a.run, dict(common, benchmark=res))
        return

    end = slurm_end_time()
    log_stdout(f"n={a.n:,}  chunk={a.chunk:,}  seed={a.seed}  tag={tag}" +
               ("" if end is None else
                f"  (Slurm job ends {time.strftime('%H:%M:%S', time.localtime(end))})"))
    res = draw(posterior, x_obs, a.n, a.chunk, seed=a.seed,
               deadline=None if end is None else end - 300)
    st = save_tag(a.run, tag, res, dict(common, chunk=a.chunk))
    log_stdout(format_stats(st))
    log_stdout(f"{res['samples_per_s']:.1f} samples/s overall; wrote {a.run / 'samples'}/{tag}_*.npy "
               f"in {res['seconds']:.1f} s")
    if a.device.startswith("cuda"):
        log_stdout(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
                   f"for --chunk {a.chunk}")


if __name__ == "__main__":
    main()
