"""
Flow-matching posterior estimation for the Type-I seesaw, following
arXiv:2608.15042 (Sec. 3).  Forward model lives in simulate.py.

Usage
-----
    python trainer.py --quick            # ~1 min smoke test on CPU
    python trainer.py                    # full run (1e6 sims, paper-like settings)
    python trainer.py --n-sims 10000000 --n-post 3000000 \
        --sims-cache $FM_RUNS_DIR/sims_1e7_seed0.npz      # paper scale (Spartan)
    python trainer.py --resume RUN_DIR   # continue an interrupted run

Training runs in chunks of --epochs-per-chunk epochs and checkpoints after each.
If the wall-time budget (--time-budget-h, or the Slurm job end time minus 20 min)
would be exceeded before sbi's early stopping fires, the run saves and exits
with status 3; `--resume RUN_DIR` continues it (spartan/train.slurm resubmits
itself automatically).

Label modes (they define different posteriors; recorded in config/summary):
    --labels L                 paper default: 11-dim deterministic label L(G)
    --labels T --label-noise   5-dim T(G) plus Gaussian noise with the sigmas of
                               chi2(), i.e. the Bayesian posterior given the
                               NuFIT uncertainties

Outputs (in --out; default $FM_RUNS_DIR/<timestamp>-<jobid> or runs/<timestamp>):
    config.json           all settings (read back by --resume / sample_posterior.py)
    checkpoint.pkl        inference object without the training data (for --resume)
    estimator.pt          trained vector-field estimator, on CPU (torch.save)
    estimator_state.pt    its state_dict
    posterior.pt          the sbi posterior object, on CPU
    samples/train_*.npy   this run's posterior draws (more via sample_posterior.py)
    G_samples.npy         all draws in samples/ that lie in the prior box (merged)
    chi2.npy              chi^2 (Eq. 3.15) of every sample in G_samples.npy
    G_chi2_lt_45.npy, G_chi2_lt_2500.npy
    summary.json          epochs, losses, wall times, chi^2 statistics
    tb/                   TensorBoard logs
Load .pt files with torch.load(path, map_location="cpu", weights_only=False).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sbi.neural_nets import posterior_flow_nn
from sbi.utils.tracking import TensorBoardTracker
from torch.utils.tensorboard.writer import SummaryWriter

import merge_samples as MS
import sample_posterior as SP
import simulate as S
from fmpe_ext import ChunkedFMPE, load_checkpoint

T_START = time.time()
EXIT_BUDGET = 3          # exit status: time budget exhausted, resume me
SAFETY_MARGIN_S = 20 * 60

# Settings that may be changed when resuming; everything else comes from config.json.
RUNTIME_KEYS = {"device", "data_device", "workers", "time_budget_h", "epochs_per_chunk",
                "n_post", "sample_chunk", "ode_atol", "ode_rtol", "max_epochs", "logdir"}
QUICK = dict(n_sims=20_000, n_post=2_000, max_epochs=3, stop_after=2)


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def build_parser(suppress: bool = False) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FMPE for the Type-I seesaw (arXiv:2608.15042)",
                                argument_default=argparse.SUPPRESS if suppress else None)

    def add(*names, **kw):
        if suppress:
            kw.pop("default", None)
        p.add_argument(*names, **kw)

    add("--quick", action="store_true", default=False,
        help=f"tiny run to check the pipeline: {QUICK}")
    add("--resume", type=Path, default=None, metavar="RUN_DIR",
        help="continue an interrupted run from RUN_DIR/checkpoint.pkl + config.json")
    # data
    add("--n-sims", type=int, default=1_000_000)
    add("--seed", type=int, default=0)
    add("--labels", choices=["L", "T"], default="L",
        help="L: 11-dim deterministic label (paper); T: the 5 observables of Eq. 3.14")
    add("--label-noise", action="store_true", default=False,
        help="with --labels T: add Gaussian noise with the chi2() sigmas to the labels")
    add("--sims-cache", type=Path, default=None, metavar="PATH.npz",
        help="load the first n_sims simulations from here, or generate and save them")
    add("--sim-chunk", type=int, default=100_000,
        help="simulations per seeded chunk (part of the data's identity: keep fixed)")
    add("--workers", type=int, default=None,
        help="simulation processes / torch CPU threads (default: SLURM_CPUS_PER_TASK or all)")
    # network and optimiser
    add("--hidden", type=int, default=100)
    add("--layers", type=int, default=5)
    add("--batch", type=int, default=2048)
    add("--lr", type=float, default=5e-4)
    add("--val-frac", type=float, default=0.1)
    add("--stop-after", type=int, default=20, help="sbi early-stopping patience (epochs)")
    add("--max-epochs", type=int, default=2**31 - 1)
    add("--clip-grad", type=float, default=5.0, help="max grad norm (sbi default); <=0 disables")
    add("--fast-loader", action=argparse.BooleanOptionalAction, default=True,
        help="batched tensor indexing instead of sbi's per-sample DataLoader")
    # hardware, budget, outputs
    add("--device", default=SP.default_device())
    add("--data-device", default=None, help="where training data lives (default: --device)")
    add("--time-budget-h", type=float, default=None,
        help="wall-time budget in hours (default: Slurm job end minus 20 min, else none; "
             "<=0: none)")
    add("--epochs-per-chunk", type=int, default=10)
    add("--out", type=Path, default=None)
    add("--logdir", type=Path, default=None, help="TensorBoard dir (default: <out>/tb)")
    # posterior sampling
    add("--n-post", type=int, default=100_000, help="posterior samples to draw")
    add("--sample-chunk", type=int, default=200_000)
    add("--ode-atol", type=float, default=None, help="ODE solver atol (sbi default 1e-6)")
    add("--ode-rtol", type=float, default=None, help="ODE solver rtol (sbi default 1e-5)")
    return p


def default_out() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = os.environ.get("FM_RUNS_DIR")
    if root:
        job = os.environ.get("SLURM_JOB_ID")
        return Path(root) / (f"{stamp}-{job}" if job else stamp)
    return Path("runs") / stamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    a = build_parser().parse_args(argv)
    explicit = set(vars(build_parser(suppress=True).parse_args(argv)))

    if a.resume is not None:
        cfg = json.loads((a.resume / "config.json").read_text())
        ignored = sorted(explicit - RUNTIME_KEYS - {"resume", "quick"})
        if ignored:
            log(f"--resume: ignoring {ignored}; these are fixed by {a.resume}/config.json")
        for k, v in cfg.items():
            # the time budget belongs to a job, not to the run: never inherit it
            if k in vars(a) and k not in (explicit & RUNTIME_KEYS) and k != "time_budget_h":
                setattr(a, k, v)
        for k in ("sims_cache", "logdir"):
            if getattr(a, k) is not None:
                setattr(a, k, Path(getattr(a, k)))
        a.out, a.resume = a.resume, True
        return a

    if a.quick:
        for k, v in QUICK.items():
            if k not in explicit:
                setattr(a, k, v)
    if a.label_noise and a.labels != "T":
        build_parser().error("--label-noise requires --labels T")
    if a.out is None:
        a.out = default_out()
    if a.logdir is None:
        a.logdir = a.out / "tb"
    a.resume = False
    return a


def mode_description(a: argparse.Namespace) -> str:
    if a.labels == "L":
        return "L: 11-dim deterministic label L(G) (paper construction)"
    if a.label_noise:
        return ("T+noise: 5-dim T(G) + N(0, sigma^2), sigma = chi2() sigmas "
                "(Bayesian posterior given NuFIT uncertainties)")
    return "T: 5-dim deterministic observables T(G)"


def n_workers(a: argparse.Namespace) -> int:
    if a.workers:
        return a.workers
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def to_jsonable(d: dict) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


# --------------------------------------------------------------------------- #
# Time budget
# --------------------------------------------------------------------------- #

def _slurm_seconds(s: str) -> float | None:
    """Parse Slurm's [D-]HH:MM:SS / MM:SS time format."""
    s = s.strip()
    if not s or not s[0].isdigit():
        return None                                   # UNLIMITED, INVALID, ...
    days, _, hms = s.rpartition("-")
    parts = [float(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return (float(days or 0) * 24 + parts[0]) * 3600 + parts[1] * 60 + parts[2]


def deadlines_from_args(a: argparse.Namespace) -> tuple[float | None, float | None]:
    """
    Absolute time.time() values (train, sample) by which training must have
    checkpointed and posterior sampling must stop, or None for no limit.  With a
    Slurm end time, training keeps a 20-minute margin and sampling 10 minutes.
    """
    if a.time_budget_h is not None:
        d = None if a.time_budget_h <= 0 else T_START + a.time_budget_h * 3600
        return d, d
    end = None
    if os.environ.get("SLURM_JOB_END_TIME"):
        end = float(os.environ["SLURM_JOB_END_TIME"])
    elif os.environ.get("SLURM_JOB_ID") and shutil.which("squeue"):
        with contextlib.suppress(Exception):
            left = subprocess.run(["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%L"],
                                  capture_output=True, text=True, timeout=60).stdout
            secs = _slurm_seconds(left.splitlines()[0]) if left.strip() else None
            if secs:
                end = time.time() + secs
    if end is None:
        return None, None
    return end - SAFETY_MARGIN_S, end - SAFETY_MARGIN_S / 2


# --------------------------------------------------------------------------- #
# Simulations
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def _single_threaded_children():
    """Spawned workers inherit the environment: keep their BLAS single-threaded."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update({k: "1" for k in keys})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def generate(a: argparse.Namespace, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n simulations from ceil(n / sim_chunk) independently seeded chunks."""
    n_chunks = math.ceil(n / a.sim_chunk)
    # Every chunk is drawn at full size and the result truncated, so the first n
    # rows are the same whatever the total (also with label noise).
    tasks = [(a.seed, k, a.sim_chunk, a.labels, a.label_noise) for k in range(n_chunks)]
    workers = min(n_workers(a), n_chunks)
    out, step = [], max(1, n_chunks // 10)
    if workers > 1:
        with _single_threaded_children(), mp.get_context("spawn").Pool(workers) as pool:
            for k, res in enumerate(pool.imap(_sim_task, tasks)):
                out.append(res)
                if (k + 1) % step == 0:
                    log(f"[sim] {k + 1}/{n_chunks} chunks")
    else:
        out = [_sim_task(t) for t in tasks]
    theta = np.concatenate([o[0] for o in out])[:n]
    x = np.concatenate([o[1] for o in out])[:n]
    return theta, x


def _sim_task(t: tuple) -> tuple[np.ndarray, np.ndarray]:
    return S.simulate_prior_chunk(*t)


def get_simulations(a: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Training set as float32 tensors, via the cache if one is given."""
    t0 = time.time()
    meta = dict(seed=a.seed, sim_chunk=a.sim_chunk, labels=a.labels, label_noise=a.label_noise)
    theta = x = None
    source = "generated"
    cache = a.sims_cache
    if cache is not None and cache.exists():
        with np.load(cache) as z:
            cmeta = json.loads(str(z["meta"]))
            n_cached = cmeta.pop("n")
            if cmeta != meta:
                raise SystemExit(f"sims cache {cache} was made with {cmeta}, this run needs "
                                 f"{meta}; use another --sims-cache path")
            if n_cached >= a.n_sims:
                theta = np.ascontiguousarray(z["theta"][:a.n_sims])
                x = np.ascontiguousarray(z["x"][:a.n_sims])
                source = f"loaded from {cache}"
            else:
                log(f"[sim] cache {cache} has only {n_cached} rows; regenerating")
    if theta is None:
        theta, x = generate(a, a.n_sims)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(cache.stem + ".tmp.npz")
            np.savez(tmp, theta=theta, x=x, meta=json.dumps(dict(meta, n=len(theta))))
            os.replace(tmp, cache)
            source += f", saved to {cache}"
    dt = time.time() - t0
    log(f"[sim] {len(theta):,} simulations, x dim {x.shape[1]}, {source} in {dt:.1f} s")
    return torch.from_numpy(theta), torch.from_numpy(x), dict(sim_s=dt, sim_source=source)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.resume:
        cfg = to_jsonable({k: v for k, v in vars(a).items() if k != "resume"})
        cfg.update(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                   slurm_job_id=os.environ.get("SLURM_JOB_ID"), host=platform.node(),
                   python=platform.python_version(), torch=torch.__version__,
                   mode=mode_description(a))
        (a.out / "config.json").write_text(json.dumps(cfg, indent=2))
    a.logdir = Path(a.logdir) if a.logdir is not None else a.out / "tb"
    workers = n_workers(a)
    torch.set_num_threads(workers)
    deadline, sample_deadline = deadlines_from_args(a)

    log("=" * 78)
    log(f"FMPE seesaw  {'RESUME ' if a.resume else ''}run dir: {a.out}")
    log(f"label mode   {mode_description(a)}")
    log(f"device={a.device}  data_device={a.data_device or a.device}  workers={workers}  "
        f"n_sims={a.n_sims:,}  n_post={a.n_post:,}  batch={a.batch}  lr={a.lr}  "
        f"hidden={a.hidden}  layers={a.layers}  seed={a.seed}")
    log("time budget  " + ("none" if deadline is None else
                           f"until {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deadline))}"
                           f" ({(deadline - time.time()) / 3600:.2f} h from now)"))
    if a.device.startswith("cuda"):
        log(f"cuda         {torch.cuda.get_device_name()}  torch {torch.__version__} "
            f"(CUDA {torch.version.cuda})")
    log("=" * 78)

    torch.manual_seed(a.seed)
    theta, x, sim_info = get_simulations(a)

    # ---- inference object: new or from checkpoint ------------------------- #
    ckpt_path = a.out / "checkpoint.pkl"
    if a.resume:
        state = load_checkpoint(ckpt_path)
        inf: ChunkedFMPE = state["inference"]
        inf.set_tracker(a.logdir)
        inf.restore_data(theta, x, data_device=a.data_device)
        train_s, n_segments = state["train_s"], state["n_segments"] + 1
        finished = state["finished"]
        log(f"[resume] {ckpt_path}: {inf.epochs_trained} epochs done, best val loss "
            f"{inf.best_val_loss:.4f}, segment {n_segments}")
    else:
        vf_builder = posterior_flow_nn(
            model="transformer",
            hidden_features=a.hidden, num_layers=a.layers,
            z_score_theta="none",        # G already lives in a bounded box
            z_score_x="independent",     # Eq. 3.17: standardise each label
        )
        inf = ChunkedFMPE(prior=SP.box_prior(a.device), vf_estimator=vf_builder,
                          device=a.device, show_progress_bars=False,
                          tracker=TensorBoardTracker(SummaryWriter(str(a.logdir))),
                          fast_loader=a.fast_loader)
        inf.append_simulations(theta, x, data_device=a.data_device)
        train_s, n_segments, finished = 0.0, 1, False
    n_valid = inf._theta_roundwise[0].size(0)
    del theta, x

    # ---- train in chunks, checkpointing against the time budget ----------- #
    train_kwargs = dict(training_batch_size=a.batch, learning_rate=a.lr,
                        validation_fraction=a.val_frac, stop_after_epochs=a.stop_after,
                        max_num_epochs=a.max_epochs,
                        clip_max_norm=a.clip_grad if a.clip_grad > 0 else None,
                        show_train_summary=False)
    while not finished:
        t0, e0 = time.time(), inf.epochs_trained
        finished = inf.train_chunk(a.epochs_per_chunk, **train_kwargs)
        dt = time.time() - t0
        train_s += dt
        n_ep = max(inf.epochs_trained - e0, 1)
        inf.save_checkpoint(ckpt_path, train_s=train_s, n_segments=n_segments,
                            finished=finished)
        torch.save(inf._neural_net.state_dict(), a.out / "estimator_state.pt")
        log(f"[train] epoch {inf.epochs_trained:4d}  val loss {inf._val_loss:.4f}  "
            f"best {inf.best_val_loss:.4f}  {dt / n_ep:.1f} s/epoch  "
            f"total train {train_s / 3600:.2f} h  checkpointed")
        if not finished and deadline is not None and time.time() + 1.2 * dt > deadline:
            log(f"[train] time budget exhausted after {inf.epochs_trained} epochs without "
                f"convergence; resume with: python trainer.py --resume {a.out}")
            return EXIT_BUDGET
    why = "early stopping" if inf.sbi_converged else f"max_epochs={a.max_epochs}"
    log(f"[train] finished after {inf.epochs_trained} epochs ({why}), best val loss "
        f"{inf.best_val_loss:.4f}, {train_s / 60:.1f} min of training")

    # ---- save estimator / posterior on the CPU ---------------------------- #
    est_cpu = copy.deepcopy(inf._neural_net).to("cpu").eval()
    torch.save(est_cpu.state_dict(), a.out / "estimator_state.pt")
    torch.save(est_cpu, a.out / "estimator.pt")
    torch.save(SP.build_posterior(copy.deepcopy(est_cpu), "cpu", a.ode_atol, a.ode_rtol),
               a.out / "posterior.pt")

    # ---- posterior conditioned on the NuFIT best fit ---------------------- #
    posterior = SP.build_posterior(copy.deepcopy(est_cpu), a.device, a.ode_atol, a.ode_rtol)
    res = SP.draw(posterior, SP.conditioning_point(a.labels), a.n_post, a.sample_chunk,
                  seed=a.seed, deadline=sample_deadline)
    SP.save_tag(a.out, "train", res, dict(seed=a.seed, device=a.device,
                                          ode_atol=a.ode_atol, ode_rtol=a.ode_rtol))
    st = SP.stats(res["chi2"], res["T"], res["n_drawn"])
    log("[sample] " + SP.format_stats(st))
    MS.merge(a.out, log=lambda s: None)

    summary = dict(
        mode=mode_description(a), labels=a.labels, label_noise=a.label_noise,
        n_sims=a.n_sims, n_sims_valid=n_valid, epochs_trained=inf.epochs_trained,
        converged=bool(inf.sbi_converged), best_val_loss=inf.best_val_loss,
        n_segments=n_segments, n_post=a.n_post, sampling_complete=res["n_drawn"] >= a.n_post,
        wall_s=dict(sim=sim_info["sim_s"], train_total=train_s, sample=res["seconds"],
                    this_job=time.time() - T_START),
        **st,
    )
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"saved to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
