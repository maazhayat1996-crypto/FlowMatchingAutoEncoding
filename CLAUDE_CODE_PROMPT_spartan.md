# Task: prepare this FMPE pipeline for a "Tier 1" paper-replica run on Spartan (UniMelb Slurm GPU cluster)

## Goal
Reproduce arXiv:2608.15042 at the paper's scale on one A100, with a pipeline that is checkpointable, measurable, and reusable for larger runs later. Paper benchmarks to match: 1e7 training simulations; transformer vector field with 100-dim tokens; batch 256, 50k steps, Adam, max lr 5e-4, ~40 min training on an RTX 4070 Laptop GPU; 3e6 posterior draws (11 h on that laptop); result: 956,607 samples (31.9%) with chi2 < 2500 and 640 (0.021%) with chi2 < 45.
Our previous run (1e6 sims, 1e5 draws, batch 256, Mac): 19.8% below 2500, 3 below 45. Tier 1 target: >= ~600 samples with chi2 < 45 from 3e6 draws, plus measured ms/step and sampling samples/s so the next (larger) tier can be sized from data.

## Codebase
- `simulate.py`: Type-I seesaw forward model G(23) -> L(11), numpy float64, vectorised, with `chi2()`, `derived_quantities()`, `labels_from_nufit_bestfit()`. Do NOT change its physics or public signatures; you may add chunked wrappers.
- `trainer.py`: FMPE via sbi 0.27 (`posterior_flow_nn(model="transformer", hidden_features=100, num_layers=5, z_score_theta="none", z_score_x="independent")`), samples the posterior at the NuFIT best fit, computes chi2, saves `G_samples.npy`/`chi2.npy`.
- `plot_posterior.py`: plots from a run dir's `G_samples.npy`/`chi2.npy`; must keep working.
- Local dev machine: macOS, venv at `./venv` is Python 3.14 (torch 2.14, sbi 0.27.0, zuko 1.6.0, numpy 2.5.3). Spartan has only Python 3.10.4 / 3.11.3, so all code must be 3.10-compatible (no 3.12+ syntax).
- BEFORE writing sbi-related code, read the installed sbi source in `venv/lib/python3.14/site-packages/sbi/` for: `FMPE.__init__` (`summary_writer`, `show_progress_bars`), `append_simulations` (does it take `data_device`?), `train` (`resume_training`, `max_num_epochs` semantics on resume, `stop_after_epochs`, convergence bookkeeping, `summary` dict), `build_posterior`, the vector-field posterior's `sample` signature (batching kwargs, ODE solver options/tolerances), and the transformer builder's kwargs. Do not guess the API. sbi has no LR scheduler; do not bolt one on - rely on early stopping.

## Spartan facts (from the official docs) to design around
- Slurm. `gpu-a100`: 4x A100 80GB per node, 32 cores, ~480 GB RAM, max 7 days. `gpu-a100-short`: 1 GPU, max 4 h (testing). Request with `#SBATCH -p gpu-a100` + `#SBATCH --gres=gpu:1`; no QoS line. Per-user: up to 48 GPUs on gpu-a100, 200 queued jobs. Priority is fairshare-dominated (14-day lookback) and backfill rewards accurate `--time`/`--mem`.
- Defaults are 10 min walltime and 4000 MB per CPU: always set `--time`, `--cpus-per-task`, `--mem`.
- Storage: `/home/$USER` (50 GB, slow) for code only; data and runs under `/data/gpfs/projects/<projectID>/` (backed up). Node-local NVMe `/tmp` available via `#SBATCH --tmp=XGB`.
- Python venv is built ONCE on a login node: `module load foss/2022a Python/3.10.4` (prefer `Python/3.11.3` if `module spider` shows it loads), `virtualenv ~/venvs/fmpe`, `source ~/venvs/fmpe/bin/activate`, `pip install -r requirements.txt`. Batch jobs then run `module purge; module load foss/2022a Python/3.10.4; source ~/venvs/fmpe/bin/activate`.
- pip torch wheels bundle the CUDA runtime, but the node driver must support the wheel's CUDA version (docs say CUDA 12.4-capable; verify with `nvidia-smi` in a GPU job). Install torch from the matching `https://download.pytorch.org/whl/cu12x` index; leave the exact index as a clearly marked placeholder in `setup_env.sh`.
- Never run computation on login nodes. Tests go to `gpu-a100-short` via sbatch or `sinteractive -p gpu-a100-short --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=0:30:00`.
- Slurm output is buffered: use `python -u`. No ssh out from Spartan; `git clone` over HTTPS works. Remote: https://github.com/maazhayat1996-crypto/FlowMatchingAutoEncoding.git . Workflow: edit locally -> push -> `git pull` on Spartan -> `sbatch`.

## Deliverables (Tier 1 only - do not build Tier 2/3 scripts or any alternative label/noise modes)

### 1. `requirements.txt`
Minimal, pinned to versions with cp310/cp311 manylinux wheels: sbi==0.27.0, torch (comment with the cu12x index URL), numpy, scipy, matplotlib, zuko, tensorboard.

### 2. `trainer.py` (keep existing flags working: `--quick --n-sims --n-post --batch --lr --device --seed --out --stop-after --max-epochs`)
- Add flags: `--hidden` (100), `--layers` (5), `--val-frac` (0.1), `--epochs-per-chunk` (10), `--time-budget-h`, `--resume RUN_DIR`, `--sims-cache PATH.npz`, `--sample-chunk` (200000), `--logdir`, `--tf32` (default on for cuda; sets `torch.backends.cuda.matmul.allow_tf32=True` for training only). Any other builder kwarg sbi exposes (e.g. heads) becomes a flag too.
- Run dir default: `$FM_RUNS_DIR/<timestamp>-<SLURM_JOB_ID>` if `FM_RUNS_DIR` is set, else `runs/<timestamp>`. Write `config.json` at start and `summary.json` at end containing: all args, n_sims, epochs trained, best validation loss, ms/step (measure around the train loop, from epoch time and steps/epoch), wall-clock for simulate/train/sample, sampling samples/s, n_post, fraction inside the prior box, chi2 min/median, counts below 45/100/2500, medians of the five observables, git commit hash, hostname, GPU name.
- Tensorboard writer directed to `<out>/tb` via FMPE's `summary_writer`; `show_progress_bars=False`; every print `flush=True`; a banner line at start with device, GPU name, torch/sbi versions.
- Simulation cache: if `--sims-cache` exists with >= n_sims rows and matching seed (store seed in the npz), load the first n_sims; else simulate and save float32 theta/x. Simulate in chunks with a `multiprocessing.Pool` sized by `SLURM_CPUS_PER_TASK` (fallback `os.cpu_count()`), each chunk seeded deterministically from `--seed` and the chunk index.
- Keep the training set on the GPU if `append_simulations` accepts `data_device` (verify in source); otherwise leave on CPU and say so in the README.
- Checkpoint/resume against a time budget: default budget from Slurm (`scontrol show job $SLURM_JOB_ID` end time, or `$SLURM_JOB_END_TIME` if present) minus a 20 min margin; unlimited if not under Slurm. Train in chunks of `--epochs-per-chunk` using `train(..., resume_training=True)` with cumulative `max_num_epochs`. After each chunk save `<out>/checkpoint.pkl` (the inference object; detach the SummaryWriter before pickling and re-attach after) plus `<out>/estimator_state.pt`. Stop when sbi's early-stopping criterion fires (detect it from the returned epoch count / sbi's convergence attributes, not by guessing). If the budget runs out first: save and `sys.exit(3)`. `--resume RUN_DIR` reloads `checkpoint.pkl` + `config.json` and continues in the same run dir.
- After training: save `estimator_state.pt` and `posterior.pt` with everything moved to CPU first; all loaders use `torch.load(..., map_location="cpu", weights_only=False)` then `.to(device)`.
- Posterior sampling and chi2 go through the shared functions in `sample_posterior.py` (below), chunked by `--sample-chunk`, accumulated on CPU; save `G_samples.npy`, `chi2.npy`, `G_chi2_lt_45.npy`, `G_chi2_lt_2500.npy`.

### 3. `sample_posterior.py` and `merge_samples.py` (new)
- `python sample_posterior.py RUN_DIR --n N --seed S --chunk 200000 --device cuda --tag T [--benchmark]`: loads the trained posterior from RUN_DIR, draws N samples at `labels_from_nufit_bestfit()`, filters to the prior box, computes chi2 in chunks, writes `RUN_DIR/samples/<tag>_G.npy` and `<tag>_chi2.npy`, prints samples/s and counts below 45/100/2500 (also appended as a JSON line to `RUN_DIR/samples/log.jsonl`). `--benchmark` times chunk sizes 50k/200k/500k on 1 chunk each and prints samples/s for each, then exits. Expose any ODE solver tolerance/steps options sbi provides as flags with sbi's defaults.
- `python merge_samples.py RUN_DIR`: concatenates all `samples/*_G.npy` / `*_chi2.npy` (plus the training job's own draws if present) into `RUN_DIR/G_samples.npy` and `chi2.npy` so `plot_posterior.py` works unchanged; prints totals and fractions and writes `RUN_DIR/samples_summary.json`.

### 4. `spartan/` scripts
Common to all: `#!/bin/bash`, explicit `--job-name --time --cpus-per-task --mem`, `-o`/`-e` into `$FM_RUNS_DIR/logs/%x-%j.out|err`, `module purge; module load ...; source ~/venvs/fmpe/bin/activate`, `export FM_RUNS_DIR=/data/gpfs/projects/<projectID>/fmpe/runs` (placeholder at the top of each file), `cd` to the repo checkout under the project dir (placeholder), `python -u`.
- `setup_env.sh`: one-time login-node venv build (module loads, virtualenv, torch install from the cu12x index placeholder, `pip install -r requirements.txt`, `pip check`, prints versions). No computation.
- `smoke.slurm`: `gpu-a100-short`, 1 GPU, 4 CPUs, 32G, 30 min. Prints `nvidia-smi` and `python -c "import torch;print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name())"`, runs `trainer.py --quick`, then `sample_posterior.py <run> --benchmark`, then `sample_posterior.py <run> --n 20000 --tag smoke`, then `merge_samples.py` and `plot_posterior.py`.
- `train_tier1.slurm`: `gpu-a100`, 1 GPU, 8 CPUs, 64G, `--time=8:00:00`. Runs `trainer.py --n-sims 10000000 --n-post 200000 --batch 1024 --lr 1e-3 --hidden 100 --layers 5 --stop-after 20 --sims-cache $FM_RUNS_DIR/sims_1e7_seed0.npz` (the 2e5 in-job draws are a sanity check; the bulk draws come from the array below). If python exits 3, resubmit itself with `sbatch --dependency=afterany:$SLURM_JOB_ID spartan/train_tier1.slurm` passing the run dir (env var `FM_RESUME`), guarded by a resubmit counter env var (max 5).
- `sample_tier1.slurm`: array `--array=1-10`, `gpu-a100-short`, 1 GPU, 4 CPUs, 32G, `--time=4:00:00`; each task `sample_posterior.py $FM_RUN --n 300000 --seed $SLURM_ARRAY_TASK_ID --tag $SLURM_ARRAY_TASK_ID` (10 x 3e5 = 3e6, the paper's count). Put a comment: resize `--n`/`--time` from the samples/s printed by smoke.slurm (paper laptop was ~76 samples/s; if the A100 is < 25 samples/s per task something is wrong).
- `merge_tier1.slurm`: `sapphire`, 1 CPU, 16G, 20 min; runs `merge_samples.py $FM_RUN` then `plot_posterior.py $FM_RUN`; README says to submit with `--dependency=afterok:<array job id>`.
- `train.slurm`: generic version that forwards `"$@"` to `trainer.py` (same resubmit logic), so larger tiers need no new script.

### 5. README section "Running on Spartan (Tier 1)"
Ordered steps with exact commands: clone into `/data/gpfs/projects/<projectID>/`, `bash spartan/setup_env.sh`, `sbatch spartan/smoke.slurm`, what to check in its log (CUDA true, ms/step, samples/s), `sbatch spartan/train_tier1.slurm`, `FM_RUN=<run dir> sbatch spartan/sample_tier1.slurm`, `FM_RUN=<run dir> sbatch --dependency=afterok:<id> spartan/merge_tier1.slurm`, `squeue --me` / `sacct` to monitor, and an `rsync` command to pull `summary.json`, `samples_summary.json` and `figures/` back to the Mac. State the paper's benchmark numbers as the pass criterion. Add a short note on resume behaviour and on the two placeholders (`<projectID>`, torch index URL).

## Verification before you finish (all local, CPU)
- `python simulate.py` self-test passes unchanged.
- `python trainer.py --quick --device cpu` runs end to end; `plot_posterior.py` works on its output; `summary.json` has every field listed above.
- Resume path: `python trainer.py --quick --device cpu --epochs-per-chunk 1 --time-budget-h 0.0005` exits 3 with a checkpoint; `python trainer.py --resume <run>` finishes it.
- `python sample_posterior.py <run> --n 5000 --device cpu --tag t1 --benchmark` and `--tag t1` (non-benchmark) work; `python merge_samples.py <run>` works and `plot_posterior.py` runs on the merged result.
- `bash -n spartan/*.slurm spartan/setup_env.sh` passes. Grep for Python 3.12+ syntax.
- `.gitignore` covers `samples/`, `*.npz`, `tb/`, `logs/`, `checkpoint.pkl`. Commit with a clear message; do not push.
Finish with: files changed, placeholders I must fill, which sbi features you verified exist (data_device, resume_training, solver options) and any you could not use.
