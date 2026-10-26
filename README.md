# Flow-matching posterior estimation for the Type-I seesaw

Reproduces the FMPE pipeline of arXiv:2608.15042 (Kitagawa, Nishimura, Otsuka).

| file | role |
|---|---|
| `simulate.py` | forward model G (23) -> L (11), observables T, chi^2 (Eq. 3.15); `python simulate.py` runs the self-test |
| `trainer.py` | simulate, train FMPE (sbi 0.27, transformer vector field), sample the posterior at the NuFIT 6.0 best fit |
| `fmpe_ext.py` | sbi extension: training in chunks, checkpoint/resume, fast batch loader, ms/step timing |
| `sample_posterior.py` | posterior draws from a trained run (one Slurm array task per seed); `--benchmark` |
| `merge_samples.py` | combine `RUN/samples/*` into `RUN/G_samples.npy` / `chi2.npy` / `samples_summary.json` |
| `plot_posterior.py` | figures from `G_samples.npy` / `chi2.npy` into `RUN/figures/` |
| `intrinsic_dim.py` | intrinsic dimension of the χ² < cut samples in observable space, with calibrating nulls (`RUN/intrinsic_dim_chi2lt<cut>.json`, `RUN/figures/`) |
| `spartan/` | Slurm scripts and the one-time environment setup |

Locally: `python trainer.py --quick --device cpu`, then `python plot_posterior.py runs/<timestamp>`. Every hyperparameter is a flag; see `python trainer.py --help`.

## Running on Spartan (Tier 1)

Tier 1 is a replica of the paper's run on one A100. It uses 1e7 training simulations, the sbi transformer vector field (100-dim tokens, 5 layers), and 3e6 posterior draws at the NuFIT 6.0 best fit.

**Pass criterion (the paper's numbers):** of 3e6 draws, 956,607 (31.9%) have chi^2 < 2500 and 640 (0.021%) have chi^2 < 45. Tier 1 succeeds with about 600 or more samples below 45. The run should also record ms/step and sampling samples/s, so that the next tier can be sized from measurements. For reference, our earlier Mac run (1e6 sims, 1e5 draws) reached 19.8% below 2500 and 3 samples below 45.

### Two placeholders to fill (locally, once)

1. **`<projectID>`**, your Spartan project (e.g. `punim1234`). It appears in every `spartan/*.slurm` file (in the `#SBATCH -o/-e` lines and in `FM_PROJECT`) and in `spartan/setup_env.sh`. The scripts refuse to run until it is filled in:
   ```bash
   sed -i '' 's/<projectID>/punim1234/g' spartan/*.slurm spartan/setup_env.sh   # macOS sed
   git commit -am "Set Spartan project ID" && git push
   ```
   The scripts assume the repository is checked out at `/data/gpfs/projects/<projectID>/FlowMatchingAutoEncoding` (`FM_REPO`).
2. **The torch wheel index** (`TORCH_INDEX` in `spartan/setup_env.sh`, default `cu126`). It must be a CUDA version the GPU driver supports. The smoke test prints `nvidia-smi`; if it then reports `False` for CUDA, follow the fallback comment at the top of `setup_env.sh`.

### Steps

**1. Clone and build the environment** (login node; no computation):
```bash
ssh <username>@spartan.hpc.unimelb.edu.au
cd /data/gpfs/projects/<projectID>
git clone https://github.com/maazhayat1996-crypto/FlowMatchingAutoEncoding.git
cd FlowMatchingAutoEncoding
bash spartan/setup_env.sh        # ~/venvs/fmpe; also creates fmpe/runs/logs
```
To use Python 3.11 instead, check `module spider Python/3.11.3` and change the `module load` line in `setup_env.sh` and in every `spartan/*.slurm`.

**2. Smoke test** (`gpu-a100-short`, at most 30 min):
```bash
sbatch spartan/smoke.slurm
less /data/gpfs/projects/<projectID>/fmpe/runs/logs/fmpe-smoke-<jobid>.out
```
Check the following in the log:
- `True 12.x NVIDIA A100...` from the torch check, and a driver CUDA version in `nvidia-smi` at least as new as the wheel's.
- A `[train] ... ms/step` line. This is the quick run's training speed.
- The `[benchmark] chunk ...: N samples/s` lines, with peak GPU memory, for 50k/200k/500k chunks. Sizes that would not fit in 10 minutes are skipped, and an out-of-memory size is reported. The quick network is barely trained, so these rates are only indicative.
- The `--n 20000 --tag smoke` draw finishes, and `figures/` is written.

**3. Train** (`gpu-a100`, 8 h; batch 1024, lr 1e-3, early stopping with patience 20, plus 2e5 in-job draws as a sanity check):
```bash
sbatch spartan/train_tier1.slurm
squeue --me
tail -f /data/gpfs/projects/<projectID>/fmpe/runs/<timestamp>-<jobid>/logs/train-<jobid>.log
```
- The first job simulates 1e7 points and caches them in `fmpe/runs/sims_1e7_seed0.npz` (about 1.4 GB). Later runs with seed 0 load the cache.
- Each `[train]` line reports the validation loss, s/epoch and ms/step for the last chunk of 10 epochs.
- The run is done when `<run>/summary.json` exists. It holds the epochs trained, best validation loss, `ms_per_step` (training pass only) and `ms_per_step_incl_validation`, the wall times, `sampling_samples_per_s` of the trained network, the chi^2 counts, the git commit, host and GPU.

**4. Bulk draws** (10 array tasks x 3e5 = 3e6). Size them first: each task needs `300000 / sampling_samples_per_s` seconds (from `summary.json`) plus a few minutes. Edit `--n`, `--array` or `--time` in `spartan/sample_tier1.slurm` if needed. `gpu-a100-short` allows at most 4 h, and a task close to its limit stops 5 min early and keeps what it has.
```bash
FM_RUN=/data/gpfs/projects/<projectID>/fmpe/runs/<run> sbatch spartan/sample_tier1.slurm   # prints <array id>
FM_RUN=/data/gpfs/projects/<projectID>/fmpe/runs/<run> sbatch --dependency=afterok:<array id> spartan/merge_tier1.slurm
```
The merge combines the array draws (`samples/1..10_*`) with the training job's own draws (`samples/train_*`). It writes `G_samples.npy`, `chi2.npy`, `G_chi2_lt_45.npy`, `G_chi2_lt_2500.npy` and `samples_summary.json`, prints the totals and fractions, and draws the figures. Every sampling call also appends a line to `<run>/samples/log.jsonl`.

**5. Monitor:**
```bash
squeue --me
sacct -j <jobid> --format=JobID,JobName,Elapsed,MaxRSS,State
```

**6. Pull the results to the Mac.** Spartan cannot ssh out, so run this on the Mac:
```bash
RUN=<timestamp>-<jobid>
rsync -avzm --include='*/' --include='figures/***' --include='logs/***' \
    --include='summary.json' --include='samples_summary.json' --include='config.json' \
    --include='log.jsonl' --include='G_chi2_lt_45.npy' --exclude='*' \
    <username>@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/<projectID>/fmpe/runs/$RUN/ runs/$RUN/
```
`.pt` files open on a Mac with `torch.load(path, map_location="cpu", weights_only=False)`.

### Resume behaviour
Training checkpoints after every 10 epochs (`checkpoint.pkl` and `estimator_state.pt`). The time budget is the Slurm end time minus 20 minutes, read from `$SLURM_JOB_END_TIME`, `scontrol show job` or `squeue`. If sbi's early stopping has not fired before the budget would be exceeded, `trainer.py` saves and exits with status 3. The job then resubmits itself with `FM_RESUME=<run>` and an `afterany` dependency on itself, at most 5 times (`FM_MAX_RESUBMIT`).

A resumed run continues exactly where it stopped, with the same weights, Adam state, train/validation split, early-stopping counters and RNG state. Locally it reproduces the uninterrupted loss curve bit for bit. To resume by hand, run `FM_RESUME=<run dir> sbatch spartan/train_tier1.slurm`. Set `FM_RESUME` only on that command line; an exported `FM_RESUME` would turn later submissions into resumes.

### Notes
- The training data lives on the GPU: sbi 0.27's `append_simulations` takes `data_device`, and it defaults to the training device.
- sbi's ODE sampler (zuko, adaptive) exposes only `atol`/`rtol`, available as `--ode-atol`/`--ode-rtol` (sbi defaults 1e-6 / 1e-5). There is no step count.
- TF32 matmuls are on for training on CUDA (`--no-tf32` disables them) and always off for sampling.
- Larger tiers use `spartan/train.slurm`, which passes its arguments to `trainer.py`. See the header of that script.

## Tier 3 (1e8 simulations, 3e7 draws)

Same network and optimiser as Tier 1; only the data grow, and the stopping rule is adapted to 10x longer epochs (Tier 1 plateaued after ~40 of its 146 epochs).

| | Tier 1 | Tier 3 |
|---|---|---|
| simulations | 1e7 | 1e8 (`sims_1e8_seed0.npz`, ~14 GB; its first 1e7 rows are the Tier 1 set) |
| early stopping / epoch cap | patience 20 / none | patience 10 / 40 epochs (~38 min each) |
| checkpoint interval | 10 epochs | 2 epochs |
| job | 8 h, 64G | 36 h, 128G |
| bulk draws | 10 x 3e5 | 30 x 1e6 |

```bash
sbatch spartan/train_tier3.slurm                       # ~26 h of training; resubmits itself if needed
FM_RUN=<run dir> sbatch spartan/sample_tier3.slurm     # after training; check summary.json samples/s first
FM_RUN=<run dir> sbatch --dependency=afterok:<array job id> spartan/merge_tier3.slurm
```
Disk: about 14 GB for the simulation cache, 1 GB per checkpoint, and about 5.5 GB of merged samples. `punim0011` is a shared project, so check its quota first.
