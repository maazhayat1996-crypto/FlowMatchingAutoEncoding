# Flow-matching posterior estimation for the Type-I seesaw

Reproduces the FMPE pipeline of arXiv:2608.15042 (Kitagawa, Nishimura, Otsuka).

| file | role |
|---|---|
| `simulate.py` | forward model G (23) -> L (11), observables T, chi^2 (Eq. 3.15); `python simulate.py` runs the self-test |
| `trainer.py` | simulate, train FMPE (sbi 0.27, transformer vector field), sample the posterior at the NuFIT 6.0 best fit |
| `fmpe_ext.py` | sbi extension: training in chunks, checkpoint/resume, fast batch loader |
| `sample_posterior.py` | extra posterior draws from a trained run (one Slurm array task per seed) |
| `merge_samples.py` | combine `RUN/samples/*` into `RUN/G_samples.npy` / `chi2.npy` |
| `plot_posterior.py` | figures from `G_samples.npy` / `chi2.npy` into `RUN/figures/` |
| `spartan/` | Slurm scripts and the one-time environment setup |

Locally: `python trainer.py --quick --device cpu`, then `python plot_posterior.py runs/<timestamp>`.

## Running on Spartan

Spartan is the University of Melbourne's Slurm cluster. Code lives in your project directory, environments in `~/venvs`, and results in `/data/gpfs/projects/<projectID>/fmpe/runs`. The workflow is: edit locally, `git push`, `git pull` on Spartan, `sbatch`.

### 0. Fill in the placeholders (locally, once) and push

`<projectID>` appears in `spartan/*.slurm` (in the `#SBATCH -o/-e` lines and in `FM_PROJECT`) and in `spartan/setup_env.sh`. The scripts refuse to run until it is filled in:

```bash
sed -i '' 's/<projectID>/punim1234/g' spartan/*.slurm spartan/setup_env.sh   # macOS sed
git commit -am "Set Spartan project ID" && git push
```

### 1. Clone and build the environment (login node)

```bash
ssh <username>@spartan.hpc.unimelb.edu.au
cd /data/gpfs/projects/<projectID>
git clone https://github.com/maazhayat1996-crypto/FlowMatchingAutoEncoding.git
cd FlowMatchingAutoEncoding
bash spartan/setup_env.sh          # venv in ~/venvs/fmpe, also creates fmpe/runs and fmpe/logs
```

`setup_env.sh` installs torch 2.14.1 from the `cu126` wheel index before `requirements.txt`. If the smoke test shows that CUDA is not usable, see the comment at the top of the script. To use Python 3.11 instead of 3.10, check `module spider Python/3.11.3` and change the `module load` line in `setup_env.sh` and in every `spartan/*.slurm`.

### 2. Smoke test (about 10 min on `gpu-a100-short`)

```bash
sbatch spartan/smoke.slurm
squeue --me
less /data/gpfs/projects/<projectID>/fmpe/logs/fmpe-smoke-<jobid>.out
```

Check the following in the log:
- `nvidia-smi` prints a driver whose CUDA version is at least the wheel's (12.6).
- `True 12.6` appears, i.e. `torch.cuda.is_available()` and `torch.version.cuda`.
- The quick training run finishes.
- The 200k-sample chunk reports its `peak GPU memory`. If it is close to 80 GiB, lower `--chunk` in `sample.slurm` and `--sample-chunk` for `trainer.py`.

### 3. Train at paper scale (1e7 simulations, 3e6 posterior draws)

```bash
sbatch spartan/train.slurm
```

- The run directory is `fmpe/runs/<timestamp>-<jobid>`. The log is `<run>/logs/train-<jobid>.log`, and Slurm's own `-o/-e` files go to `fmpe/logs/`.
- The 1e7 simulations are cached in `fmpe/runs/sims_1e7_seed0.npz` (about 1.4 GB). Later runs with the same seed and label mode load them instead of simulating again.
- Training checkpoints every 10 epochs (`checkpoint.pkl`, `estimator_state.pt`). If sbi's early stopping has not fired 20 minutes before the 36 h wall time, the job saves, exits with status 3 and resubmits itself as `spartan/train.slurm --resume <run>` (`afterany` dependency, at most 5 times; set `FM_MAX_RESUBMIT` to change this). A resumed run continues exactly where it stopped: same weights, Adam state, train/validation split, early-stopping counters and RNG state. To resume by hand: `sbatch spartan/train.slurm --resume <run>`.
- Monitor with `squeue --me`, `sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State`, and `tail -f <run>/logs/train-*.log`. TensorBoard logs are in `<run>/tb`.
- The run is finished when `<run>/summary.json` exists. It holds epochs, losses, wall times and the chi^2 counts below 45/100/2500.
- Extra `trainer.py` flags can be appended, e.g. `sbatch spartan/train.slurm --batch 4096 --lr 1e-3`. `python trainer.py --help` lists them all.

### 4. More posterior samples (10 x 2e6 draws), then merge and plot

```bash
sbatch spartan/sample.slurm <run>                                # prints the array job id
sbatch --dependency=afterok:<array job id> spartan/merge.slurm <run>
```

Each array task writes `<run>/samples/<task>_G.npy` and `_chi2.npy`, using seeds 1..10. The trainer's own draws are in `samples/train_*` and use the training seed (0). `merge.slurm` concatenates every set into `G_samples.npy`, `chi2.npy`, `G_chi2_lt_45.npy` and `G_chi2_lt_2500.npy`, and then runs `plot_posterior.py`.

### 5. Pull the results to the Mac

Spartan cannot ssh out, so pull from the Mac. This copies the figures, summaries, the chi^2 < 45 samples and the logs, but not the large sample files or checkpoints:

```bash
RUN=<timestamp>-<jobid>
rsync -avzm --include='*/' --include='figures/***' --include='logs/***' \
    --include='summary.json' --include='config.json' --include='merged.json' \
    --include='G_chi2_lt_45.npy' --include='estimator.pt' --exclude='*' \
    <username>@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/<projectID>/fmpe/runs/$RUN/ runs/$RUN/
```

`.pt` files open on a Mac with `torch.load(path, map_location="cpu", weights_only=False)`. To sample locally, run `python sample_posterior.py runs/$RUN --n 10000 --device cpu`.

### Label modes

These two modes define different posteriors. The mode is recorded in `config.json` and `summary.json`, and is printed in the banner.

- `--labels L` (default) is the paper's construction: the network is conditioned on the 11-dim deterministic label L(G) (two log mass splittings, nine |U_PMNS| entries) and evaluated at L built from the NuFIT best fit.
- `--labels T --label-noise` conditions on the five fitted observables T(G) plus Gaussian noise whose sigmas are the ones in `chi2()`, and evaluates at the NuFIT best-fit T. With noisy labels the network learns the Bayesian posterior p(G | T_obs), whose likelihood is proportional to exp(-chi^2/2). Use a separate simulation cache:
  ```bash
  SIMS_CACHE=/data/gpfs/projects/<projectID>/fmpe/runs/sims_1e7_seed0_Tnoise.npz \
      sbatch spartan/train.slurm --labels T --label-noise
  ```
