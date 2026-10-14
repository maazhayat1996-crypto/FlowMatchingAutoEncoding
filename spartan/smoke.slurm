#!/bin/bash
# GPU smoke test: CUDA visible to torch, quick training (prints ms/step),
# sampling benchmark (samples/s per chunk size), a 2e4-sample draw, merge, plots.
#   sbatch spartan/smoke.slurm
#SBATCH --job-name=fmpe-smoke
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH -o /data/gpfs/projects/<projectID>/fmpe/runs/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/<projectID>/fmpe/runs/logs/%x-%j.err

# ---- placeholders: project directory and repository checkout ---------------- #
FM_PROJECT="/data/gpfs/projects/<projectID>"
FM_REPO="$FM_PROJECT/FlowMatchingAutoEncoding"
case $FM_PROJECT in *"<projectID>"*) echo "fill in <projectID> in spartan/* first" >&2; exit 1;; esac
export FM_RUNS_DIR=$FM_PROJECT/fmpe/runs
cd "$FM_REPO" || exit 1

module purge
module load foss/2022a Python/3.10.4
source ~/venvs/fmpe/bin/activate

RUN=$FM_RUNS_DIR/smoke-$SLURM_JOB_ID
mkdir -p "$RUN/logs"
exec > >(tee -a "$RUN/logs/smoke-$SLURM_JOB_ID.log") 2>&1

nvidia-smi
python -c "import torch;print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name())"

python -u trainer.py --quick --device cuda --out "$RUN" || exit 1
# The quick network is barely trained, so its ODE may be stiffer than a trained
# one's; treat these rates as indicative and size sample_tier1.slurm from the
# real run's summary.json.
python -u sample_posterior.py "$RUN" --benchmark --device cuda --benchmark-max-s 600
python -u sample_posterior.py "$RUN" --n 20000 --tag smoke --device cuda || exit 1
python -u merge_samples.py "$RUN"
python -u plot_posterior.py "$RUN"
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State
