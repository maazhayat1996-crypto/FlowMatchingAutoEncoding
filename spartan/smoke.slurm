#!/bin/bash
# GPU smoke test: CUDA visible to torch, quick training, one full-size sampling chunk.
#   sbatch spartan/smoke.slurm
#SBATCH --job-name=fmpe-smoke
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH -o /data/gpfs/projects/<projectID>/fmpe/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/<projectID>/fmpe/logs/%x-%j.err

FM_PROJECT="/data/gpfs/projects/<projectID>"
case $FM_PROJECT in *"<projectID>"*) echo "fill in <projectID> in spartan/*.slurm first" >&2; exit 1;; esac
export FM_RUNS_DIR=$FM_PROJECT/fmpe/runs
cd "$FM_PROJECT/FlowMatchingAutoEncoding" || exit 1

module purge
module load foss/2022a Python/3.10.4
source ~/venvs/fmpe/bin/activate

RUN=$FM_RUNS_DIR/smoke-$SLURM_JOB_ID
mkdir -p "$RUN/logs"
exec > >(tee -a "$RUN/logs/smoke-$SLURM_JOB_ID.log") 2>&1

nvidia-smi
python -c "import torch;print(torch.cuda.is_available(), torch.version.cuda)"

python -u trainer.py --quick --device cuda --out "$RUN" || exit 1
# one chunk of the production size, to check GPU memory and sampling speed
python -u sample_posterior.py "$RUN" --n 200000 --chunk 200000 --seed 1 --tag smoke --device cuda || exit 1
python -u merge_samples.py "$RUN"
python -u plot_posterior.py "$RUN"
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State
