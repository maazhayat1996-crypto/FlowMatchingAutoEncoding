#!/bin/bash
# Generic training job: all arguments go to trainer.py (do not pass --out; the
# run dir is $FM_RUNS_DIR/<timestamp>-<jobid>).  Adjust --time/--mem at
# submission for larger tiers, e.g.
#   sbatch --time=1-00:00:00 spartan/train.slurm --n-sims 30000000 --n-post 200000 \
#       --batch 1024 --lr 1e-3 --sims-cache /data/gpfs/projects/<projectID>/fmpe/runs/sims_3e7_seed0.npz
#   FM_RESUME=<run dir> sbatch spartan/train.slurm        # continue by hand
# Exit status 3 (time budget used up) resubmits this script with the same args.
#SBATCH --job-name=fmpe-train
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
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

source spartan/train_common.sh
FM_SCRIPT_ARGS=("$@")
run_trainer spartan/train.slurm "$@"
exit $?
