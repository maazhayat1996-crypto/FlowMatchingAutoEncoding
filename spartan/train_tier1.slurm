#!/bin/bash
# Tier 1 (paper replica): 1e7 simulations, transformer FMPE, early stopping,
# plus 2e5 posterior draws as a sanity check (bulk draws: sample_tier1.slurm).
#   sbatch spartan/train_tier1.slurm                      # new run
#   FM_RESUME=<run dir> sbatch spartan/train_tier1.slurm  # continue by hand
# Exit status 3 (time budget used up) resubmits this script automatically.
#SBATCH --job-name=fmpe-train-t1
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH -o /data/gpfs/projects/punim0011/mhhaya/fmpe/runs/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/punim0011/mhhaya/fmpe/runs/logs/%x-%j.err

# ---- placeholders: project directory and repository checkout ---------------- #
FM_PROJECT="/data/gpfs/projects/punim0011/mhhaya"
FM_REPO="$FM_PROJECT/FlowMatchingAutoEncoding"
case $FM_PROJECT in *"<projectID>"*) echo "fill in <projectID> in spartan/* first" >&2; exit 1;; esac
export FM_RUNS_DIR=$FM_PROJECT/fmpe/runs
cd "$FM_REPO" || exit 1

module purge
module load foss/2022a Python/3.10.4
source ~/venvs/fmpe/bin/activate

source spartan/train_common.sh
FM_SCRIPT_ARGS=()
run_trainer spartan/train_tier1.slurm \
    --n-sims 10000000 --n-post 200000 --batch 1024 --lr 1e-3 \
    --hidden 100 --layers 5 --stop-after 20 \
    --sims-cache "$FM_RUNS_DIR/sims_1e7_seed0.npz"
exit $?
