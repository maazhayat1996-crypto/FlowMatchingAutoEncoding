#!/bin/bash
# Tier 3: 1e8 simulations (10x Tier 1), same network and optimiser as Tier 1,
# epochs capped at 40 (Tier 1 plateaued after ~40 of its 146), plus 2e5
# posterior draws as a sanity check (bulk draws: sample_tier3.slurm).
#   sbatch spartan/train_tier3.slurm                      # new run
#   FM_RESUME=<run dir> sbatch spartan/train_tier3.slurm  # continue by hand
# Exit status 3 (time budget used up) resubmits this script automatically.
#
# Sizing (from Tier 1: 229 s/epoch for 1e7 incl. validation): ~38 min/epoch,
# <= 41 epochs ~ 26 h.  The first job also simulates 1e8 points (~4 min, 8
# workers) and writes a ~14 GB cache, sims_1e8_seed0.npz; its first 1e7 rows
# are identical to the Tier 1 training set.  Host RAM peaks around 40 GB
# (simulation, and the CPU copy sbi makes to build the network); the GPU holds
# ~14 GB of training data plus transient copies of similar size.
#SBATCH --job-name=fmpe-train-t3
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
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
run_trainer spartan/train_tier3.slurm \
    --n-sims 100000000 --n-post 200000 --batch 1024 --lr 1e-3 \
    --hidden 100 --layers 5 --stop-after 10 --max-epochs 40 --epochs-per-chunk 2 \
    --sims-cache "$FM_RUNS_DIR/sims_1e8_seed0.npz"
exit $?
