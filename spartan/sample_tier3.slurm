#!/bin/bash
# Tier 3 bulk posterior draws: 30 tasks x 1e6 = 3e7.
#   FM_RUN=<run dir> sbatch spartan/sample_tier3.slurm     # prints the array job id
# then: FM_RUN=<run dir> sbatch --dependency=afterok:<array job id> spartan/merge_tier3.slurm
#
# Sizing: at Tier 1's 462 samples/s a task takes ~36 min; check
# "sampling_samples_per_s" in <run>/summary.json for the Tier 3 network and
# adjust --time (or --array / --n) before submitting.  gpu-a100-short runs at
# most 2 tasks per user at once (~10 h for all 30); for more concurrency submit
# with -p gpu-a100 (longer queue, but far more GPUs per user).  A task near its
# time limit stops 5 min early and saves what it has.
#SBATCH --job-name=fmpe-sample-t3
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:15:00
#SBATCH --array=1-30
#SBATCH -o /data/gpfs/projects/punim0011/mhhaya/fmpe/runs/logs/%x-%A_%a.out
#SBATCH -e /data/gpfs/projects/punim0011/mhhaya/fmpe/runs/logs/%x-%A_%a.err

# ---- placeholders: project directory and repository checkout ---------------- #
FM_PROJECT="/data/gpfs/projects/punim0011/mhhaya"
FM_REPO="$FM_PROJECT/FlowMatchingAutoEncoding"
case $FM_PROJECT in *"<projectID>"*) echo "fill in <projectID> in spartan/* first" >&2; exit 1;; esac
export FM_RUNS_DIR=$FM_PROJECT/fmpe/runs
cd "$FM_REPO" || exit 1

module purge
module load foss/2022a Python/3.10.4
source ~/venvs/fmpe/bin/activate

RUN=${FM_RUN:?set FM_RUN=<run dir>: FM_RUN=... sbatch spartan/sample_tier3.slurm}
mkdir -p "$RUN/logs"
# Seeds 1..30 are independent of each other and of the trainer's own draws (seed 0).
python -u sample_posterior.py "$RUN" --n 1000000 --chunk 200000 --device cuda \
    --seed "$SLURM_ARRAY_TASK_ID" --tag "$SLURM_ARRAY_TASK_ID" \
    2>&1 | tee -a "$RUN/logs/sample-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exit "${PIPESTATUS[0]}"
