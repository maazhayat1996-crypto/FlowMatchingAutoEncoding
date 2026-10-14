#!/bin/bash
# Tier 1 bulk posterior draws: 10 tasks x 3e5 = 3e6, the paper's count.
#   FM_RUN=<run dir> sbatch spartan/sample_tier1.slurm     # prints the array job id
# then: FM_RUN=<run dir> sbatch --dependency=afterok:<array job id> spartan/merge_tier1.slurm
#
# Resize --n / --time from measured throughput: "sampling_samples_per_s" in
# <run>/summary.json (the trained network, best estimate) or the samples/s of
# the smoke benchmark.  Each task needs n / (samples/s) seconds plus ~5 min.
# The paper's laptop GPU managed ~76 samples/s; if an A100 task is below
# ~25 samples/s, something is wrong.  A task near its time limit stops 5 min
# early and saves what it has.
#SBATCH --job-name=fmpe-sample-t1
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=1-10
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

RUN=${FM_RUN:?set FM_RUN=<run dir>: FM_RUN=... sbatch spartan/sample_tier1.slurm}
mkdir -p "$RUN/logs"
# Seeds 1..10 are independent of each other and of the trainer's own draws
# (training seed, 0 by default).
python -u sample_posterior.py "$RUN" --n 300000 --chunk 200000 --device cuda \
    --seed "$SLURM_ARRAY_TASK_ID" --tag "$SLURM_ARRAY_TASK_ID" \
    2>&1 | tee -a "$RUN/logs/sample-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exit "${PIPESTATUS[0]}"
