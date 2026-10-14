#!/bin/bash
# Merge <run>/samples/* (array tasks + the training job's own draws) into
# G_samples.npy / chi2.npy / samples_summary.json, then draw the figures.
#   FM_RUN=<run dir> sbatch --dependency=afterok:<array job id> spartan/merge_tier1.slurm
#SBATCH --job-name=fmpe-merge-t1
#SBATCH -p sapphire
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0:20:00
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

RUN=${FM_RUN:?set FM_RUN=<run dir>: FM_RUN=... sbatch spartan/merge_tier1.slurm}
mkdir -p "$RUN/logs"
{
    python -u merge_samples.py "$RUN" && python -u plot_posterior.py "$RUN"
} 2>&1 | tee -a "$RUN/logs/merge-$SLURM_JOB_ID.log"
exit "${PIPESTATUS[0]}"
