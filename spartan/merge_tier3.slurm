#!/bin/bash
# Merge the Tier 3 sample sets (30 array tasks + the training job's own draws)
# into G_samples.npy / chi2.npy / samples_summary.json, then draw the figures.
#   FM_RUN=<run dir> sbatch --dependency=afterok:<array job id> spartan/merge_tier3.slurm
# ~3e7 draws: ~5.5 GB of in-box float64 samples, held twice while merging;
# plotting derives masses/phases for the ~7e6 chi2 < 2500 subset.
#SBATCH --job-name=fmpe-merge-t3
#SBATCH -p sapphire
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
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

RUN=${FM_RUN:?set FM_RUN=<run dir>: FM_RUN=... sbatch spartan/merge_tier3.slurm}
mkdir -p "$RUN/logs"
{
    python -u merge_samples.py "$RUN" && python -u plot_posterior.py "$RUN"
} 2>&1 | tee -a "$RUN/logs/merge-$SLURM_JOB_ID.log"
exit "${PIPESTATUS[0]}"
