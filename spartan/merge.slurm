#!/bin/bash
# Merge RUN_DIR/samples/* into G_samples.npy / chi2.npy and draw the figures.
#   sbatch --dependency=afterok:<sample array job id> spartan/merge.slurm RUN_DIR
# 32G rather than 16G: 10 x 2e6 + 3e6 draws are ~4 GB of float64 samples, and
# merging plus plotting holds a few copies of that.
#SBATCH --job-name=fmpe-merge
#SBATCH -p sapphire
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH -o /data/gpfs/projects/<projectID>/fmpe/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/<projectID>/fmpe/logs/%x-%j.err

FM_PROJECT="/data/gpfs/projects/<projectID>"
case $FM_PROJECT in *"<projectID>"*) echo "fill in <projectID> in spartan/*.slurm first" >&2; exit 1;; esac
export FM_RUNS_DIR=$FM_PROJECT/fmpe/runs
cd "$FM_PROJECT/FlowMatchingAutoEncoding" || exit 1

module purge
module load foss/2022a Python/3.10.4
source ~/venvs/fmpe/bin/activate

RUN=${1:?usage: sbatch spartan/merge.slurm RUN_DIR}
mkdir -p "$RUN/logs"
{
    python -u merge_samples.py "$RUN" && python -u plot_posterior.py "$RUN"
} 2>&1 | tee -a "$RUN/logs/merge-$SLURM_JOB_ID.log"
exit "${PIPESTATUS[0]}"
