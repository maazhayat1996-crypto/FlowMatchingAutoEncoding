#!/bin/bash
# Extra posterior draws from a trained run: 10 independent tasks x 2e6 samples.
#   sbatch spartan/sample.slurm RUN_DIR
# then, with the array job id it prints:
#   sbatch --dependency=afterok:<array job id> spartan/merge.slurm RUN_DIR
#SBATCH --job-name=fmpe-sample
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=1-10
#SBATCH -o /data/gpfs/projects/<projectID>/fmpe/logs/%x-%A_%a.out
#SBATCH -e /data/gpfs/projects/<projectID>/fmpe/logs/%x-%A_%a.err

FM_PROJECT="/data/gpfs/projects/<projectID>"
case $FM_PROJECT in *"<projectID>"*) echo "fill in <projectID> in spartan/*.slurm first" >&2; exit 1;; esac
export FM_RUNS_DIR=$FM_PROJECT/fmpe/runs
cd "$FM_PROJECT/FlowMatchingAutoEncoding" || exit 1

module purge
module load foss/2022a Python/3.10.4
source ~/venvs/fmpe/bin/activate

RUN=${1:?usage: sbatch spartan/sample.slurm RUN_DIR}
mkdir -p "$RUN/logs"
# Seeds 1..10 are independent of each other and of the trainer's own draws
# (which use the training seed, 0 by default).
python -u sample_posterior.py "$RUN" --n 2000000 --chunk 200000 --device cuda \
    --seed "$SLURM_ARRAY_TASK_ID" --tag "$SLURM_ARRAY_TASK_ID" \
    2>&1 | tee -a "$RUN/logs/sample-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exit "${PIPESTATUS[0]}"
