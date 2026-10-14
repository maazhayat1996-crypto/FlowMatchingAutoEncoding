#!/bin/bash
# Paper-scale training (1e7 sims) + 3e6 posterior draws on one A100.
#   sbatch spartan/train.slurm [extra trainer.py args]     # new run
#   sbatch spartan/train.slurm --resume RUN_DIR            # continue a run
# If trainer.py stops on the time budget (exit 3) the job resubmits itself with
# --resume, at most FM_MAX_RESUBMIT (default 5) times in a row.
# Label-noise mode needs its own simulation cache, e.g.
#   SIMS_CACHE=/data/gpfs/projects/<projectID>/fmpe/runs/sims_1e7_seed0_Tnoise.npz \
#       sbatch spartan/train.slurm --labels T --label-noise
#SBATCH --job-name=fmpe-train
#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH -o /data/gpfs/projects/<projectID>/fmpe/logs/%x-%j.out
#SBATCH -e /data/gpfs/projects/<projectID>/fmpe/logs/%x-%j.err

FM_PROJECT="/data/gpfs/projects/<projectID>"
case $FM_PROJECT in *"<projectID>"*) echo "fill in <projectID> in spartan/*.slurm first" >&2; exit 1;; esac
export FM_RUNS_DIR=$FM_PROJECT/fmpe/runs
cd "$FM_PROJECT/FlowMatchingAutoEncoding" || exit 1

module purge
module load foss/2022a Python/3.10.4
source ~/venvs/fmpe/bin/activate

if [ "${1:-}" = "--resume" ]; then
    RUN=${2:?usage: sbatch spartan/train.slurm --resume RUN_DIR}
    shift 2
    ARGS=(--resume "$RUN" "$@")
else
    RUN=$FM_RUNS_DIR/$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID
    SIMS_CACHE=${SIMS_CACHE:-$FM_RUNS_DIR/sims_1e7_seed0.npz}
    ARGS=(--out "$RUN" --n-sims 10000000 --n-post 3000000 --sims-cache "$SIMS_CACHE" "$@")
fi
mkdir -p "$RUN/logs"
LOG=$RUN/logs/train-$SLURM_JOB_ID.log
echo "run dir: $RUN   (job $SLURM_JOB_ID, resubmission ${FM_RESUBMIT:-0})" | tee -a "$LOG"

python -u trainer.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}

if [ "$status" -eq 3 ]; then
    n=${FM_RESUBMIT:-0}
    if [ "$n" -lt "${FM_MAX_RESUBMIT:-5}" ]; then
        sbatch --dependency=afterany:"$SLURM_JOB_ID" --export=ALL,FM_RESUBMIT=$((n + 1)) \
            spartan/train.slurm --resume "$RUN" 2>&1 | tee -a "$LOG"
    else
        echo "not resubmitting: FM_MAX_RESUBMIT=${FM_MAX_RESUBMIT:-5} reached;" \
             "continue by hand with: sbatch spartan/train.slurm --resume $RUN" | tee -a "$LOG"
    fi
fi
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State | tee -a "$LOG"
exit "$status"
