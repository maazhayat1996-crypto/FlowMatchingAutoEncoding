# Shared by spartan/train_tier1.slurm and spartan/train.slurm (sourced after the
# module/venv block and the cd into the repository).
#
#   run_trainer SCRIPT [trainer.py args...]
#
# FM_RESUME unset:  new run in $FM_RUNS_DIR/<timestamp>-<jobid> with the given args.
# FM_RESUME=RUN:    python trainer.py --resume RUN [args]; settings that define the
#                   run come from RUN/config.json, so repeated args are ignored.
# If trainer.py exits with 3 (time budget used up before convergence), SCRIPT is
# resubmitted with FM_RESUME=RUN and an afterany dependency on this job, at most
# FM_MAX_RESUBMIT (default 5) times in a row.  The resubmitted job gets the
# script arguments in FM_SCRIPT_ARGS (an array set by the caller).

run_trainer() {
    local script=$1
    shift
    local run
    local -a args
    if [ -n "${FM_RESUME:-}" ]; then
        run=$FM_RESUME
        args=(--resume "$run" "$@")
    else
        run=$FM_RUNS_DIR/$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID
        args=(--out "$run" "$@")
    fi
    mkdir -p "$run/logs"
    local log=$run/logs/train-$SLURM_JOB_ID.log
    echo "run dir: $run  job: $SLURM_JOB_ID  FM_RESUME=${FM_RESUME:-}  resubmission: ${FM_RESUBMIT:-0}" | tee -a "$log"

    python -u trainer.py "${args[@]}" 2>&1 | tee -a "$log"
    local status=${PIPESTATUS[0]}

    if [ "$status" -eq 3 ]; then
        local n=${FM_RESUBMIT:-0}
        if [ "$n" -lt "${FM_MAX_RESUBMIT:-5}" ]; then
            sbatch --dependency=afterany:"$SLURM_JOB_ID" \
                --export=ALL,FM_RESUME="$run",FM_RESUBMIT=$((n + 1)) \
                "$script" "${FM_SCRIPT_ARGS[@]}" 2>&1 | tee -a "$log"
        else
            echo "not resubmitting: FM_MAX_RESUBMIT=${FM_MAX_RESUBMIT:-5} reached; continue with" \
                 "FM_RESUME=$run sbatch $script" | tee -a "$log"
        fi
    fi
    sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,State | tee -a "$log"
    return "$status"
}
