#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=gfem-scale
# One Slurm job per rung per solver: runs each line of a step file in order, and leaves a
# record for every step, however it ended. Account, QoS, nodes, limit and output belong to
# the command line, since they are the user's and the row's:
#
#   export SCALE_DIR=...   # made by scale_env.sh: env/, COMMIT, records/, logs/, the scripts
#   sbatch --account=... --qos=... --nodes=N --time=... --constraint=highmem \
#       --output="$SCALE_DIR/logs/job-%j.out" scale_job.sbatch STEPFILE
#
# A step line, as tools/scale_steps.py writes them:
#   NAME RPN CPT CASE SOLVER GHZ [scale_run options...]   RPN 0: python with no launcher, CPT threads
#   check NAME REF GATE|observe
#   bench NAME RPN CPT node|net [scale_bench options...]  RPN 0: python with no launcher
#
# Before a run its stub says STARTED, so a job killed with this script still leaves a row;
# close it by hand afterwards with what sacct says (`scale_run.py close NAME --state ...`).
# A step killed by Slurm disables later steps in its job (the script then stops with rc 3),
# so a step that may be killed runs alone. SCALE_MODULES names the MPI stack's modules.
set -u
: "${SCALE_DIR:?set SCALE_DIR to the directory scale_env.sh made}"
STEPS=${1:?usage: scale_job.sbatch STEPFILE}
O=$SCALE_DIR/records; L=$SCALE_DIR/logs/steps; PY=$SCALE_DIR/env/bin/python; RUN=$SCALE_DIR/scale_run.py
mkdir -p "$O" "$L"
module purge >/dev/null 2>&1
# shellcheck disable=SC2086
module load ${SCALE_MODULES:-openmpi/4.1.6-gcc} >/dev/null 2>&1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMPI_MCA_opal_warn_on_missing_libcuda=0
SCALE_COMMIT=$(cat "$SCALE_DIR/COMMIT"); export SCALE_COMMIT
JOBINFO=$(scontrol show job "$SLURM_JOB_ID" -o)
SCALE_QOS=$(sed -n 's/.* QOS=\([^ ]*\).*/\1/p' <<<"$JOBINFO"); export SCALE_QOS
SCALE_CONSTRAINT=$(sed -n 's/.* Features=\([^ ]*\).*/\1/p' <<<"$JOBINFO")
[ "$SCALE_CONSTRAINT" = "(null)" ] && SCALE_CONSTRAINT=
export SCALE_CONSTRAINT
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NGS_NUM_THREADS
echo "job $SLURM_JOB_ID nodes $SLURM_JOB_NUM_NODES $(scontrol show hostnames | tr '\n' ' ') qos $SCALE_QOS constraint $SCALE_CONSTRAINT steps $STEPS commit $SCALE_COMMIT $(date +%T)"
END=$(( $(date +%s) + $(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{n=NF; s=$n+60*$(n-1); if(n>=3)s+=3600*$(n-2); if(n>=4)s+=86400*$(n-3); print s}') ))

# Slurm's state of the newest numbered step, once it has settled; a by-whom is dropped.
step_state() {
  local st= i
  for i in 1 2 3 4 5 6; do
    st=$(sacct -n -P -j "$SLURM_JOB_ID" --format=JobID,State | awk -F'|' '$1 ~ /\.[0-9]+$/ {s=$2} END{print s}')
    case "$st" in ''|RUNNING|COMPLETING) sleep 5 ;; *) break ;; esac
  done
  echo "${st%% *}"
}

FAIL=0
while read -r name a b c d e rest; do
  case "$name" in ''|\#*) continue ;; esac
  if [ "$name" = check ]; then
    if [ "$c" = observe ]; then g=--observe; else g="--gate $c"; fi
    # shellcheck disable=SC2086
    "$PY" "$RUN" check "$O/$a" "$O/$b" $g || FAIL=1
    continue
  fi
  left=$(( END - $(date +%s) - 60 ))
  if [ "$name" = bench ]; then
    # a bench line: bench NAME RPN CPT MODE [options...]
    log=$L/$a-$SLURM_JOB_ID.log
    echo "--- bench $a rpn=$b cpt=$c mode=$d $e $rest left=${left}s $(date +%T)"
    s=$SECONDS
    if [ "$b" = 0 ]; then
      # shellcheck disable=SC2086
      timeout -k 10 "$left" "$PY" -u "$SCALE_DIR/scale_bench.py" "$d" --out "$O/$a.json" $e $rest > "$log" 2>&1 < /dev/null
    else
      # shellcheck disable=SC2086
      timeout -k 30 "$left" srun --mpi=pmix --kill-on-bad-exit=1 -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node="$b" --cpus-per-task="$c" \
        "$PY" -u "$SCALE_DIR/scale_bench.py" "$d" --out "$O/$a.json" $e $rest > "$log" 2>&1 < /dev/null
    fi
    rc=$?
    echo "rc=$rc wall=$((SECONDS-s))s"
    [ "$rc" -ne 0 ] && FAIL=1 && tail -25 "$log" | cut -c1-240
    continue
  fi

  log=$L/$name-$SLURM_JOB_ID.log
  repeat=$(sed -n 's/.*--repeat \([0-9]*\).*/\1/p' <<<"$rest")
  facts=(--case "$c" --solver "$d" --ghz "$e" --ranks-per-node "$a" --threads-per-rank "$b" ${repeat:+--repeat "$repeat"})
  if [ "$left" -lt 120 ]; then
    echo "--- $name not started: ${left}s left"
    "$PY" "$RUN" close "$O/$name" --state NOT_STARTED "${facts[@]}"
    FAIL=1
    continue
  fi
  "$PY" "$RUN" close "$O/$name" --state STARTED "${facts[@]}"
  echo "--- $name rpn=$a cpt=$b case=$c solver=$d ghz=$e $rest left=${left}s $(date +%T)"
  s=$SECONDS
  if [ "$a" = 0 ]; then
    # shellcheck disable=SC2086
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS="$b" OPENBLAS_NUM_THREADS=$(( b < 64 ? b : 64 )) NGS_NUM_THREADS="$b" \
      timeout -k 10 "$left" "$PY" -u "$RUN" run "$c" "$O/$name" --solver "$d" --ghz "$e" --threads "$b" \
      --trace --dump-after $(( left - 60 )) $rest > "$log" 2>&1 < /dev/null
  else
    # shellcheck disable=SC2086
    timeout -k 30 "$left" srun --mpi=pmix --kill-on-bad-exit=1 -N "$SLURM_JOB_NUM_NODES" --ntasks-per-node="$a" --cpus-per-task="$b" \
      "$PY" -u "$RUN" run "$c" "$O/$name" --solver "$d" --ghz "$e" \
      --trace --dump-after $(( left - 90 )) $rest > "$log" 2>&1 < /dev/null
  fi
  rc=$?
  echo "rc=$rc wall=$((SECONDS-s))s"
  if [ "$rc" = 0 ]; then state=COMPLETED
  elif [ "$rc" = 124 ]; then state=TIMEOUT
  elif [ "$a" = 0 ]; then state=EXIT_$rc
  else state=$(step_state); fi
  "$PY" "$RUN" close "$O/$name" --log "$log" --state "${state:-UNKNOWN}" --rc "$rc" "${facts[@]}"
  [ "$rc" -ne 0 ] && FAIL=1 && echo "state=$state" && tail -25 "$log" | cut -c1-240
  grep -aq "step creation still disabled" "$log" && { echo "STOP: step creation disabled"; exit 3; }
done < "$STEPS"
echo "done fail=$FAIL $(date +%T)"
exit $FAIL
