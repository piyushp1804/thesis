#!/usr/bin/env bash
# Resilient overnight runner for 200-bar batches.
#
# Safe to invoke multiple times (or after a crash): each seed that
# already has a pickle on disk is skipped via run_batch.py checkpoint
# logic, so work resumes from wherever the previous attempt died.
#
# Never uses `set -e` — every batch runs independently, every failure
# is logged, the script continues to the next step.

cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
LOG="logs/200bar_overnight.log"
mkdir -p logs results

source venv/bin/activate

# tqdm is noisy with ANSI on non-tty; disable fancy bar for a clean log.
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

stamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

run_with_retry() {
  local max_tries=3
  local try=1
  local label="$1"
  shift
  while [ $try -le $max_tries ]; do
    log "begin $label attempt $try/$max_tries"
    if "$@" >>"$LOG" 2>&1; then
      log "ok    $label attempt $try"
      return 0
    fi
    log "fail  $label attempt $try (exit $?)"
    try=$((try + 1))
    sleep 5
  done
  log "GAVE_UP $label after $max_tries attempts — continuing to next batch"
  return 1
}

log "=========================================================="
log "overnight 200-bar runner starting"
log "ROOT=$ROOT"
log "python=$(which python)"
log "=========================================================="

run_with_retry "GA 200bar"    python scripts/run_batch.py --algo ga    --bench 200bar --seeds 42 123 456 789 2024 --pop-size 100 --n-gen 200
run_with_retry "PSO 200bar"   python scripts/run_batch.py --algo pso   --bench 200bar --seeds 42 123 456 789 2024 --pop-size 100 --n-gen 200
run_with_retry "NSGA2 200bar" python scripts/run_batch.py --algo nsga2 --bench 200bar --seeds 42 123 456             --pop-size 100 --n-gen 150

log "aggregating results"
run_with_retry "aggregate" python scripts/aggregate_results.py

log "DONE — summary:"
for algo in ga pso nsga2; do
  f="results/200bar_${algo}_summary.csv"
  if [ -f "$f" ]; then
    log "  $f  ($(wc -l <"$f") lines)"
  else
    log "  $f  MISSING"
  fi
done
log "overnight runner exiting"
