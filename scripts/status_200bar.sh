#!/usr/bin/env bash
# Quick status check on overnight 200-bar run. Safe to run any time.
#
# Usage: bash scripts/status_200bar.sh
#
# Shows:
#   - whether the overnight runner is alive
#   - which seeds have completed (pickles on disk)
#   - last 15 lines of the progress log
#   - whether all three summary CSVs exist

cd "$(dirname "$0")/.." || exit 1

echo "=========================================================="
echo " 200-bar overnight status at $(date +%H:%M:%S)"
echo "=========================================================="

if pgrep -fl "run_batch.py.*200bar" >/dev/null; then
  echo "[alive] run_batch.py is running:"
  pgrep -fl "run_batch.py.*200bar" | sed 's/^/  /'
else
  echo "[dead]  no run_batch.py process"
fi

if pgrep -f "run_overnight_200bar.sh" >/dev/null; then
  echo "[alive] overnight orchestrator is running"
else
  echo "[dead]  overnight orchestrator gone"
fi

echo ""
echo "--- seeds completed (pickles on disk) ---"
for algo in ga pso nsga2; do
  echo -n "  $algo: "
  ls results/200bar_${algo}_seed*.pkl 2>/dev/null | \
    sed 's/.*seed//; s/.pkl$//' | tr '\n' ' '
  echo ""
done

echo ""
echo "--- summary CSVs ---"
for algo in ga pso nsga2; do
  f="results/200bar_${algo}_summary.csv"
  if [ -f "$f" ]; then
    n=$(($(wc -l <"$f") - 1))
    echo "  $f  ($n seeds)"
  else
    echo "  $f  MISSING"
  fi
done

echo ""
echo "--- last 15 log lines ---"
tail -15 logs/200bar_overnight.log 2>/dev/null | sed 's/^/  /'
echo ""
echo "--- if anything looks stuck, restart with: ---"
echo "  bash scripts/run_overnight_200bar.sh &"
echo "(safe: checkpoint logic skips every seed already on disk)"
