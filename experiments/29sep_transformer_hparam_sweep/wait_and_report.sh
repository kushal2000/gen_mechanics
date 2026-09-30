#!/bin/bash
# Wake the operator: exit on the first failed sweep job, or after $1 minutes; then print the board.
cd /share/portal/kk837/gen_mechanics
D=experiments/29sep_transformer_hparam_sweep; MIN=${1:-60}; t0=$(date +%s)
jobs=$(awk '{print $2}' $D/.round*_jobs 2>/dev/null | paste -sd,)
while (( $(date +%s) - t0 < MIN * 60 )); do
  # Only NEW failures wake the operator; ones already reported are listed in .reported_failures.
  bad=$(sacct -j "$jobs" -X -n -o JobID,JobName,State 2>/dev/null | grep -E "FAILED|OUT_OF_ME|TIMEOUT|NODE_FAIL" \
        | grep -vwFf <(cat $D/.reported_failures 2>/dev/null; echo __none__))
  if [[ -n "$bad" ]]; then
    echo "NEW FAILURES:"; echo "$bad"; awk '{print $1}' <<< "$bad" >> $D/.reported_failures; break
  fi
  sleep 60
done
echo "== queue"; squeue -u kk837 -h -o "%i %j %T %M" | grep -vE "ladder|bash"
echo "== board"; .venv_isaacsim/bin/python $D/sweep_status.py gen_sharpa 2>/dev/null
