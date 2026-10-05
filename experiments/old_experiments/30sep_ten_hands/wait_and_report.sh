#!/bin/bash
# Wake the operator on a NEW failure (Slurm state, CUDA OOM, PhysX scene corruption) or after $1 minutes.
cd /share/portal/kk837/gen_mechanics
D=experiments/old_experiments/30sep_ten_hands; R=debug_outputs/train_logs/30sep_ten_hands; MIN=${1:-60}; t0=$(date +%s)
while (( $(date +%s) - t0 < MIN * 60 )); do
  bad=""
  while read name j; do
    grep -qws "$j" $D/.reported_failures 2>/dev/null && continue
    st=$(sacct -j $j -X -n -o State 2>/dev/null | tr -d ' '); e=$(ls $R/*-$j.err 2>/dev/null | head -1)
    [[ "$st" =~ ^(FAILED|CANCELLED.*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)$ ]] && bad+="$name $j $st"$'\n'
    [[ -n "$e" ]] && grep -qa "OutOfMemoryError" "$e" && bad+="$name $j CUDA_OOM"$'\n'
    [[ -n "$e" ]] && grep -qa "Scene state is corrupted" "$e" && { scancel $j; bad+="$name $j PHYSX_CORRUPTED (cancelled)"$'\n'; }
  done < $D/.jobs
  if [[ -n "$bad" ]]; then echo "NEW FAILURES:"; echo -n "$bad"; awk '{print $2}' <<< "$bad" >> $D/.reported_failures; break; fi
  sleep 60
done
echo "== queue"; squeue -u kk837 -h -o "%i %j %T %M" | grep -E "L_|sw" 
echo "== board"; .venv_isaacsim/bin/python $D/status.py 2>/dev/null
