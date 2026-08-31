#!/bin/bash
# Sample CPU/RAM and GPU usage for as long as the parent job lives.
#
#   experiments/monitor_usage.sh <out_prefix> [interval_seconds] &
#   MONITOR_PID=$!
#   trap 'kill $MONITOR_PID 2>/dev/null || true' EXIT
#
# Writes <prefix>.ram_usage.csv and <prefix>.gpu_usage.csv. Exits on its own
# when the parent does, so a caller that forgets the trap still cleans up.
#
# RAM is summed over the job's whole process group, not just python: Isaac Sim
# forks helpers, and the number that matters for --mem is the total.
set -uo pipefail

PREFIX="${1:?usage: monitor_usage.sh <out_prefix> [interval_seconds]}"
INTERVAL="${2:-30}"
PARENT=$PPID
PGID=$(ps -o pgid= -p "$PARENT" 2>/dev/null | tr -d ' ')

echo "timestamp,rss_gb,cpu_pct,nprocs" > "${PREFIX}.ram_usage.csv"
echo "timestamp,gpu,name,util_gpu_pct,util_mem_pct,mem_used_mib,mem_total_mib,sm_clock_mhz,power_w" \
    > "${PREFIX}.gpu_usage.csv"

while kill -0 "$PARENT" 2>/dev/null; do
    ts=$(date +%Y-%m-%dT%H:%M:%S)

    # `ps -g` selects by session in procps, not by process group -- filter on
    # the pgid column instead, or every sample reads zero.
    ps -eo pgid=,rss=,pcpu= 2>/dev/null | awk -v ts="$ts" -v pg="$PGID" '
        $1 == pg {rss += $2; cpu += $3; n++}
        END {printf "%s,%.2f,%.1f,%d\n", ts, rss/1048576, cpu, n}' >> "${PREFIX}.ram_usage.csv"

    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.sm,power.draw \
               --format=csv,noheader,nounits 2>/dev/null \
        | sed 's/, /,/g' | sed "s|^|${ts},|" >> "${PREFIX}.gpu_usage.csv"

    sleep "$INTERVAL"
done
