#!/bin/bash
# Overnight watchdog for the three multi-embodiment control arms.
#
# Runs detached (nohup/setsid) so it survives the session that started it.
#
# What it does:
#   1. waits for the 24k population manifest, verifies the hand count, and
#      submits arm 3 -- it will NOT submit against a short population, because
#      a truncated pool would cycle designs and log identically to a real k=n run
#   2. samples all three arms every 5 minutes and appends one line each to a
#      single report file
#   3. if an arm disappears from the queue without reaching MAX EPOCHS, captures
#      the tail of its stderr into the report so the cause is on record
#   4. resubmits an arm AT MOST ONCE, and only when its log contains no Python
#      traceback -- i.e. when the failure looks like infrastructure rather than
#      our code. A real config error must not burn allocations in a retry loop.
#
# Everything lands in bench_dir/mec_overnight/report.log.

set -uo pipefail
REPO_ROOT="/share/portal/kk837/gen_mechanics"
cd "$REPO_ROOT"

OUT_DIR="${REPO_ROOT}/bench_dir/mec_overnight"
mkdir -p "$OUT_DIR"
REPORT="${OUT_DIR}/report.log"
MANIFEST="${REPO_ROOT}/assets/urdf/generated/population/seed_0002/manifest.json"
POP_JOB="${POP_JOB:-81949}"
INTERVAL="${INTERVAL:-300}"
MAX_HOURS="${MAX_HOURS:-14}"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$REPORT"; }

say "=== overnight monitor started (pid $$) ==="
say "arms: 1=mec_sharpa 2=mec_gen_sharpa_like 3=mec_population24k"

# --- 1. wait for the population, then submit arm 3 ------------------------
ARM3_JOB=""
while true; do
    if [ -f "$MANIFEST" ]; then
        N=$(python3 -c "import json;print(len(json.load(open('$MANIFEST'))['hands']))" 2>/dev/null || echo 0)
        if [ "${N:-0}" -ge 24576 ]; then
            say "population ready: $N hands -- submitting arm 3"
            ARM3_JOB=$(sbatch --parsable experiments/multi_embodiment_control/03_population_24k.sub 2>&1)
            say "arm 3 submitted: job $ARM3_JOB"
        else
            say "population manifest has only $N hands (<24576) -- NOT submitting arm 3"
        fi
        break
    fi
    if ! squeue -j "$POP_JOB" -h >/dev/null 2>&1 || [ -z "$(squeue -j "$POP_JOB" -h 2>/dev/null)" ]; then
        say "population job $POP_JOB ended with no manifest -- arm 3 NOT submitted"
        say "$(tail -5 "${REPO_ROOT}/bench_dir/population/seed2_n24576.log" 2>/dev/null)"
        break
    fi
    sleep 60
done

# --- 2. monitor loop -------------------------------------------------------
declare -A RETRIED=( [mec_sharpa]=0 [mec_gen_sharpa_like]=0 [mec_population24k]=0 )
declare -A SUBFILE=(
    [mec_sharpa]=01_sharpa.sub
    [mec_gen_sharpa_like]=02_gen_sharpa_like.sub
    [mec_population24k]=03_population_24k.sub
)
END=$(( $(date +%s) + MAX_HOURS * 3600 ))

while [ "$(date +%s)" -lt "$END" ]; do
    running=0
    for tag in mec_sharpa mec_gen_sharpa_like mec_population24k; do
        # newest run dir for this arm
        d=$(ls -td "${REPO_ROOT}"/train_dir/gen_mechanics/multi_embodiment_control/${tag}_seed*/ 2>/dev/null | head -1)
        alive=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^${tag}_seed" || true)
        if [ "${alive:-0}" -gt 0 ]; then
            running=$(( running + 1 ))
            if [ -n "$d" ]; then
                line=$(grep -aoE "epoch +: [0-9]+ / [0-9]+|fps step: *[0-9.]+|rew: *[-0-9.]+" \
                       "$d/slurm.log" 2>/dev/null | tail -3 | tr '\n' ' ')
                say "$tag RUNNING  ${line:-（no metrics yet）}"
            else
                say "$tag RUNNING  (no run dir yet)"
            fi
        else
            if [ -z "$d" ]; then
                say "$tag NOT RUNNING, no run dir"
                continue
            fi
            if grep -qa "MAX EPOCHS NUM" "$d/slurm.log" 2>/dev/null; then
                say "$tag FINISHED (reached max epochs)"
                continue
            fi
            # died
            if [ "${RETRIED[$tag]}" -eq 0 ]; then
                if grep -qa "^Traceback" "$d/slurm.err" 2>/dev/null; then
                    say "$tag DIED with a Python traceback -- NOT retrying:"
                    say "$(grep -a -A 6 '^Traceback' "$d/slurm.err" | tail -12)"
                    RETRIED[$tag]=1
                else
                    say "$tag DIED with no traceback (looks like infrastructure) -- retrying once"
                    say "$(tail -8 "$d/slurm.err" 2>/dev/null)"
                    sbatch "experiments/multi_embodiment_control/${SUBFILE[$tag]}" >> "$REPORT" 2>&1
                    RETRIED[$tag]=1
                fi
            fi
        fi
    done
    say "--- $running/3 arms running ---"
    sleep "$INTERVAL"
done

say "=== monitor window ended after ${MAX_HOURS}h ==="
