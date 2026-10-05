#!/bin/bash
# The starting population: 1024 hands sampled at seed 0, then 500 rounds of
# neutral drift (one mutation per hand per round, no selection). Drift lifts the
# population off the MIN_FINGERS wall the sampler pins it against -- 3 -> 11
# joints per hand -- so the first selection is not simply "more joints".
#
#   make_population.sh [seed] [count] [rounds]
#
# Writes assets/populations/gen_s<seed>_n<count>_drift<rounds>_s<seed>/round_<rounds>.json.
# A drifted population is reproducible only while mutate_design holds still;
# the file is the record, and coevolution_v1's is gen_s0_n1024_drift500_s0.
set -euo pipefail
SEED="${1:-0}" COUNT="${2:-1024}" ROUNDS="${3:-500}"
cd /share/portal/kk837/gen_mechanics
.venv_isaacsim/bin/python -m hand_sampler.population_io "gen_s${SEED}_n${COUNT}"
.venv_isaacsim/bin/python -m hand_sampler.drift "gen_s${SEED}_n${COUNT}" --seed "$SEED" \
    --max-rounds "$ROUNDS" --snapshot-every 25 --force
echo "population: assets/populations/gen_s${SEED}_n${COUNT}_drift${ROUNDS}_s${SEED}/round_$(printf '%04d' "$ROUNDS").json"
