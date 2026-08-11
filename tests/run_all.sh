#!/usr/bin/env bash
# Run the test suite serially.
#
# These are standalone scripts, not pytest cases: each boots Isaac Sim Kit in
# its own process (~60-120 s) because Kit cannot be torn down and re-created
# in-process. Running them under one pytest session would boot Kit once and then
# hang on the second env teardown, so they must stay separate processes.
#
#   tests/run_all.sh              # everything
#   tests/run_all.sh fast         # skip the slow parity + pretrained rollout
#
# Requires a GPU. Only run one instance at a time — one Kit per GPU.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv_isaacsim/bin/python}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_CACHE_PATH="${OMNI_KIT_CACHE_PATH:-/tmp/${USER}_ov_cache}"
mkdir -p "$OMNI_KIT_CACHE_PATH"

MODE="${1:-all}"

TESTS=(
    "tests/test_load_isaacsim.py"
    "tests/test_gym_register.py"
    "tests/test_env_smoke.py --num_envs 8 --num_assets_per_type 2 --steps 10"
    "tests/test_obs_action_spec.py"
    "tests/test_action_pipeline.py --num_envs 4 --num_assets_per_type 1"
    "tests/test_robot_spec_invariants.py"
)
SLOW=(
    "tests/test_sharpa_parity.py"
    "tests/test_pretrained_rollout.py --num_envs 8 --num_assets_per_type 2 --num_steps 600"
)
[[ "$MODE" != "fast" ]] && TESTS+=("${SLOW[@]}")

pass=0; fail=0; failed_names=()
for t in "${TESTS[@]}"; do
    name="${t%% *}"
    printf '\n\033[1m=== %s ===\033[0m\n' "$t"
    start=$SECONDS
    if $PY $t > "/tmp/$(basename "$name" .py).log" 2>&1; then
        printf '\033[32mPASS\033[0m  %s  (%ss)\n' "$name" "$((SECONDS - start))"
        pass=$((pass + 1))
    else
        printf '\033[31mFAIL\033[0m  %s  (%ss)  — see /tmp/%s.log\n' \
            "$name" "$((SECONDS - start))" "$(basename "$name" .py)"
        tail -20 "/tmp/$(basename "$name" .py).log"
        fail=$((fail + 1)); failed_names+=("$name")
    fi
done

printf '\n\033[1m=== %d passed, %d failed ===\033[0m\n' "$pass" "$fail"
if (( fail )); then printf 'failed: %s\n' "${failed_names[*]}"; exit 1; fi
