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
# Requires a GPU. Only run one instance at a time -- one Kit per GPU.
#
# WHY EXIT CODES ARE NOT ENOUGH. Kit installs a shutdown handler that calls
# os._exit(0) while unwinding, so a test that dies on an uncaught exception can
# still return 0. This runner once reported 8/8 passing while a test was failing
# on an ImportError. Every test must therefore print an explicit success
# sentinel as its LAST meaningful output, and a test counts as passed only if
# the exit code is 0, the sentinel is present, and no traceback appears.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# -u is load-bearing, not a preference. Kit's shutdown handler calls os._exit(0),
# which skips flushing stdout -- so with block buffering a passing test can lose
# its success sentinel and be reported as a failure (and, worse, the reverse).
# This cost an afternoon once; do not drop the -u.
PY="${PY:-$REPO_ROOT/.venv_isaacsim/bin/python -u}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_CACHE_PATH="${OMNI_KIT_CACHE_PATH:-/tmp/${USER}_ov_cache}"
mkdir -p "$OMNI_KIT_CACHE_PATH"

LOG_DIR="${LOG_DIR:-/tmp/${USER}_genmech_tests}"
mkdir -p "$LOG_DIR"

MODE="${1:-all}"

# "script args :: success sentinel"
TESTS=(
    "tests/test_load_isaacsim.py :: Isaac Sim load OK"
    "tests/test_imports.py :: module import test OK"
    "tests/test_object_pool_reconstruct.py :: object pool reconstruction test OK"
    "tests/test_pose_viewer.py :: pose viewer test OK"
    "tests/test_gym_register.py :: registration smoke test OK"
    "tests/test_env_smoke.py --num_envs 8 --num_assets_per_type 2 --steps 10 :: [smoke] OK"
    "tests/test_obs_action_spec.py :: obs/action spec test OK"
    "tests/test_action_pipeline.py --num_envs 4 --num_assets_per_type 1 :: action pipeline test OK"
    "tests/test_robot_spec_invariants.py :: robot spec invariants OK"
    "tests/test_multi_embodiment_env.py --num_envs 16 --population_count 8 --num_assets_per_type 2 --steps 10 :: multi-embodiment env test OK"
    "tests/test_shape_layout_record.py --population_count 256 :: shape layout record test OK"
    "tests/test_embodiment_viewer_switch.py :: embodiment switch test OK"
)
SLOW=(
    "tests/test_sharpa_parity.py :: SHARPA parity test OK"
    "tests/test_pretrained_rollout.py --num_envs 8 --num_assets_per_type 2 --num_steps 600 :: pretrained rollout test OK"
)
[[ "$MODE" != "fast" ]] && TESTS+=("${SLOW[@]}")

pass=0; fail=0; failed_names=()
for entry in "${TESTS[@]}"; do
    cmd="${entry%% :: *}"
    sentinel="${entry##* :: }"
    name="${cmd%% *}"
    log="$LOG_DIR/$(basename "$name" .py).log"

    printf '\n\033[1m=== %s ===\033[0m\n' "$cmd"
    start=$SECONDS
    $PY $cmd > "$log" 2>&1
    code=$?
    elapsed=$((SECONDS - start))

    reason=""
    if (( code != 0 )); then
        reason="exit $code"
    elif ! grep -qF "$sentinel" "$log"; then
        reason="missing sentinel '$sentinel' (exit code was 0, but Kit's shutdown handler masks failures)"
    elif grep -qE '^Traceback \(most recent call last\)' "$log"; then
        reason="traceback in output"
    fi

    if [[ -z "$reason" ]]; then
        printf '\033[32mPASS\033[0m  %s  (%ss)\n' "$name" "$elapsed"
        pass=$((pass + 1))
    else
        printf '\033[31mFAIL\033[0m  %s  (%ss) -- %s\n' "$name" "$elapsed" "$reason"
        printf '      log: %s\n' "$log"
        grep -E '^(Traceback|[A-Za-z_.]*Error|AssertionError)' "$log" | head -5 | sed 's/^/      /'
        tail -5 "$log" | sed 's/^/      /'
        fail=$((fail + 1)); failed_names+=("$name")
    fi
done

printf '\n\033[1m=== %d passed, %d failed ===\033[0m\n' "$pass" "$fail"
if (( fail )); then printf 'failed: %s\n' "${failed_names[*]}"; exit 1; fi
