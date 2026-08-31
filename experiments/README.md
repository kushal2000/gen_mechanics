# experiments

SLURM submit scripts. One `.sub` per kind of run, parameterised by environment
variable so a variant is a prefix (`SEED=1 sbatch ...`) rather than an edited
copy.

Start from [`train_single_embodiment_sharpa.sub`](train_single_embodiment_sharpa.sub)
— it is the reference run and the template to copy.

```bash
sbatch experiments/train_single_embodiment_sharpa.sub
SEED=1 MAX_EPOCHS=20000 sbatch experiments/train_single_embodiment_sharpa.sub

squeue -u $USER
tail -f train_dir/gen_mechanics/sharpa_iiwa14/<run>/slurm.log
```

`old_experiments/` is an archive of what was run before the repo was split into
three packages. Several of its scripts launch modules that no longer exist; it
is kept as a record, not as scripts that still work.

## Writing a new one

Copy the reference script and change the knobs. The five sections below are the
skeleton, in the order they must appear.

### 1. Resources

```bash
#SBATCH --cpus-per-task=8      # GPU jobs queue faster with fewer CPUs
#SBATCH --mem=96000            # memory is the real constraint, not CPUs
#SBATCH -t 3-00:00:00          # a ceiling, not the budget (see below)
#SBATCH --gres=gpu:1
#SBATCH --partition=portal
#SBATCH -o /dev/null           # redirect inside the script instead
#SBATCH -e /dev/null
```

Ask for fewer CPUs than feels right. A 24,576-env run measured 32.7 GB MaxRSS
against a 200 GB request, so memory was over-asked while CPU count was what
kept the job queued. Dropping to 1 CPU is too far — it regressed Kit boot from
733 s to 916 s. Eight is the sweet spot found so far.

`-o`/`-e` go to `/dev/null` on purpose: the script redirects into the run
directory once it knows the name, so logs sit beside the checkpoints instead of
wherever `sbatch` happened to be called from.

### 2. Knobs with defaults

```bash
ROBOT="${ROBOT:-sharpa_iiwa14}"
SEED="${SEED:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-120000}"
```

Every tunable gets a default. This is what makes `SEED=1 sbatch ...` work
without touching the file, and it keeps the committed script a faithful record
of the canonical run.

### 3. Guards

Check anything that would otherwise fail deep inside rl_games with an
unhelpful traceback, and fail loudly at the top instead:

- `num_envs % expl_coef_block_size == 0` — SAPG partitions envs into
  exploration blocks and needs an exact division.
- `minibatch_size` must divide `horizon_length * num_envs`. rl_games floors
  `num_minibatches` to 0 otherwise and dies on a `ZeroDivisionError` inside
  `init_tensors`. Scale it with `NUM_ENVS` rather than hardcoding the YAML's
  value, which is sized for 24,576.
- `agent.params.config.name` must start with `<int>_` — SAPG parses the policy
  index off that prefix.

### 4. Run directory, then redirect

```bash
EXPERIMENT_NAME="${ROBOT}_seed${SEED}_$(date +%Y-%m-%d_%H-%M-%S)"
[ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" JobName="$EXPERIMENT_NAME" || true
RUN_DIR="${REPO_ROOT}/train_dir/${WANDB_PROJECT}/${WANDB_GROUP}/${EXPERIMENT_NAME}"
mkdir -p "$RUN_DIR"
exec > "${RUN_DIR}/slurm.log" 2> "${RUN_DIR}/slurm.err"
```

The `scontrol` rename makes `squeue` readable when a dozen jobs are pending.

### 5. Environment, then launch

```bash
cd "$REPO_ROOT"
source .venv_isaacsim/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_CACHE_PATH="/tmp/${USER}_ov_cache"   # local disk, never NFS
mkdir -p "$OMNI_KIT_CACHE_PATH"

python -u coevolution/train.py \
    --task GenMech-PoseReach-Direct-v0 \
    --agent rl_games_sapg_cfg_entry_point \
    --headless \
    env.assets.robot_spec="$ROBOT" \
    hydra.run.dir="$RUN_DIR"
```

`--flags` come first, bare `KEY=VALUE` hydra overrides after. Use `python -u`
so the log is not a blank file for the first ten minutes.

Task ids: `GenMech-PoseReach-Direct-v0` for one hand,
`GenMech-PoseReachMulti-Direct-v0` for a population sharing one articulation
view.

## Traps

These cost time when violated.

- **Epoch-bound, not walltime-bound.** A hand that simulates faster collects
  more gradient steps in the same wall clock, a confound indistinguishable from
  better hardware. Set `max_epochs`; treat SBATCH `-t` as a ceiling.
- **Hydra type-checks overrides against the *runtime* type of the default.** A
  field declared `str | None = None` rejects every string override with
  `Incorrect type under namespace`. Declare `str = ""` and test for empty.
  This has killed launches twice, on `reset.fixed_trajectory_file` and on
  `assets.robot_population_path`.
- **Never build a command with `${var:+KEY="$val"}` inside it.** Bash parses
  the *next* assignment as a command name and the job silently never
  submits. Use `env KEY="$val" sbatch ...`, and check the returned job id
  rather than counting loop iterations.
- **Validate every job id.** `sbatch` failing per-job is easy to miss in a
  submit loop; 66 jobs once reported as submitted when zero were.
- **`AppLauncher` before any `isaaclab.*` import.** Isaac Lab's sub-namespaces
  only resolve after it has booted Kit.
- **One Isaac Sim instance per GPU.** A second Kit boot on a busy GPU can crash
  the booting process mid-startup.
- **Grep logs with `[0-9,]*`.** Epoch lines print `epoch : 2,007 / 120,000`
  with comma separators, and a `[0-9]*` pattern silently returns a stale
  comma-free number — which twice looked like a stalled run that was healthy.
