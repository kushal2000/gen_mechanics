# gen_mechanics — Evolve to Generalize

**How does a hand's hardware design shape how efficiently it learns, and how well it generalizes across tasks?**

Human-like hands are the default in dexterous manipulation, but nothing says they are the best design for a given task
or set of tasks. This project studies design and control together: train the same policy recipe on many hand designs,
measure what each design makes easy or hard to learn, and evolve designs toward hands that learn quickly and transfer
across tasks.

- **Flagship environment: in-hand olympics** (`isaacsimenvs/inhand_olympics/`). A hand on a 6-DOF dummy wrist, palm down
  over a table, with a pen and a family of tasks under one observation and reward format: pen lift, spin (on the table
  and in the air), flip, 6D repose and twisting a cap off. One run can train several tasks at once (a task one-hot
  tells the policy which), and one scene can hold several hands, each on its own envs.
- **Learning efficiency.** Designs are compared by how fast they learn under one recipe (epochs to one goal per
  episode), not only by final score. Early results: many reasonable designs reach similar final performance, but
  differ in how quickly and reliably they get there; growing the embodiment during training (a thumb+index policy
  fine-tuned toward the full hand) helps the complex hand get started.
- **Generalization across tasks and hands.** Policies are fine-tuned from one task to another (repose to spin and
  flip), and joint-token transformer policies control each joint from its own token, so one policy can run on hands
  it never trained on.
- **Evolution with the hand grammar** (`hand_sampler/grammar/`, Martin's latest). The grammar defines a space of hands
  buildable from Feetech servos; evolution experiments search it with in-hand olympics as the fitness.

## Main code

```bash
export OMNI_KIT_ACCEPT_EULA=YES
PY=.venv_isaacsim/bin/python
```

| What | Command |
|---|---|
| Train a hand on olympics tasks (SLURM) | `TASKS=pen_repose HAND=<robot spec> sbatch experiments/12oct_inhand_olympics/olympics_sapg_mlp.sub` |
| ... the same, locally, as a quick check | `NUM_ENVS=3072 MINIBATCH=12288 EPOCHS=20 WANDB=0 bash experiments/12oct_inhand_olympics/olympics_sapg_mlp.sub` |
| Play any trained fixed-palm policy; switch its task and goals live | `$PY experiments/16oct_viser_evals/viser_play_all.py` |
| Drive a hand by joint sliders in the sim | `$PY experiments/14oct_design_search/viser_hand_play.py` |
| Evaluate checkpoints per task | `$PY experiments/12oct_inhand_olympics/analysis/eval_tasks.py` |
| Build a primitive hand from finger bearings | `$PY experiments/14oct_design_search/primitive_hands.py --bearings 0,90,180,270` |
| Browse the hand grammar (CPU only, no Isaac) | `$PY experiments/grammar_viewer/viewer.py` |
| The original pose-reaching env | `$PY coevolution/train.py --task GenMech-PoseReach-Direct-v0 --agent rl_games_sapg_cfg_entry_point --headless` |

Every experiment folder (`experiments/<date>_<name>/`) has a README with its runs, results and figures; training logs
and checkpoints go to `debug_outputs/train_logs/<experiment>/`.

## Layout

Three source packages, in dependency order. The arrow points the only way an import is allowed to go:

```
hand_sampler  <--  isaacsimenvs  <--  coevolution
```

```
hand_sampler/        hand designs -- PURE PYTHON, no Isaac
  grammar/             the hand grammar (Martin): what a hand can be, the Feetech rules, viability checks
  grammar_bench/       the grammar's tests and benchmarks (analytic and oracle FK, commercial-hand fits)
  design_space.py, mutate_design.py, evolve.py, build.py   the earlier generated-hand design space
isaacsimenvs/        everything that touches Isaac; one subpackage per task
  inhand_olympics/     FLAGSHIP: hand on a dummy wrist + pen; tasks/ (lift, spin, flip, repose, twist),
                       scene.py (single- and multi-hand scenes), twist_pen.py, observations.py, pose_viewer.py
  pose_reaching_6d/    the original task: grasp a tool and drive it through SE(3) goals (arm + hand)
  inhand_reorient/     palm-up in-hand reorientation, a config-only sibling of pose_reaching_6d
coevolution/         training and evaluation
  train.py, cfg/       entry point + hydra task / train (PPO, SAPG) configs
  eval/                checkpoint player, evals; pose_viewer.py: the wandb interactive pose viewer
assets/urdf/         robot + table URDFs and meshes; generated hands land in assets/urdf/generated/
third_party/rl_games/  vendored SAPG fork (NOT the PyPI package)
experiments/         one folder per experiment: SLURM scripts, analysis, plots, README
```

`hand_sampler` imports no simulator, so design sampling, mutation and the grammar's checks run on CPU without booting
Kit.

## Origins

The repo began as a benchmark for comparing dexterous hands on 6D pose reaching (grasp a tool off a table and drive it
through SE(3) goals) under held-out physics, geometry and goals, built on the Isaac Sim path of
[simtoolreal](https://github.com/tylerlum/simtoolreal) (Kedia, Lum, Bohg, Liu). That env lives on as
`isaacsimenvs/pose_reaching_6d/`.

## Installation

This repo runs the pose-reaching task in Isaac Sim via Isaac Lab. It needs
**Python 3.11** (an Isaac Sim 5.x / Isaac Lab 2.3.x requirement) in its own venv at
`.venv_isaacsim/`.

### Prerequisites

- Python 3.11, NVIDIA driver >= 525.60, CUDA 12+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

### Install

```bash
cd /share/portal/kk837/gen_mechanics
uv venv .venv_isaacsim --python 3.11
PY=.venv_isaacsim/bin/python

# PyTorch for CUDA 12.6, pinned. Isaac Sim 5.1 is built against torch 2.7 —
# leaving this unpinned resolves to a much newer torch and breaks the isaacsim
# wheels in non-obvious ways.
uv pip install --python $PY \
  "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" \
  --index-url https://download.pytorch.org/whl/cu126

# Vendored rl_games — the SAPG fork, NOT the PyPI package (see below)
uv pip install --python $PY -e ./third_party/rl_games/

uv pip install --python $PY \
  omegaconf hydra-core "gym==0.23.1" scipy "numpy==1.26.0" yourdfpy viser requests tqdm tyro \
  "imageio[ffmpeg]" wandb termcolor trimesh pandas matplotlib tensorboard

# Isaac Lab + Isaac Sim (~15 GB; first launch builds RTX shaders, ~2-5 min)
uv pip install --python $PY "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com

# CoACD (offline collision decomposition) + the tyro CLI fix. Install AFTER
# isaaclab so it wins the resolution: tyro.cli needs NoExtraItems from
# typing_extensions>=4.13, but the isaaclab install pulls in 4.12.2.
uv pip install --python $PY coacd "typing_extensions>=4.13"

# Register the three packages
uv pip install --python $PY -e . --no-deps
```

Verify:

```bash
.venv_isaacsim/bin/python -c "
import torch, rl_games, hand_sampler, pathlib
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('rl_games', pathlib.Path(rl_games.__file__).resolve())
print('hand_sampler', pathlib.Path(hand_sampler.__file__).resolve())
"
```

`rl_games` must resolve under `third_party/rl_games/`. The PyPI `rl-games` lacks
the SAPG additions this project trains with (`expl_coef_block_size`,
`use_others_experience`, and the `policy_idx` parse from the config `name`'s
`<int>_` prefix), and swapping it in fails in confusing ways rather than loudly.

**Keep the pins exact.** The validated set is `isaaclab==2.3.2.post1`,
`isaacsim==5.1.0.0`, `torch==2.7.0+cu126`, `numpy==1.26.0` — copied from
simtoolreal's working environment. Newer Isaac Lab releases change the
`DirectRLEnv` and `UrdfConverter` APIs that `isaacsimenvs/pose_reaching_6d/` depends
on, and an unpinned `torch` resolves far past what the isaacsim wheels support.

### Running

```bash
export OMNI_KIT_ACCEPT_EULA=YES
# Point the Omniverse shader cache at local disk, not NFS
export OMNI_KIT_CACHE_PATH=/tmp/$USER/ov_cache && mkdir -p "$OMNI_KIT_CACHE_PATH"
```

Training:

```bash
.venv_isaacsim/bin/python coevolution/train.py \
  --task GenMech-PoseReach-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless
```

### Gotchas

These are the ones that cost time when violated:

- **`AppLauncher` before any `isaaclab.*` import.** Isaac Lab's sub-namespaces
  (`isaaclab.sim`, `isaaclab.envs`, …) only resolve once `AppLauncher(args)` has
  run. Every entry-point script constructs it first, then imports.
- **Kit hangs on shutdown.** Scripts flush stdout/stderr and call `os._exit(0)`
  rather than waiting for a clean teardown.
- **One Isaac Sim instance per GPU.** Booting a second Kit process on a GPU that
  already has one can crash the booting process mid-startup.
- **First startup is slow.** Object URDFs are generated and converted to USD per
  launch; time scales with `assets.num_assets_per_type`.
- **SAPG constraints.** `agent.params.config.name` must start with `<int>_`, and
  `num_envs % expl_coef_block_size == 0`.
- **Training runs must be epoch-bound, not walltime-bound.** A walltime budget silently gives the cheaper-to-simulate
  hand more gradient steps.

## License

MIT, inheriting from simtoolreal.
