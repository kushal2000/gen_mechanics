# gen_mechanics

**Does dexterous hand hardware generalize?**

A benchmark for comparing dexterous hands on a 6D pose-reaching task — grasp a
tool-like object off a table and drive it through a sequence of SE(3) goals —
under held-out object physics, object geometry, domain-randomization settings,
and goal distributions.

The question is not which hand scores highest on its training distribution, but
which hand's policy **retains** the most performance when the world shifts.

Built on the Isaac Sim path of
[simtoolreal](https://github.com/tylerlum/simtoolreal) (Kedia, Lum, Bohg, Liu).
Isaac Gym, ROS deployment, FoundationPose, and the retargeting baselines are not
carried over.

## Status

| Milestone | State |
|---|---|
| M0 — repo bootstrap, venv | in progress |
| M1 — SHARPA port parity vs simtoolreal | pending |
| M2 — `RobotSpec` registry (swappable hands) | pending |
| M3 — SHARPA training run | pending |
| M4 — generalization eval harness | pending |
| M5 — iiwa14 + Allegro robot | pending |
| M6/M7 — Allegro training, full sweep, analysis | pending |

## Layout

Three source packages, in dependency order. The arrow points the only way an
import is allowed to go:

```
hand_sampler  <--  isaacsimenvs  <--  coevolution
```

```
hand_sampler/        the generated-hand design space -- PURE PYTHON, no Isaac
  params.py            HandParams: the grammar, tiers, validity
  mutate.py            the five operators (palm, scale, mounting, num_joints +/-1)
  population.py        sample, gate and serialise a population
  synth_spec.py        hand params -> RobotSpec, mounted on the iiwa14
  urdf.py              the URDF backend; inertia.py, flexion.py, spec.py
  gates/               capsule.py (analytic) and mesh.py self-collision gates
  workspace.py         table + goal-volume geometry shared by the viewers
  *_viewer.py          design-space, grammar and mutation viser viewers
isaacsimenvs/        everything that touches Isaac Sim
  pose_reach/          the DirectRLEnv task; all task math in utils/
  authoring/           designs -> USD prims, without the URDF converter
  robots/              RobotSpec registry: one spec per (hand, arm)
coevolution/         searching over designs and control together
  train.py, cfg/       entry point + hydra task / train (PPO, SAPG) configs
  eval/                DR suites, checkpoint player, cross-body eval
  population/          population eval: 24,576 designs, one per env
  loop/                the training-free design-evolution loop + its viewer
  pose_viewer.py       the wandb interactive pose viewer
assets/urdf/         robot + table URDFs and meshes; generated hands land here
third_party/rl_games/  vendored SAPG fork (NOT the PyPI package)
experiments/         SLURM job scripts (+ old_experiments/ archive)
```

`hand_sampler` imports no simulator. That is what lets a design search run on
CPU: sampling, mutation, the geometry gates and value-function scoring all work
without booting Kit.

## Quick start

Installation is at the bottom of this file.

```bash
export OMNI_KIT_ACCEPT_EULA=YES
PY=.venv_isaacsim/bin/python

# Sample a population of hands -- no Isaac needed, runs on CPU
$PY -m hand_sampler.population --seed 0 --count 64

# Train
sbatch experiments/train.sub          # ROBOT=sharpa_iiwa14 SEED=0
```

## Methodology

What is held fixed to keep this a *hardware* comparison — identical arm, byte-identical reward, equal gradient
steps (not equal walltime), frozen held-out sets, retention rather than raw
reward — and the residual confounds that remain.

## Where this is going

The follow-on: instead of comparing two fixed hands, *search* over hand morphology jointly with
a morphology-conditioned policy. It covers the two target results (co-design as
a training curriculum; co-design for generalization and sim-to-real), the
Thompson-sampling outer loop. Heterogeneous morphology per environment was
measured to be feasible in this stack, and is what `isaacsimenvs/pose_reach/`
now does: 24,576 distinct hands share one articulation view.

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
`DirectRLEnv` and `UrdfConverter` APIs that `isaacsimenvs/pose_reach/` depends
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
