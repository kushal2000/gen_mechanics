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

```
genmech/
  robots/          RobotSpec registry — one spec per (hand, arm); adjacency maps
  tasks/pose_reach/  the DirectRLEnv task; all task math lives in utils/
  cfg/             hydra task + train (PPO / SAPG) configs
  eval/            generalization eval harness: suites, runner, sweep, aggregate
  tools/           URDF authoring, geometry calibration, reachability viewer
  viewer/          pose viewer + interactive viser viewer
  utils/           hydra / rl_games / wandb glue
assets/urdf/       robot + table + object URDFs and meshes
third_party/rl_games/  vendored SAPG fork (NOT the PyPI package)
tests/             standalone Kit-booting smoke and invariant tests
experiments/       SLURM job scripts
docs/              installation, methodology
```

## Quick start

See [docs/installation.md](docs/installation.md). Then:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
PY=.venv_isaacsim/bin/python

$PY tests/test_load_isaacsim.py
$PY tests/test_gym_register.py
$PY tests/test_env_smoke.py --num_envs 8 --num_assets_per_type 2 --steps 10

sbatch experiments/train.sub          # ROBOT=sharpa_iiwa14 SEED=0
```

## Methodology

[docs/methodology.md](docs/methodology.md) states what is held fixed to keep this
a *hardware* comparison — identical arm, byte-identical reward, equal gradient
steps (not equal walltime), frozen held-out sets, retention rather than raw
reward — and the residual confounds that remain.

## License

MIT, inheriting from simtoolreal.
