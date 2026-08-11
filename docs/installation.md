# Installation

`genmech` runs the pose-reaching task in Isaac Sim via Isaac Lab. It needs
**Python 3.11** (an Isaac Sim 5.x / Isaac Lab 2.3.x requirement) in its own venv at
`.venv_isaacsim/`.

## Prerequisites

- Python 3.11, NVIDIA driver >= 525.60, CUDA 12+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

## Install

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

# Register genmech itself
uv pip install --python $PY -e . --no-deps
```

Verify:

```bash
.venv_isaacsim/bin/python -c "
import torch, rl_games, genmech, pathlib
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('rl_games', pathlib.Path(rl_games.__file__).resolve())
print('genmech', pathlib.Path(genmech.__file__).resolve())
"
```

`rl_games` must resolve under `third_party/rl_games/`. The PyPI `rl-games` lacks
the SAPG additions this project trains with (`expl_coef_block_size`,
`use_others_experience`, and the `policy_idx` parse from the config `name`'s
`<int>_` prefix), and swapping it in fails in confusing ways rather than loudly.

**Keep the pins exact.** The validated set is `isaaclab==2.3.2.post1`,
`isaacsim==5.1.0.0`, `torch==2.7.0+cu126`, `numpy==1.26.0` — copied from
simtoolreal's working environment. Newer Isaac Lab releases change the
`DirectRLEnv` and `UrdfConverter` APIs that `genmech/tasks/pose_reach/` depends
on, and an unpinned `torch` resolves far past what the isaacsim wheels support.

## Running

```bash
export OMNI_KIT_ACCEPT_EULA=YES
# Point the Omniverse shader cache at local disk, not NFS
export OMNI_KIT_CACHE_PATH=/tmp/$USER/ov_cache && mkdir -p "$OMNI_KIT_CACHE_PATH"
```

Smoke tests — one Kit boot per process (~1-2 min each), run individually, **not**
under pytest:

```bash
.venv_isaacsim/bin/python tests/test_load_isaacsim.py
.venv_isaacsim/bin/python tests/test_gym_register.py
.venv_isaacsim/bin/python tests/test_env_smoke.py --num_envs 8 --num_assets_per_type 2 --steps 10
```

Training:

```bash
.venv_isaacsim/bin/python genmech/train.py \
  --task GenMech-PoseReach-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless
```

## Gotchas

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
- **Training runs must be epoch-bound, not walltime-bound.** See
  `docs/methodology.md` — a walltime budget silently gives the cheaper-to-simulate
  hand more gradient steps.
