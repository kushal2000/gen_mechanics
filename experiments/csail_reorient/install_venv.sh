#!/usr/bin/env bash
# Build .venv_isaacsim on the CSAIL cluster (run as a --tier cpu job, never on the login node).
# Same pinned recipe as README.md "Install"; uv and its caches come from the cluster env.sh
# (sourced by every `cluster submit` job), so Python 3.11 and wheels land on the NFS volume.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv_isaacsim/bin/python
[ -x "$PY" ] || uv venv .venv_isaacsim --python 3.11
uv pip install --python $PY "torch==2.7.0" "torchvision==0.22.0" "torchaudio==2.7.0" --index-url https://download.pytorch.org/whl/cu126
uv pip install --python $PY -e ./third_party/rl_games/
uv pip install --python $PY omegaconf hydra-core "gym==0.23.1" scipy "numpy==1.26.0" yourdfpy viser requests tqdm tyro "imageio[ffmpeg]" wandb termcolor trimesh pandas matplotlib tensorboard
uv pip install --python $PY "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
uv pip install --python $PY coacd "typing_extensions>=4.13"
uv pip install --python $PY -e . --no-deps
uv pip install --python $PY pytest pytest-timeout
$PY -c "import torch, rl_games, hand_sampler, pathlib, importlib.metadata as md; print('torch', torch.__version__); print('rl_games', pathlib.Path(rl_games.__file__).resolve()); print('isaaclab', md.version('isaaclab'), 'isaacsim', md.version('isaacsim'))"
echo INSTALL_DONE
