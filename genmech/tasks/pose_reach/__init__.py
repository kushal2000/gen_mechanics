"""PoseReach task registration.

Registers ``GenMech-PoseReach-Direct-v0`` with the gymnasium registry
for the DirectRLEnv training path.

Entry points:
- ``env_cfg_entry_point``           → PoseReachEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → cfg/task/PoseReach.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/PoseReachPPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/PoseReachSAPG.yaml (default)
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .env import PoseReachEnv
from .env_cfg import PoseReachEnvCfg

__all__ = ["PoseReachEnv", "PoseReachEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="GenMech-PoseReach-Direct-v0",
    entry_point="genmech.tasks.pose_reach.env:PoseReachEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "genmech.tasks.pose_reach.env_cfg:PoseReachEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "PoseReach.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PoseReachPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PoseReachSAPG.yaml"),
    },
)
