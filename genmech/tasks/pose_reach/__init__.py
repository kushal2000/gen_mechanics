"""PoseReach task registration.

Registers ``GenMech-PoseReach-Direct-v0`` with the gymnasium registry for the
DirectRLEnv training path.

Entry points:
- ``env_cfg_entry_point``           → PoseReachEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → cfg/task/PoseReach.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/PoseReachPPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/PoseReachSAPG.yaml (default)

``PoseReachEnv`` and ``PoseReachEnvCfg`` are re-exported **lazily**. Importing
them eagerly would pull in ``isaaclab.envs`` at package-import time, which only
resolves after ``AppLauncher`` has booted Kit — that would make every Kit-free
module under ``utils/`` (reward math, goal sampling, object generation) require
a Kit boot to import, and with it the offline tools in ``genmech.tools``.

All four registry entry points are strings, so gym resolves them at
``gym.make`` time; nothing here needs the env class up front.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

__all__ = ["PoseReachEnv", "PoseReachEnvCfg",
           "PoseReachMultiEnv", "PoseReachMultiEnvCfg"]

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

# Multi-embodiment variant: one distinct hand per env, one articulation view,
# one policy. Separate id (and separate env class) rather than a flag, so the
# single-embodiment path stays exactly what the pretrained policy and the parity
# test were built against.
gym.register(
    id="GenMech-PoseReachMulti-Direct-v0",
    entry_point="genmech.tasks.pose_reach.env_multi:PoseReachMultiEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            "genmech.tasks.pose_reach.env_multi_cfg:PoseReachMultiEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "PoseReach.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PoseReachPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PoseReachSAPG.yaml"),
    },
)


def __getattr__(name: str) -> Any:
    """Resolve the env classes on first access (requires a booted Kit)."""
    if name == "PoseReachEnv":
        from .env import PoseReachEnv

        return PoseReachEnv
    if name == "PoseReachEnvCfg":
        from .env_cfg import PoseReachEnvCfg

        return PoseReachEnvCfg
    if name == "PoseReachMultiEnv":
        from .env_multi import PoseReachMultiEnv

        return PoseReachMultiEnv
    if name == "PoseReachMultiEnvCfg":
        from .env_multi_cfg import PoseReachMultiEnvCfg

        return PoseReachMultiEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
