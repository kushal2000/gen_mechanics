"""PoseReach task registration.

Registers ``GenMech-PoseReach-Direct-v0`` with the gymnasium registry for the
DirectRLEnv training path.

Entry points:
- ``env_cfg_entry_point``           → PoseReachEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → cfg/task/PoseReach.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/PoseReachPPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/PoseReachSAPG.yaml (default)
- ``rl_games_sapg_nolstm_cfg_entry_point`` → cfg/train/PoseReachSAPGNoLSTM.yaml
- ``rl_games_joint_transformer_cfg_entry_point``
  → cfg/train/PoseReachJointTransformerSAPG.yaml (decentralized per-joint policy)

``PoseReachEnv`` and ``PoseReachEnvCfg`` are re-exported **lazily**. Importing
them eagerly would pull in ``isaaclab.envs`` at package-import time, which only
resolves after ``AppLauncher`` has booted Kit — that would make every Kit-free
module under ``utils/`` (reward math, goal sampling, object generation) require
a Kit boot to import, and with it the offline tools in ``genmech.tools``.

All registry entry points are strings, so gym resolves them at
``gym.make`` time; nothing here needs the env class up front.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

__all__ = ["PoseReachEnv", "PoseReachEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "coevolution" / "cfg"

gym.register(
    id="GenMech-PoseReach-Direct-v0",
    entry_point="isaacsimenvs.pose_reaching_6d.env:PoseReachEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.pose_reaching_6d.env_cfg:PoseReachEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "PoseReach.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PoseReachPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PoseReachSAPG.yaml"),
        "rl_games_sapg_nolstm_cfg_entry_point": str(
            _CFG_DIR / "train" / "PoseReachSAPGNoLSTM.yaml"
        ),
        "rl_games_joint_transformer_cfg_entry_point": str(
            _CFG_DIR / "train" / "PoseReachJointTransformerSAPG.yaml"
        ),
    },
)

# The multi-embodiment id is an ALIAS, kept so archived submit scripts and the
# 24k run configs still resolve. There is one env class and one config now:
# whether a run drives one hand or 24,576 is decided by
# assets.robot_population_seed / robot_population_path, not by which id you ask
# for.
gym.register(
    id="GenMech-PoseReachMulti-Direct-v0",
    entry_point="isaacsimenvs.pose_reaching_6d.env:PoseReachEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.pose_reaching_6d.env_cfg:PoseReachEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "PoseReach.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PoseReachPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PoseReachSAPG.yaml"),
        "rl_games_sapg_nolstm_cfg_entry_point": str(
            _CFG_DIR / "train" / "PoseReachSAPGNoLSTM.yaml"
        ),
        "rl_games_joint_transformer_cfg_entry_point": str(
            _CFG_DIR / "train" / "PoseReachJointTransformerSAPG.yaml"
        ),
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
