"""Stationary-hand in-hand reorientation task.

Registers ``GenMech-InHandReorient-Direct-v0``. Follows the lazy-import
pattern ``pose_reaching_6d/__init__.py`` uses: registration is pure
gymnasium (safe at import time, no Kit needed), the env class resolves lazily
so this package stays importable -- for the URDF cutter and hand-spec tests
-- without booting Isaac Sim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym

__all__ = ["InHandReorientEnv", "InHandReorientEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "coevolution" / "cfg"

gym.register(
    id="GenMech-InHandReorient-Direct-v0",
    entry_point="isaacsimenvs.inhand_reorient.env:InHandReorientEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.inhand_reorient.env_cfg:InHandReorientEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "InHandReorient.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "InHandReorientSAPG.yaml"),
    },
)


def __getattr__(name: str) -> Any:
    if name == "InHandReorientEnv":
        from .env import InHandReorientEnv

        return InHandReorientEnv
    if name == "InHandReorientEnvCfg":
        from .env_cfg import InHandReorientEnvCfg

        return InHandReorientEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
