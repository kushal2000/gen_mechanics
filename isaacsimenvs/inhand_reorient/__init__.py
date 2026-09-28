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
        # Population training (grammar hands via env.assets.hand_population):
        # no implicit SAPG entropy bonus and a bounded policy log-std (I41).
        "rl_games_sapg_pop_cfg_entry_point": str(_CFG_DIR / "train" / "InHandReorientPopSAPG.yaml"),
    },
)

# Registers the `inhand_actor_critic` rl_games network the population config
# uses. rl_games is present wherever training or evaluation can run; a bare
# CPU tool environment without it simply skips the registration.
try:
    from . import policy_network  # noqa: F401
except ModuleNotFoundError as _exc:  # pragma: no cover - depends on the environment
    if _exc.name is None or not _exc.name.startswith("rl_games"):
        raise


def __getattr__(name: str) -> Any:
    if name == "InHandReorientEnv":
        from .env import InHandReorientEnv

        return InHandReorientEnv
    if name == "InHandReorientEnvCfg":
        from .env_cfg import InHandReorientEnvCfg

        return InHandReorientEnvCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
