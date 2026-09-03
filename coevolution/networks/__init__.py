"""Custom rl_games networks.

Importing this package registers them. rl_games looks a network up by
``params.network.name`` in a module-level registry that is populated by
``model_builder.register_network``; nothing auto-imports it, so ``train.py``
and ``eval/rl_player.py`` import this package for the side effect.
"""

from __future__ import annotations

from rl_games.algos_torch import model_builder

from coevolution.networks.joint_transformer import (
    JointTransformerBuilder, JointTransformerNet,
)

model_builder.register_network("joint_transformer", JointTransformerBuilder)

__all__ = ["JointTransformerBuilder", "JointTransformerNet"]
