"""Deterministic policy playback for evaluation.

Ported from simtoolreal ``deployment/rl_player.py``. Wraps an rl_games
``PpoPlayerContinuous`` restored from a training checkpoint.

One behavioral change from the source: the SAPG exploration-coefficient column
appended to every observation is now an explicit constructor argument rather
than a hardcoded ``50.0``.

Why that matters here. SAPG-trained policies condition on a per-block
exploration coefficient carried as a trailing observation column, so playback
must append it; PPO-trained policies have no such column and appending one
shifts the whole observation vector by a slot, silently producing garbage
actions with no error. Since this project compares checkpoints across hands
(and potentially across algorithms), an unconditional append would corrupt the
comparison. ``sapg_expl_coef`` must therefore be passed explicitly: ``50.0``
reproduces simtoolreal's SAPG behavior, ``None`` is the correct value for a
plain PPO checkpoint.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from gym import spaces  # rl_games speaks old-gym spaces, not gymnasium
from rl_games.torch_runner import Runner, players

import coevolution.networks  # noqa: F401  registers the joint_transformer net
from coevolution.eval.rl_player_utils import read_cfg


def assert_equals(a, b) -> None:
    assert a == b, f"{a} != {b}"


class RlPlayer:
    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        config_path: str,
        checkpoint_path: Optional[str],
        device: str,
        sapg_expl_coef: Optional[float],
        num_envs: int = 1,
    ) -> None:
        """
        Args:
            sapg_expl_coef: Value for the trailing exploration-coefficient
                observation column that SAPG-trained policies expect. Pass
                ``50.0`` for a SAPG checkpoint (matches simtoolreal), or
                ``None`` for a plain PPO checkpoint (no column appended).
                Required — there is no safe default, and getting it wrong
                fails silently.
        """
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.device = device
        self.sapg_expl_coef = sapg_expl_coef

        # rl_games reads these off the "env" it is handed (which is self).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_observations,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_actions,), dtype=np.float32
        )
        self.num_envs = num_envs
        self.set_env_state = lambda *args, **kwargs: None

        self.cfg = read_cfg(config_path=config_path, device=self.device)
        self.player = self.create_rl_player(checkpoint_path=checkpoint_path)

    def create_rl_player(
        self, checkpoint_path: Optional[str]
    ) -> players.PpoPlayerContinuous:
        from rl_games.common import env_configurations

        env_configurations.register(
            "rlgpu", {"env_creator": lambda **kwargs: self, "vecenv_type": "RLGPU"}
        )

        config = self.cfg["train"]

        if self.device == "cpu":
            try:
                config["params"]["config"]["player"]["device_name"] = "cpu"
            except KeyError:
                config["params"]["config"]["player"] = {"device_name": "cpu"}
            config["params"]["config"]["device"] = "cpu"

        if checkpoint_path is not None:
            # rl_games' PpoPlayerContinuous.restore is `if os.path.exists(fn):`
            # -- a path that does not exist is skipped in silence and the player
            # keeps its RANDOM initialisation. Every downstream number then looks
            # like a policy that cannot do the task, which is indistinguishable
            # from a real result and has already cost one eval run. Fail here.
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"checkpoint not found: {checkpoint_path}. rl_games would "
                    "silently run with random weights, so this is fatal.")
            config["load_path"] = checkpoint_path
        runner = Runner()
        runner.load(config)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        player = runner.create_player()
        player.init_rnn()
        player.has_batch_dimension = True
        if checkpoint_path is not None:
            player.restore(checkpoint_path)
        return player

    def get_normalized_action(
        self, obs: torch.Tensor, deterministic_actions: bool = True
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        assert_equals(obs.shape, (batch_size, self.num_observations))

        if self.sapg_expl_coef is not None:
            coef = torch.full(
                (batch_size, 1), self.sapg_expl_coef, device=self.device, dtype=obs.dtype
            )
            obs = torch.cat([obs, coef], dim=1)

        normalized_action = self.player.get_action(
            obs=obs, is_deterministic=deterministic_actions
        )

        normalized_action = normalized_action.reshape(-1, self.num_actions)
        assert_equals(normalized_action.shape, (batch_size, self.num_actions))
        return normalized_action

    def get_action_and_value(
        self, obs: torch.Tensor, deterministic_actions: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Action plus the critic's V(s), from ONE forward pass.

        rl_games' get_action computes ``values`` and throws them away, so
        recording the value function during a rollout is free -- no second
        forward, no extra sim. This mirrors PpoPlayerContinuous.get_action
        exactly, INCLUDING advancing self.states, because the actor's recurrent
        state must evolve identically to a normal rollout or the trajectory
        being scored is not the one the policy would have produced.

        What V means here: the critic was trained on the TRAINING reward, which
        carries two morphology-dependent shaping terms -- fingertip approach
        sums over fingertips, and the hand action penalty sums over hand DoF
        (docs/proposal_codesign.md 1 measures these at ~20% and ~27%). So V is
        NOT a clean stand-in for goals achieved: it systematically favours more
        fingers and more joints for reasons that are about reward shaping rather
        than the task. Use it as a predictor to be validated, never as a fitness
        to select on, until that bias is measured.
        """
        batch_size = obs.shape[0]
        assert_equals(obs.shape, (batch_size, self.num_observations))
        if self.sapg_expl_coef is not None:
            coef = torch.full(
                (batch_size, 1), self.sapg_expl_coef, device=self.device, dtype=obs.dtype
            )
            obs = torch.cat([obs, coef], dim=1)

        p = self.player
        x = p._preproc_obs(obs)
        input_dict = {"is_train": False, "prev_actions": None,
                      "obs": x, "rnn_states": p.states}
        with torch.no_grad():
            res = p.model(input_dict)
        p.states = res["rnn_states"]
        action = res["mus"] if deterministic_actions else res["actions"]
        action = action.reshape(-1, self.num_actions)
        assert_equals(action.shape, (batch_size, self.num_actions))
        return action, res["values"].reshape(-1)

    def reset(self) -> None:
        self.player.reset()


__all__ = ["RlPlayer"]
