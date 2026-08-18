"""PoseReach with a DIFFERENT hand in every parallel environment.

The single-embodiment env (``env.py``) trains one policy on one hand. This one
trains one policy across a population of morphologies simultaneously -- the
cross-embodied setting the co-design loop needs, where a design's score comes
from a policy that has actually adapted to it (docs/proposal_codesign.md §2).

It is a separate class, not a flag on ``PoseReachEnv``, for a blunt reason: the
single-embodiment env is what the pretrained policy, the SHARPA parity test and
the running training jobs depend on, and it should not carry branches for a mode
it never takes. The two share everything that is genuinely shared -- the task,
reward, reset, action pipeline and object pipeline all live in ``utils/`` and are
imported by both -- and differ only where they actually differ.

WHAT MAKES THIS POSSIBLE

  * Ghosting pads every generated design to the same 37-joint template, so an
    arbitrary number of morphologies costs ONE articulation view. PhysX reads
    ``shared_metatype.dof_count`` once per view, so designs with different joint
    counts could not share a scene at all.
  * Diversity is free per step: measured flat from k=8 to k=24,576 distinct
    designs at fixed env count (docs/multi_embodiment.md §2).

WHAT IT ADDS TO THE OBSERVATION

A morphology descriptor (``utils/morphology.py``), because proprioception plus
fingertip positions is not enough to control a hand you have never seen. The
policy commands per-joint targets; without the mount transforms and link
geometry it cannot know what those commands do, and joint_pos normalization
actively hides the abduction range -- 0.5 means 1 degree on one design and 12.5
on another. The descriptor is constant per env, so it is computed once and
indexed, not recomputed per step.

Fingertip positions stay in the observation exactly as before; the descriptor is
additive.
"""

from __future__ import annotations

import torch

from .env import PoseReachEnv
from .env_multi_cfg import PoseReachMultiEnvCfg
from .utils.morphology import DESCRIPTOR_DIM, describe_layout, population_descriptors
from .utils.scene_utils import (
    _resolve_robot_population,
    _verify_robot_design_assignment,
)


class PoseReachMultiEnv(PoseReachEnv):
    """One distinct hand per env, one articulation view, one policy."""

    cfg: PoseReachMultiEnvCfg

    def _resolve_spec(self, cfg: PoseReachMultiEnvCfg):
        """Take the layout from the population, not from ``robot_spec``.

        Every design in a population shares the joint template, gain tables,
        home pose and fingertip-slot count, so any member defines the action and
        observation layout for all of them. Honouring ``robot_spec`` here would
        be actively wrong: a fixed hand and a generated one do not share a joint
        set (29 vs 37), so it would build the spaces for a robot the scene does
        not contain.

        Resolved before ``super().__init__`` because ``_setup_scene`` needs the
        population and the spaces are derived from it.
        """
        self._robot_population_specs = _resolve_robot_population(cfg.assets)
        if self._robot_population_specs is None:
            raise ValueError(
                "PoseReachMultiEnv requires assets.robot_population_seed. "
                "For a single hand, use PoseReachEnv "
                "(GenMech-PoseReach-Direct-v0)."
            )

        # The morphology descriptor is REQUIRED here, so enforce it rather than
        # trusting the config to carry it.
        #
        # MultiObsCfg declares it, and that was not enough: the task YAML
        # overlay (genmech/cfg/task/PoseReach.yaml, shared with the
        # single-embodiment task) lists both field lists explicitly, and the
        # overlay is applied AFTER the configclass defaults -- so it silently
        # overwrote them and dropped `morphology`. A smoke run caught it only
        # because the observation came out 186 wide instead of 329; every other
        # signal, including the population resolving and the design assignment
        # verifying, looked correct.
        #
        # Without the descriptor this env trains a cross-embodied policy that
        # cannot tell its hands apart, which is the one thing it exists to do.
        # cfg.include_morphology_obs is the ONLY way to turn this off, and it
        # strips the field rather than merely declining to add it -- otherwise
        # MultiObsCfg's default, the YAML overlay, or
        # `obs.obs_list=${env.obs.state_list}` would each put it back.
        include = bool(getattr(cfg, "include_morphology_obs", True))
        for field in ("obs_list", "state_list"):
            current = tuple(getattr(cfg.obs, field))
            if include and "morphology" not in current:
                setattr(cfg.obs, field, current + ("morphology",))
            elif not include and "morphology" in current:
                setattr(cfg.obs, field,
                        tuple(f for f in current if f != "morphology"))
        if not include:
            # Loud, because every other signal in this run looks normal: the
            # population resolves, the design assignment verifies, the curves
            # have the same shape. Only the observation width differs.
            print("[multi] ABLATION: morphology descriptor REMOVED from "
                  "obs_list and state_list. The policy cannot distinguish the "
                  f"{len(self._robot_population_specs)} designs it is driving.",
                  flush=True)

        return self._robot_population_specs[0]

    def _post_init_hook(self) -> None:
        # Confirm against the LIVE SIM that env i holds the design its
        # observation buffers were built for. The identical assumption about the
        # object pool, left unchecked, gave 510 of 512 envs the wrong asset and
        # cost 5.07 -> 3.00 goals/episode while every asset-level comparison
        # passed (docs/multi_embodiment.md §4).
        _verify_robot_design_assignment(self, self._robot_population_specs)
        self._build_morphology_obs()

    def _build_morphology_obs(self) -> None:
        """Per-env morphology descriptor, computed once."""
        from genmech.robots.generated.population import load_population

        hands = load_population(int(self.cfg.assets.robot_population_seed))
        count = int(self.cfg.assets.robot_population_count or 0)
        if count:
            hands = hands[:count]
        if len(hands) != len(self._robot_population_specs):
            raise RuntimeError(
                f"population/spec mismatch: {len(hands)} hands, "
                f"{len(self._robot_population_specs)} specs")

        table = torch.as_tensor(
            population_descriptors(hands), device=self.device, dtype=torch.float32
        )  # (k, D)
        # Indexed by the SAME design tensor the scene build recorded, so the
        # descriptor and the robot in the env cannot describe different designs.
        self._morphology_per_env = table[self._robot_design_index_per_env]  # (N, D)

        if self.cfg.log_morphology_layout:
            print(f"[multi] morphology descriptor {DESCRIPTOR_DIM} dims, "
                  f"{len(hands)} designs -> {tuple(self._morphology_per_env.shape)}")
            print(describe_layout())


__all__ = ["PoseReachMultiEnv"]
