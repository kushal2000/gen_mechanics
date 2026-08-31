"""DirectRLEnv for the 6D pose-reaching task, one hand or many.

One env class, not two. Which mode it runs is a property of the config, not of
the class: set ``assets.robot_population_seed`` or ``assets.robot_population_path``
and every env gets its own design, sharing one articulation view; leave them
unset and ``assets.robot_spec`` names a single hand for every env.

The env owns Isaac Lab hook wiring and state buffers. Task math lives in the
utility modules called from each hook, and everything hardware-specific comes
from ``self.robot_spec`` (see isaacsimenvs/pose_reaching_6d/scene_utils/robots/), so swapping hands is a config
change rather than an edit here.
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from isaacsimenvs.pose_reaching_6d.scene_utils.robots import get_robot_spec

from .env_cfg import PoseReachEnvCfg
from .obs_utils import apply_action_pipeline, apply_wrench_dr
from .reset_utils import log_step_metrics
from .obs_utils import (
    build_observations,
    compute_intermediate_values,
    compute_obs_dim,
)
from .reset_utils import allocate_state_buffers, reset_env_state
from .reward_utils import compute_rewards
from .obs_utils import DESCRIPTOR_DIM, describe_layout, population_descriptors
from .scene_utils import (
    _ensure_robot_population,
    _verify_robot_design_assignment,
    apply_physx_material_properties,
    setup_scene,
)
from .reward_utils import compute_terminations, update_tolerance_curriculum


__all__ = ["PoseReachEnv", "PoseReachEnvCfg"]


class PoseReachEnv(DirectRLEnv):
    cfg: PoseReachEnvCfg

    def __init__(
        self, cfg: PoseReachEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        # Resolve the robot BEFORE super().__init__: DirectRLEnv and rl_games
        # read action_space / observation_space off the configclass, and all
        # three are derived from the spec. _setup_scene also needs self.robot_spec.
        spec = self._resolve_spec(cfg)
        self.robot_spec = spec

        # action_space defaults to 0 ("derive"). A non-zero value is a caller
        # assertion about the robot; disagreeing with the spec means a stale
        # config, which would otherwise silently truncate the action vector.
        if cfg.action_space not in (0, spec.num_joints):
            raise ValueError(
                f"cfg.action_space={cfg.action_space} disagrees with robot_spec "
                f"{spec.name!r} ({spec.num_joints} joints). Leave it 0 to derive."
            )
        cfg.action_space = spec.num_joints

        # Obs/state widths follow the configured field lists and the hand's
        # joint and fingertip counts.
        cfg.observation_space = compute_obs_dim(cfg.obs.obs_list, spec)
        cfg.state_space = compute_obs_dim(cfg.obs.state_list, spec)

        super().__init__(cfg, render_mode, **kwargs)  # runs _setup_scene
        apply_physx_material_properties(self)
        allocate_state_buffers(self)
        self._post_init_hook()

    def _resolve_spec(self, cfg: PoseReachEnvCfg):
        """The RobotSpec defining the action and observation layout.

        With a population, the layout comes from the population rather than
        ``robot_spec``: every design shares the joint template, gain tables,
        home pose and fingertip-slot count, so any member defines it for all of
        them. Resolved before ``super().__init__`` because ``_setup_scene`` needs
        the population and the spaces are derived from it.
        """
        population = _ensure_robot_population(self, cfg.assets)
        if population is None:
            return get_robot_spec(cfg.assets.robot_spec)

        # The morphology descriptor is REQUIRED with a population, so enforce it
        # rather than trusting the config to carry it. The task YAML overlay
        # lists both field lists explicitly and is applied AFTER the configclass
        # defaults, so it silently dropped `morphology` once. A smoke run caught
        # it only because the observation came out 186 wide instead of 329;
        # every other signal, including the population resolving and the design
        # assignment verifying, looked correct.
        include = bool(getattr(cfg, "include_morphology_obs", True))
        for field_name in ("obs_list", "state_list"):
            current = tuple(getattr(cfg.obs, field_name))
            if include and "morphology" not in current:
                setattr(cfg.obs, field_name, current + ("morphology",))
            elif not include and "morphology" in current:
                setattr(cfg.obs, field_name,
                        tuple(f for f in current if f != "morphology"))
        if not include:
            # Loud, because every other signal in the run looks normal: the
            # population resolves, the assignment verifies, the curves have the
            # same shape. Only the observation width differs.
            print("[pose_reach] ABLATION: morphology descriptor REMOVED from "
                  "obs_list and state_list. The policy cannot distinguish the "
                  f"{len(population.specs)} designs it is driving.",
                  flush=True)
        return population.specs[0]

    def _post_init_hook(self) -> None:
        """Runs after the scene, materials and state buffers exist."""
        if self._robot_population_specs is None:
            return
        # Confirm against the LIVE SIM that env i holds the design its
        # observation buffers were built for. The identical assumption about the
        # object pool, left unchecked, gave 510 of 512 envs the wrong asset and
        # cost 5.07 -> 3.00 goals/episode while every asset-level comparison
        # passed.
        _verify_robot_design_assignment(self, self._robot_population_specs)
        self._build_morphology_obs()

    def _build_morphology_obs(self) -> None:
        """Per-env morphology descriptor, computed once and indexed.

        Reads the hands the population already loaded rather than re-parsing the
        manifest, so the descriptor and the specs cannot come from two reads of
        a file that changed in between.
        """
        hands = self._robot_population.hands
        table = torch.as_tensor(
            population_descriptors(hands), device=self.device, dtype=torch.float32
        )  # (k, D)
        # Indexed by the SAME design tensor the scene build recorded, so the
        # descriptor and the robot in the env cannot describe different designs.
        self._morphology_per_env = table[self._robot_design_index_per_env]  # (N, D)

        if self.cfg.log_morphology_layout:
            print(f"[pose_reach] morphology descriptor {DESCRIPTOR_DIM} dims, "
                  f"{len(hands)} designs -> {tuple(self._morphology_per_env.shape)}")
            print(describe_layout())

    # --- checkpointable env state -------------------------------------------
    # rl_games saves whatever vec_env.get_env_state() returns into the
    # checkpoint and hands it back on resume. Nothing implemented it, so
    # env_state was None in every checkpoint and a resumed run silently
    # restarted its curriculum -- an easier task on a different reward scale
    # than the checkpoint was written under, since the tolerance also scales the
    # keypoint reward. The state is defined HERE, on the env that owns it,
    # rather than in the rl_games adapter that merely ferries it.

    def get_curriculum_state(self) -> dict:
        """Curriculum state for the checkpoint. See utils/curriculum.py."""
        from .reward_utils import get_curriculum_state
        return get_curriculum_state(self)

    def set_curriculum_state(self, state: dict | None) -> None:
        """Restore curriculum state from a checkpoint."""
        from .reward_utils import set_curriculum_state
        set_curriculum_state(self, state)

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        reset_env_state(
            self,
            torch.as_tensor(env_ids, device=self.device, dtype=torch.long),
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        apply_action_pipeline(self, actions)
        apply_wrench_dr(self)

    def _apply_action(self) -> None:
        # Called decimation times per policy step; idempotent.
        self.robot.set_joint_position_target(self._cur_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        compute_intermediate_values(self)
        return compute_terminations(self)

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        log_step_metrics(self)
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return build_observations(self)

