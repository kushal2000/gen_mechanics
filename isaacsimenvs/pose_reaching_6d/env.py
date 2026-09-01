"""DirectRLEnv for 6D pose reaching, one hand or many.

Set ``assets.robot_population_seed`` or ``robot_population_path`` and every env
gets its own design sharing one articulation view; leave them unset and
``assets.robot_spec`` names one hand for all of them. Task math lives in the
modules called from each hook.
"""

from __future__ import annotations

import torch
from isaaclab.envs import DirectRLEnv

from .env_cfg import PoseReachEnvCfg
from .obs_utils import (
    DESCRIPTOR_DIM, apply_action_pipeline, apply_wrench_dr, build_observations,
    compute_intermediate_values, derive_spaces, describe_layout,
    force_morphology_field,
    population_descriptors,
)
from .reset_utils import allocate_state_buffers, log_step_metrics, reset_env_state
from .reward_utils import (
    compute_rewards, compute_terminations, update_tolerance_curriculum,
)
from .reward_utils import get_curriculum_state as _curriculum_state
from .reward_utils import set_curriculum_state as _restore_curriculum_state
from .scene_utils import (
    _ensure_robot_population, _verify_robot_design_assignment,
    apply_physx_material_properties, setup_scene,
)
from .scene_utils.robots import get_robot_spec

__all__ = ["PoseReachEnv", "PoseReachEnvCfg"]


class PoseReachEnv(DirectRLEnv):
    """Grasp a tool-like object off a table and drive it through SE(3) goals."""

    cfg: PoseReachEnvCfg

    def __init__(
        self, cfg: PoseReachEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        # cfg is IN/OUT: callers read the derived spaces back to size the policy.
        self.robot_spec = spec = self._resolve_spec(cfg)
        derive_spaces(cfg, spec)
        super().__init__(cfg, render_mode, **kwargs)   # runs _setup_scene
        apply_physx_material_properties(self)          # needs the built stage
        allocate_state_buffers(self)
        self._post_init_hook()                         # needs both of the above

    # --- construction --------------------------------------------------------

    def _resolve_spec(self, cfg: PoseReachEnvCfg):
        """RobotSpec defining the action and observation layout.

        A population's designs share one joint template, so any member defines
        it and ``robot_spec`` is ignored. _worker overrides this to inject one.
        """
        population = _ensure_robot_population(self, cfg.assets)
        if population is None:
            return get_robot_spec(cfg.assets.robot_spec)
        force_morphology_field(cfg, len(population.specs))
        return population.specs[0]

    def _post_init_hook(self) -> None:
        if self._robot_population is None:
            return
        # Against the live sim: unchecked, the same assumption put 510 of 512
        # envs on the wrong object.
        _verify_robot_design_assignment(self, self._robot_population.specs)
        self._build_morphology_obs()

    def _build_morphology_obs(self) -> None:
        """Per-env descriptor, from the hands the specs were built from."""
        hands = self._robot_population.hands
        table = torch.as_tensor(population_descriptors(hands),
                                device=self.device, dtype=torch.float32)
        self._morphology_per_env = table[self._robot_design_index_per_env]
        if self.cfg.log_morphology_layout:
            print(f"[pose_reach] descriptor {DESCRIPTOR_DIM} dims, {len(hands)} "
                  f"designs -> {tuple(self._morphology_per_env.shape)}")
            print(describe_layout())

    # --- checkpointable state ------------------------------------------------
    # Unimplemented, env_state was None in every checkpoint and resumed runs
    # silently restarted the curriculum, on a different reward scale.

    def get_curriculum_state(self) -> dict:
        return _curriculum_state(self)

    def set_curriculum_state(self, state: dict | None) -> None:
        _restore_curriculum_state(self, state)

    # --- Isaac Lab hooks -----------------------------------------------------

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        reset_env_state(
            self, torch.as_tensor(env_ids, device=self.device, dtype=torch.long))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        apply_action_pipeline(self, actions)
        apply_wrench_dr(self)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._cur_targets)  # decimation x / step

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
