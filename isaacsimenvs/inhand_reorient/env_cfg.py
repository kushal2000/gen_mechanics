"""Config for in-hand reorientation: the pose-reaching config, retuned.

Only task-level fields move. Decimation, the sim/PhysX settings and every
domain-randomization range are inherited untouched -- they are tuned for this
hardware and this task family, and changing them alongside the task would make
any comparison against the pose-reaching runs unreadable.

What changes, and why:

  hand_only       the hand mounts on a fixed stub link, no arm DOF at all
  env_spacing     the arm's reach is what forced 1.2; without it the envs pack
  object_in_hand  the object starts in the palm instead of on a table
  orientation_only_goal  keypoints are compared about their own centres, so the
                  metric is a pure rotation residual (still metres)
  the lift terms  zeroed: there is no approach-and-lift phase to shape
  drop_distance_m replaces the absolute floor test with a palm-relative one
"""
from __future__ import annotations

from isaaclab.utils import configclass

from isaacsimenvs.pose_reaching_6d.env_cfg import PoseReachEnvCfg


@configclass
class InHandReorientEnvCfg(PoseReachEnvCfg):
    """Hand-only in-hand reorientation of a single cube."""

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        # --- hand-only mount -------------------------------------------------
        # Selected by a "handonly:" prefix on assets.robot_spec, NOT set here:
        # the agent YAML interpolates the network's spec from that same string,
        # so a separate flag would let the policy and the articulation disagree
        # about the joint count. See hand_sampler.robot_spec.HANDONLY_PREFIX.
        # No arm reach to clear, so envs pack far tighter.
        self.scene.env_spacing = 0.6

        # --- the task --------------------------------------------------------
        # The object starts in the palm, and the goal is an orientation only.
        self.reset.object_in_hand = True
        self.obs.orientation_only_goal = True
        # Dropped means "left the hand", not "fell below a floor height".
        self.termination.drop_distance_m = 0.3

        # --- reward: keep the progress term, retire the approach phase --------
        # keypoint_reward and reach_goal_bonus keep their pose-reaching values
        # so returns stay on a comparable scale. There is no lift to shape.
        self.reward.lifting_rew_scale = 0.0
        self.reward.lifting_bonus = 0.0
        self.reward.distance_delta_rew_scale = 0.0
