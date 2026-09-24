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
  the cube        one 45 mm object, replacing the 24 curated tools
  tolerance       FIXED at 0.1 rad, no curriculum (equal bounds make it inert)
  wrench DR       off: those impulses fired only when lifted, which is now always

These are mirrored in coevolution/cfg/task/InHandReorient.yaml, which is what
a run actually loads (hydra applies it on top of this class). Kept here too so
that constructing the config directly -- an eval script, a test -- gives the
task rather than a silently-pose-reaching hybrid.
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

        # --- the object: one 45 mm cube --------------------------------------
        self.assets.object_pool = "cube1"
        self.assets.handle_head_types = ("cube",)
        self.assets.object_assignment = "env_modulo"
        self.assets.object_friction = 1.0        # a cube on a flat palm needs grip
        self.reward.object_base_size = 0.045     # == the edge, so object_scale is (1,1,1)
        self.reward.fixed_size = (0.045, 0.045, 0.045)

        # --- reward: keep the progress term, retire the approach phase --------
        # There is no lift to shape. keypoint_rew_scale is raised because the
        # centred metric spans only 0..117 mm where the transport task's spanned
        # metres -- see the task YAML for the arithmetic.
        self.reward.lifting_rew_scale = 0.0
        self.reward.lifting_bonus = 0.0
        self.reward.distance_delta_rew_scale = 0.0
        self.reward.keypoint_rew_scale = 2000.0
        # Measured at 0.003: velocity penalty ~-28/episode against +8 of
        # keypoint progress, so the policy froze. See the task YAML.
        self.reward.hand_actions_penalty_scale = 0.0003

        # --- goals: uniform SO(3); position is not scored, only drawn ---------
        self.reset.goal_sampling_type = "absolute"
        self.reset.target_volume_mins = (0.0, 0.0, 0.65)
        self.reset.target_volume_maxs = (0.0, 0.0, 0.65)
        # The table surface is table_reset_z + 0.15 and the palm is at z=0.50,
        # so the default 0.38 would put it through the hand.
        self.reset.table_reset_z = -1.0

        # --- tolerance: FIXED, no curriculum ---------------------------------
        # Equal bounds make update_tolerance_curriculum inert by construction:
        # it clamps into [target, success], which is a single point. 0.0039 is
        # 0.1 rad (5.7 deg) on a 45 mm cube -- IsaacLab's threshold. Successes
        # may stay at 0 for a long time; the progress reward is dense regardless,
        # so designs are still ranked while the success rate stays honest.
        self.termination.success_tolerance = 0.0039
        self.termination.target_success_tolerance = 0.0039

        # --- wrench DR off: force_only_when_lifted is now always true ---------
        self.domain_randomization.force_scale = 0.0
        self.domain_randomization.torque_scale = 0.0
