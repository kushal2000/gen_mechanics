"""A ladder from IsaacLab's in-hand task toward ours, one delta per rung.

Each class changes exactly ONE thing from the class above it. The agent config stays
IsaacLab's at every rung (see this package's ``__init__``), so a rung that stops
learning is a fact about the environment change rather than about a different learner.

Why a ladder and not a port: our task differs from the reference in a dozen ways at
once -- decimation, episode length, reward shape, success threshold, object placement,
observation, hand -- and it does not learn. A single diff cannot say which difference
matters. Changing one at a time can.

The rungs are ordered cheapest-to-suspect first: timing before reward, reward before
geometry. Where a delta cannot be expressed as a config change on their cfg it stops
being a rung and becomes a note in the README; that boundary is itself informative,
because it marks the differences that are structural rather than parametric.

Read the README in this folder for the measured reference values and what each rung is
expected to cost.
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.allegro_env_cfg import (
    AllegroCubeEnvCfg,
)


@configclass
class InHandIsaacLabReferenceCfg(AllegroCubeEnvCfg):
    """Rung 0: the reference, UNCHANGED.

    Present as a rung of its own rather than pointing at their task id, so that every
    rung including the baseline is launched by the same script with the same agent
    config. If rung 0 does not learn here, nothing below it is worth reading and the
    problem is our install or our launcher, not our task.

    Verified in this install: 16 joints, 21 bodies, 72-d observation, 16-d action,
    decimation 4, dt 1/120, episode 20 s.
    """


@configclass
class InHandIsaacLabDecimationCfg(InHandIsaacLabReferenceCfg):
    """Rung 1: our control rate. decimation 4 -> 2, so 120 Hz control, not 60 Hz.

    Cheap to suspect and cheap to test. It doubles the policy's decisions per second
    and halves the torque applied per decision, and it is the difference most likely to
    be invisible in a reward curve while changing what the policy can do.
    """

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        # Their cfg ties the render interval to decimation; keep that relationship
        # rather than leaving it at the parent's value.
        self.sim.render_interval = self.decimation


@configclass
class InHandIsaacLabEpisodeCfg(InHandIsaacLabDecimationCfg):
    """Rung 2: our episode budget. 20 s -> 10 s per goal.

    Both theirs and ours reset the clock on a goal hit, so this is time to reach EACH
    goal, not the whole episode. Halving it halves the exploration a policy gets per
    goal before truncation.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0


@configclass
class InHandIsaacLabToleranceCfg(InHandIsaacLabEpisodeCfg):
    """Rung 3: our success threshold, as an angle.

    Theirs is 0.1 rad on the orientation command's success term. Ours is set in degrees
    and converted; 20 degrees is what the current runs use, and the 5-vs-20 comparison
    on our own task showed 66x more goals per episode at 20. So this rung is expected
    to make the task EASIER, not harder -- which is the point of putting it on the
    ladder: if the reference's learning collapses anyway, the threshold is not what our
    task's difficulty rests on.
    """

    SUCCESS_THRESHOLD_DEG = 20.0

    def __post_init__(self):
        super().__post_init__()
        threshold = math.radians(self.SUCCESS_THRESHOLD_DEG)
        # The command term owns the threshold in their formulation, and the reward and
        # the termination both read it from there, so setting it in one place is enough.
        self.commands.object_pose.orientation_success_threshold = threshold


__all__ = [
    "InHandIsaacLabDecimationCfg",
    "InHandIsaacLabEpisodeCfg",
    "InHandIsaacLabReferenceCfg",
    "InHandIsaacLabToleranceCfg",
]
