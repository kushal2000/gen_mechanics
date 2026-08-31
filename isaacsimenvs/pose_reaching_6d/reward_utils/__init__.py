"""Reward terms, episode termination, and the success-tolerance curriculum.

Grouped because they are one feedback loop: the reward measures progress toward
a goal, termination decides when the goal counts as reached, and the curriculum
tightens the tolerance that decision uses.
"""

from .curriculum import (  # noqa: F401
    CURRICULUM_STATE_VERSION, get_curriculum_state, initial_success_tolerance,
    set_curriculum_state,
)
from .rewards import (  # noqa: F401
    action_penalty, compute_rewards, distance_delta_reward, keypoint_reward,
    lifting_reward, reach_goal_bonus, update_near_goal_steps,
)
from .termination import compute_terminations, update_tolerance_curriculum  # noqa: F401
