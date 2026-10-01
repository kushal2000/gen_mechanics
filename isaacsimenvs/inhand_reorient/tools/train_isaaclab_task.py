r"""Train a stock Isaac Lab task (unmodified) through ``coevolution/train.py``.

Used to get the reference learning curve for NVIDIA's in-hand cube
reorientation benchmark, ``Isaac-Repose-Cube-Allegro-Direct-v0``, with the
same rl_games, observers and run-dir layout as our own tasks:

    timeout -k 30 4200 .venv_isaacsim/bin/python \
        isaacsimenvs/inhand_reorient/tools/train_isaaclab_task.py \
        --task Isaac-Repose-Cube-Allegro-Direct-v0 --headless \
        agent.params.config.max_epochs=100000

``coevolution/train.py`` already registers every Isaac Lab task: its
``coevolution.utils.hydra_utils`` import pulls in ``isaaclab_tasks``, whose
package ``__init__`` registers all of them, after AppLauncher has booted Kit.
The task's own ``rl_games_cfg_entry_point`` YAML is train.py's default
``--agent``.

The only addition here is logging. The stock env logs just
``consecutive_successes``; this script wraps the instance's ``_get_rewards``
(the env's code is untouched) to publish, in the shapes
``EnvStatsAlgoObserver`` reads, the same per-episode metrics our in-hand env
logs: ``episode_final/successes`` (goals reached per finished episode, read
before ``_reset_idx`` zeroes the counter), ``episode_final/done_drop`` and
``episode_final/done_timeout``, and ``rot_error_mean``/``rot_error_median``
(radians, against the goal in force before this step's goal resample).
"""

from __future__ import annotations

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def instrument_inhand_manipulation_env(env) -> bool:
    """Add our in-hand metrics to an Isaac Lab ``InHandManipulationEnv``.

    Returns False (and changes nothing) for any other env class.
    """
    try:
        from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import (
            InHandManipulationEnv,
            rotation_distance,
        )
    except ImportError:
        return False
    if not isinstance(env, InHandManipulationEnv):
        return False

    original_get_rewards = env._get_rewards

    def _get_rewards_with_metrics():
        # Rotation error against the goal in force this step; the original
        # call below resamples goals for the envs that reached theirs.
        rot_dist = rotation_distance(env.object_rot, env.goal_rot).abs()
        reward = original_get_rewards()
        env.extras["episode_final"] = {
            "successes": env.successes.clone(),
            "done_drop": env.reset_terminated.float(),
            "done_timeout": env.reset_time_outs.float(),
            "rot_error_final": rot_dist,
        }
        # 0-dim tensors: the observer converts them only when it prints.
        env.extras["rot_error_mean"] = rot_dist.mean()
        env.extras["rot_error_median"] = rot_dist.median()
        return reward

    env._get_rewards = _get_rewards_with_metrics
    return True


def main() -> None:
    import gymnasium

    original_make = gymnasium.make

    def make_and_instrument(*args, **kwargs):
        env = original_make(*args, **kwargs)
        if instrument_inhand_manipulation_env(env.unwrapped):
            print("[train_isaaclab_task] in-hand metrics attached", flush=True)
        return env

    # train.py looks up gymnasium.make at call time, after AppLauncher.
    gymnasium.make = make_and_instrument

    from coevolution.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
