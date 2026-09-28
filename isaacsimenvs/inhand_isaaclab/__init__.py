"""IsaacLab's own in-hand reorientation task, unmodified, plus a ladder toward ours.

WHY THIS EXISTS. ``isaacsimenvs/inhand_reorient`` is our task: our reward, our
placement, our observation, our hands. It has never scored a goal reliably, and when a
task does not learn there is no way to tell a broken environment from a hard one
without a reference that does learn. IsaacLab ships one for exactly this problem --
``Isaac-Repose-Cube-Allegro-v0`` -- and it is installed here already.

So: start from the reference, confirm it trains, then change ONE thing at a time toward
our configuration. Whichever step stops learning is the answer.

WHAT IS AND IS NOT CLONED. Nothing is reimplemented. Importing this package imports
``isaaclab_tasks``, which registers the reference task; it is theirs, running their
manager-based env, their reward terms, their observation and their rl_games config.
Verified in this install: 16 joints, 21 bodies, 72-d observation, 16-d action,
decimation 4, dt 1/120, episode 20 s, and its Allegro USD resolves from the Isaac cloud
asset root over HTTPS.

THE LADDER is in ``env_cfg.py``: a chain of subclasses, each one delta from the
previous, ending at something close to ours. They register as
``GenMech-InHandIsaacLab-<step>-v0``. Read the README for the deltas and what each is
expected to cost.

HOW TO RUN. ``train.py`` in this folder, which uses IsaacLab's own rl_games wrapper and
their shipped agent config rather than our training stack -- our stack is built around
the joint-transformer and a RobotSpec, neither of which a manager-based env has. The
point of this folder is a reference that works, so it runs their way.
"""

from __future__ import annotations

REFERENCE_TASK = "Isaac-Repose-Cube-Allegro-v0"

# The ladder. Each rung is one delta further from the reference and closer to ours; the
# agent config stays IsaacLab's throughout, so a rung that stops learning is a fact
# about the environment change and not about a different learner.
LADDER: tuple[tuple[str, str], ...] = (
    ("step0-reference", "InHandIsaacLabReferenceCfg"),
    ("step1-decimation", "InHandIsaacLabDecimationCfg"),
    ("step2-episode", "InHandIsaacLabEpisodeCfg"),
    ("step3-tolerance", "InHandIsaacLabToleranceCfg"),
)

LADDER_TASKS: tuple[str, ...] = tuple(f"GenMech-InHandIsaacLab-{s}-v0" for s, _ in LADDER)

_AGENT = (
    "isaaclab_tasks.manager_based.manipulation.inhand.config.allegro_hand.agents"
    ":rl_games_ppo_cfg.yaml"
)


def register() -> tuple[str, ...]:
    """Register the ladder. Call AFTER AppLauncher, and only then.

    NOT done at import time. ``isaaclab_tasks`` is reachable only once Kit has booted,
    and importing this package is what ``python -m isaacsimenvs.inhand_isaaclab.train``
    does FIRST -- before its own AppLauncher line runs. Registering eagerly therefore
    fails with ModuleNotFoundError on the reference task's own package, which is a
    confusing way to learn about import order. ``inhand_reorient`` defers for the same
    reason.
    """
    import gymnasium as gym

    # Registers Isaac-Repose-Cube-Allegro-v0 and friends, by side effect. The reference
    # task is not redefined here; this folder only adds rungs beside it.
    import isaaclab_tasks  # noqa: F401

    from isaacsimenvs.inhand_isaaclab import env_cfg

    for step, cls in LADDER:
        task_id = f"GenMech-InHandIsaacLab-{step}-v0"
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{env_cfg.__name__}:{cls}",
                # Deliberately the REFERENCE agent config at every rung: the experiment
                # is which environment change breaks learning, so the learner is fixed.
                "rl_games_cfg_entry_point": _AGENT,
            },
        )
    return LADDER_TASKS


__all__ = ["LADDER", "LADDER_TASKS", "REFERENCE_TASK", "register"]
