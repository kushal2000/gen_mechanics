"""In-hand olympics: several manipulation tasks on one object, one hand on a 6-DOF dummy wrist above a table.

A task is sampled per env at reset and given to the policy as a one-hot; every task is a sequence of 6D goal
poses for the pen. Tasks: pen_spin (spin it flat on the table, in place). Planned: pen pickup + holding against
pushes.

    env_cfg.py   InHandOlympicsEnvCfg: hand, wrist, actions, table, pen, tasks, goals, rewards, terminations
    wrist.py     the hand's URDF on the dummy wrist, palm down (Kit-free)
    scene.py     robot articulation + kinematic table + pen (capsule) + ground + light
    env.py       InHandOlympicsEnv: actions, goals and success, rewards, terminations, observations
    goals.py     pen keypoints (tip, centre, tail) and goal-pose helpers
    tasks/       one module per task: where the pen starts, how the next goal follows a hit one
    tests/       viser_hand_table.py: drive every joint from sliders and watch the sim

Not imported by isaacsimenvs/__init__.py yet, so nothing here can break a training job's startup; import this
package to register the gym id.
"""
import gymnasium as gym

gym.register(
    id="GenMech-InHandOlympics-Direct-v0",
    entry_point="isaacsimenvs.inhand_olympics.env:InHandOlympicsEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "isaacsimenvs.inhand_olympics.env_cfg:InHandOlympicsEnvCfg"},
)
