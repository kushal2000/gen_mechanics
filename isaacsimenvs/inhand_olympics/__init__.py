"""In-hand olympics: several manipulation tasks on one object, one hand on a 6-DOF dummy wrist above a table.

A task is sampled per env at reset and given to the policy as a one-hot; every task is a sequence of 6D goal
poses for the pen. Tasks: pen_spin (spin it flat on the table, in place) and pen_lift (pick it up flat and hold it
while it is pushed and twisted) pen_spin_air (pick it up and spin it in the air) and pen_flip_air (pick it up and flip it end over end).

    env_cfg.py   InHandOlympicsEnvCfg: hand, wrist, actions, table, pen, tasks, goals, rewards, terminations
    wrist.py     the hand's URDF on the dummy wrist, palm down (Kit-free)
    scene.py     robot articulation + kinematic table + pen (capsule) + ground + light
    env.py       InHandOlympicsEnv: actions, goals and success, rewards, terminations
    observations.py  in-hand reorientation's observation format (obs_list / state_list)
    goals.py     pen keypoints (4 along the long axis, tip to tail) and goal-pose helpers
    tasks/       one module per task: where the pen starts, how the next goal follows a hit one
    pose_viewer.py  the wandb pose viewer for this env (train.py --capture_viewer)
    tests/       viser_hand_table.py: drive every joint from sliders and watch the sim

Registered as GenMech-InHandOlympics-Direct-v0 with in-hand's rl_games configs.
"""
from pathlib import Path

import gymnasium as gym

_CFG_DIR = Path(__file__).resolve().parents[2] / "coevolution" / "cfg" / "train"

gym.register(
    id="GenMech-InHandOlympics-Direct-v0",
    entry_point="isaacsimenvs.inhand_olympics.env:InHandOlympicsEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.inhand_olympics.env_cfg:InHandOlympicsEnvCfg",
        "rl_games_cfg_entry_point": str(_CFG_DIR / "PoseReachPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "PoseReachSAPG.yaml"),
        "rl_games_sapg_nolstm_cfg_entry_point": str(_CFG_DIR / "PoseReachSAPGNoLSTM.yaml"),
    },
)
