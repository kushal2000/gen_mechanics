"""In-hand olympics: several manipulation tasks on one object, one hand on a 6-DOF dummy wrist above a table.

Planned: a task sampled per env at reset and given to the policy as a one-hot; first tasks pen spinning on the
table and pen pickup + holding against pushes. Today it is a skeleton:

    env_cfg.py   InHandOlympicsEnvCfg: hand spec, wrist joint ranges and drives, table
    wrist.py     the hand's URDF on the dummy wrist, palm down (Kit-free)
    scene.py     robot articulation + kinematic table + ground + light
    env.py       InHandOlympicsEnv: joint-target actions, placeholder obs, zero reward, timeouts only
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
