"""In-hand reorientation: a fixed palm-up hand, no arm, one cube.

A sibling of GenMech-PoseReach-Direct-v0 that shares its env class outright.
Every difference between the two tasks is a config field, each defaulting to
the pose-reaching behaviour:

    assets.hand_only            mount on a fixed stub link, no arm DOF
    reset.object_in_hand        start holding the object, and latch _lifted_object
    obs.orientation_only_goal   compare keypoints about their own centres
    termination.drop_distance_m dropped = left the palm, not fell below a floor

So there is no env subclass. If a variant ever needs one of the duck-typed
hooks (_keypoint_success_tolerance_m, _curriculum_eligible_mask,
_curriculum_success_threshold, _wrench_dr_active_mask) it can subclass
PoseReachEnv then; an empty subclass now would only be a place for drift.
"""
from pathlib import Path

import gymnasium as gym

__all__ = ["InHandReorientEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "coevolution" / "cfg"

gym.register(
    id="GenMech-InHandReorient-Direct-v0",
    entry_point="isaacsimenvs.pose_reaching_6d.env:PoseReachEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.inhand_reorient.env_cfg:InHandReorientEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "InHandReorient.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PoseReachPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PoseReachSAPG.yaml"),
        "rl_games_sapg_nolstm_cfg_entry_point": str(
            _CFG_DIR / "train" / "PoseReachSAPGNoLSTM.yaml"
        ),
        "rl_games_joint_transformer_cfg_entry_point": str(
            _CFG_DIR / "train" / "PoseReachJointTransformerSAPG.yaml"
        ),
    },
)


def __getattr__(name):
    """Lazy re-export: importing the cfg pulls in Isaac Lab, which must not
    happen at registration time (Kit has to start first)."""
    if name == "InHandReorientEnvCfg":
        from .env_cfg import InHandReorientEnvCfg
        return InHandReorientEnvCfg
    raise AttributeError(name)
