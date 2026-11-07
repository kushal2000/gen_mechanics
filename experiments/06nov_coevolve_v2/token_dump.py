"""Dump the joint tokens a policy sees for one hand: a 1-env scene built as the evals build it (the training run's
saved env config), reset with the table-safe reset off so every hand starts from the same zero-pose fingers, and
the observation's per-slot joint fields read exactly as observations.py computes them.

    .venv_isaacsim/bin/python experiments/06nov_coevolve_v2/token_dump.py --spec <registered name | .spec.json> \
        --train-run <training run dir> --out <file.json>

Per hand slot (the 6 x 5 finger-major grid + the palm slot): the real joint in it, enabled, joint_origin and
joint_link_vec (mm, canonical palm frame at the palm centre), joint_axis, joint_lower / joint_upper (deg) and the
normalised joint_pos; plus palm_extents (mm), the fingertips (mm, palm frame) and the palm frame itself.
A grammar spec runs in the multi:template: scene (as in training), a registered robot in multi:finger: (same grid).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRID = {"n_rigid": 6, "n_moving": 0, "n_slots": 5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(REPO))
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app  # noqa: F841

    import torch
    from isaaclab.utils.math import quat_mul
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.common.obs_geometry import (apply_local_offset, joint_link_geometry_obs, normalize_joint_pos,
                                                  quat_apply_broadcast, rotate_into)
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg

    full = OmegaConf.to_container(OmegaConf.load(next(Path(args.train_run).rglob(".hydra/config.yaml"))), resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    if args.spec.endswith(".spec.json") and "design" in json.loads(((REPO / args.spec) if not Path(args.spec).is_absolute() else Path(args.spec)).read_text()):
        mdir = Path(tempfile.mkdtemp())                 # the template loader needs a file named *manifest.json
        (mdir / "manifest.json").write_text(json.dumps({"generation": 0, "grid": GRID,
                                                        "hands": [{"id": "h", "spec": args.spec}]}))
        ref = f"multi:template:{mdir / 'manifest.json'}"
    else:
        ref = f"multi:finger:{args.spec}"
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = 1
    cfg.reset.stagger_initial_episodes = False
    cfg.reset.table_safe = False
    cfg.log.group_stats = False
    cfg.log.design_stats = False
    cfg.seed = 0
    env = InHandOlympicsEnv(cfg)
    env.reset()
    # zero-pose hand: every joint at 0 (the tokens' geometry, independent of the reset's random finger pose)
    jp = torch.zeros_like(env.robot.data.joint_pos)
    env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)

    origins = env.scene.env_origins
    palm = env.robot.data.body_state_w[:, env.palm_body]
    palm_pos_w, palm_rot_body = palm[:, 0:3], palm[:, 3:7]
    centre_w = apply_local_offset(palm_pos_w, palm_rot_body, env.palm_center_offset, (env.num_envs,))
    palm_rot = quat_mul(palm_rot_body, env.palm_frame_quat)
    ee_pos = centre_w - origins
    ones = torch.ones(env.num_envs, 1, device=env.device)
    _, joint_origins, _ = joint_link_geometry_obs(env, centre_w, palm_rot, origins, ones)
    link_rot = env.robot.data.body_state_w[:, env._joint_link_body_ids, 3:7]
    en = env.joint_enabled.unsqueeze(-1)
    axis = rotate_into(palm_rot, quat_apply_broadcast(link_rot, env._joint_axis_local)) * en
    lvec = rotate_into(palm_rot, quat_apply_broadcast(link_rot, env._joint_link_vec_local)) * en
    jorig = rotate_into(palm_rot, joint_origins - ee_pos.unsqueeze(1)) * en
    perm = env.perm_lab_to_canon
    hand = slice(6, None)
    jpos = normalize_joint_pos(env.robot.data.joint_pos[:, perm], env.joint_lower_canon, env.joint_upper_canon)[:, hand]
    tips = rotate_into(palm_rot, env.fingertips() - ee_pos.unsqueeze(1))

    hs = env.hand_set
    spec, lay = hs.specs[0], hs.layouts[0]
    name_of_slot = {s: n for n, s in zip(spec.hand_joint_names, lay.slot_of_joint)}
    rows = []
    for k in range(env.joint_enabled.shape[1]):
        rows.append({"slot": k, "joint": name_of_slot.get(k, ""), "enabled": bool(env.joint_enabled[0, k] > 0.5),
                     "origin_mm": (jorig[0, k] * 1000).tolist(), "axis": axis[0, k].tolist(),
                     "link_vec_mm": (lvec[0, k] * 1000).tolist(),
                     "lower_deg": math.degrees(float(env.joint_lower_hand[0, k])),
                     "upper_deg": math.degrees(float(env.joint_upper_hand[0, k])),
                     "joint_pos_norm": float(jpos[0, k])})
    out = {"spec": args.spec, "scene": ref.split(":")[1], "palm_extents_mm": (env.palm_extents[0] * 1000).tolist(),
           "fingertips_mm": (tips[0] * 1000).tolist(), "finger_slot_of_finger": list(lay.finger_slot_of_finger),
           "slots": rows}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[token_dump] {args.spec}: {sum(r['enabled'] for r in rows)} enabled slots -> {args.out}", flush=True)
    import os
    sys.stdout.flush()
    os._exit(0)                                         # app.close() can hang (7 Nov)


if __name__ == "__main__":
    main()
