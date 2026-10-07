"""Capture one hand's SPAWN state in the in-hand reorientation task, for viser_spawn.py to compare.

Builds GenMech-InHandReorient-Direct-v0 with a trained run's saved config (default: 12491, the full Wuji v2 at
5 rad/s with moving average 0.1 -- canonical palm frame + palm_extents, the observation list the 06oct runs use),
swaps in the hand to capture (env.assets.robot_spec), resets, and writes env 0's state as JSON:

  * the policy observation, split into its terms (layout.field_offsets) with a label per entry
    (joint name for per-joint terms), raw and as the policy sees it (after clipping);
  * every robot body's world pose (relative to the env origin), joint positions and limits;
  * object, goal, palm frame and palm-centre poses, fingertip bodies.

One process per hand (Isaac Sim builds one scene per process):

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/capture_spawn.py \
        --spec wuji2_left_uniform_handonly --name wuji2_commercial
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/capture_spawn.py \
        --spec handonly:assets/urdf/unified_dynamics_commercial_hands_replicas/wuji2/wuji2_left_replica.json \
        --name wuji2_replica
    -> results/spawn/<name>.json

Use the same --seed for both hands of a pair. At seed 0 the object and goal orientations came out identical across
all six hands captured on 2026-10-07; the viewer still compares the poses rather than assuming it.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

TASK = "GenMech-InHandReorient-Direct-v0"
DEFAULT_RUN = "debug_outputs/train_logs/06oct_wuji_5rads/0_scale_train_left_wuji2_full_v5_ema0.1_c01_*_12491_0_*"
PER_JOINT = {"joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets"}
PER_HAND_JOINT = {"joint_lower", "joint_upper", "joint_enabled"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="registry name, or handonly:<population .json>")
    ap.add_argument("--name", required=True)
    ap.add_argument("--run", default=DEFAULT_RUN, help="run directory (glob) whose saved config to use")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    spec_ref = args.spec
    if spec_ref.startswith("handonly:") and not spec_ref[len("handonly:"):].startswith("/"):
        spec_ref = "handonly:" + str(REPO / spec_ref[len("handonly:"):])

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    app_args, _ = lp.parse_known_args([])
    app_args.headless, app_args.device = True, args.device
    app = AppLauncher(app_args).app

    import gymnasium as gym
    import torch
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml
    from isaacsimenvs.pose_reaching_6d.obs_utils.layout import field_offsets
    from isaacsimenvs.pose_reaching_6d.scene_utils.assembly import _resolve_spec

    run = sorted(glob.glob(str(REPO / args.run)))
    if not run:
        raise SystemExit(f"no run matches {args.run}")
    saved = OmegaConf.load(pathlib.Path(run[0]) / "rank_0/.hydra/config.yaml")
    saved.env.assets.robot_spec = spec_ref
    sys.argv = [sys.argv[0]]

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed

        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        obs, _ = env.reset()
        _, spec = _resolve_spec(inner.cfg)
        e = 0
        org = inner.scene.env_origins[e]

        def pose(pos, quat):            # world pose relative to env origin, quaternion wxyz
            return [float(v) for v in (pos - org)] + [float(v) for v in quat]

        canon = list(spec.joint_names_canonical)
        hand = list(spec.hand_joint_names)
        obs_list = list(inner.cfg.obs.obs_list)
        offs = field_offsets(obs_list, spec)
        pol = obs["policy"][e].float().cpu()
        terms = {}
        for f in obs_list:
            a, b = offs[f]
            vals = pol[a:b].tolist()
            if f in PER_JOINT:
                labels = canon
            elif f in PER_HAND_JOINT:
                labels = hand
            elif len(vals) % len(hand) == 0 and len(vals) > len(hand) and f in ("joint_link_bbox",
                                                                               "object_keypoints_rel_joint"):
                per = len(vals) // len(hand)
                labels = [f"{j}[{i}]" for j in hand for i in range(per)]
            else:
                labels = [str(i) for i in range(len(vals))]
            terms[f] = {"slice": [a, b], "values": vals, "labels": labels}

        R = inner.robot.data
        from isaacsimenvs.pose_reaching_6d.obs_utils.observations import _apply_local_offset
        ps = R.body_state_w[:, inner._palm_body_id, :]
        palm_center_w = _apply_local_offset(ps[:, 0:3], ps[:, 3:7], inner._palm_center_offset, (inner.num_envs,))
        bodies = {n: pose(R.body_state_w[e, i, 0:3], R.body_state_w[e, i, 3:7])
                  for i, n in enumerate(inner.robot.body_names)}
        tip_ids = getattr(inner, "_fingertip_body_ids", [])
        tips = [inner.robot.body_names[int(i)] for i in (tip_ids.tolist() if hasattr(tip_ids, "tolist") else tip_ids)]
        # What reset.py's object_in_hand placement reads, so the cube's spawn position can be decomposed offline:
        # the canonical palm frame (palm body quat x palm_frame_quat), fingertip PAD centres, which pads are real,
        # and the palm slab keypoints (keypoint 1 - keypoint 0 is the thickness edge = grasp normal).
        from isaaclab.utils.math import quat_apply, quat_mul
        ft = R.body_state_w[e, inner._fingertip_body_ids, :]
        offs = inner._fingertip_offsets[e] if inner._fingertip_offsets.dim() == 3 else inner._fingertip_offsets
        pads_w = ft[:, 0:3] + quat_apply(ft[:, 3:7], offs)
        canon_q = quat_mul(ps[e:e + 1, 3:7], inner._palm_frame_quat[e:e + 1])[0]
        out = {
            "name": args.name, "spec": args.spec, "config_run": str(pathlib.Path(run[0]).relative_to(REPO)),
            "seed": args.seed, "obs_list": obs_list, "obs_dim": int(pol.numel()), "terms": terms,
            "joint_names": list(inner.robot.joint_names),
            "joint_pos": R.joint_pos[e].tolist(),
            "joint_limits": R.joint_pos_limits[e].tolist(),
            "canonical_joint_names": canon, "hand_joint_names": hand,
            "bodies": bodies, "root": pose(R.root_state_w[e, 0:3], R.root_state_w[e, 3:7]),
            "palm_body": inner.robot.body_names[int(inner._palm_body_id)],
            "palm_center": [float(v) for v in (palm_center_w[e] - org)],
            "fingertip_bodies": tips,
            "fingertip_pads": (pads_w - org).tolist(),
            "fingertip_valid": inner._fingertip_mask[e].tolist(),
            "palm_frame_canonical_wxyz": canon_q.tolist(),
            "palm_keypoints_local": inner._palm_keypoints_local[e].tolist(),
            "joint_link_bodies": list(spec.joint_link_bodies),
            # What joint_link_bbox is built from (observations._joint_link_geometry_obs), per hand joint in policy
            # order: the body each box rides on, its 4 corners in that body's frame, and whether it is real.
            "bbox_bodies": [inner.robot.body_names[int(i)] for i in inner._joint_link_body_ids.tolist()]
            if hasattr(inner._joint_link_body_ids, "tolist") else
            [inner.robot.body_names[int(i)] for i in inner._joint_link_body_ids],
            "bbox_local": inner._joint_link_bbox_local[e].tolist(),
            "bbox_valid": inner._joint_geometry_valid[e].tolist(),
            "reset": {k: getattr(inner.cfg.reset, k) for k in ("object_in_hand", "in_hand_placement",
                                                                "in_hand_fingertip_fraction", "in_hand_clearance",
                                                                "in_hand_drop_margin")},
            "object_size": list(inner.cfg.reward.fixed_size),
            "object": pose(inner.object.data.root_pos_w[e], inner.object.data.root_quat_w[e]),
            "goal": pose(inner.goal_viz.data.root_pos_w[e], inner.goal_viz.data.root_quat_w[e]),
        }
        path = pathlib.Path(args.out) if args.out else HERE / "results/spawn" / f"{args.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))
        print(f"[spawn] {args.name}: obs {out['obs_dim']} ({len(obs_list)} terms), {len(bodies)} bodies -> {path}",
              flush=True)
        env.close()

    _run()
    app.close()


if __name__ == "__main__":
    main()
