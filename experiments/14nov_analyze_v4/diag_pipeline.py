"""Piping check for a hand under a v4 policy's env (14 Nov 2026): what the policy sees and what its actions do, per
joint slot -- to compare a vendor hand with its v4 replica (or with a training design) slot by slot.

  1. tokens at reset (env 0): per slot, the joint it holds, its normalised joint_pos, joint_origin / joint_link_vec
     (mm, canonical palm frame: normal, width, length), joint_axis, enabled; palm_extents, ee height.
  2. action probe: env 0 holds the zero action (home pose under finger_target=home); env 1 + k pushes ONLY slot k's
     action to --push (the policy's own action layout, through env.step), for --steps steps. Per slot: which real
     joint moved and by how much, and where its finger's tip went in the palm frame (normal, width, length mm)
     relative to env 0 -- flexion should curl the tip the same way on every hand.

    .venv_isaacsim/bin/python experiments/14nov_analyze_v4/diag_pipeline.py --hand wuji2 [--policy-run k5_none:48]
Output: experiments/14nov_analyze_v4/results/diag/<hand>.json and a printed table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(Path(__file__).parent)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True, help="vendor name (wuji2, leap, ...) or a .spec.json")
    ap.add_argument("--policy-run", default="k5_none:48", help="whose training config the env replays")
    ap.add_argument("--push", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    la, _ = lp.parse_known_args([])
    la.headless = True
    app = AppLauncher(la).app  # noqa: F841
    try:
        run(a)
    except Exception:
        import traceback
        traceback.print_exc()
    sys.stdout.flush()
    os._exit(0)


def run(a) -> None:
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from isaaclab.utils.math import quat_apply_inverse, quat_mul
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    from isaacsimenvs.inhand_olympics.observations import field_sizes
    from zero_shot_commercial import hand_ref, resolve

    pol = resolve(a.policy_run)
    full = OmegaConf.to_container(OmegaConf.load(Path(pol["train_run"]) / ".hydra/config.yaml"), resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    ref = f"multi:finger:{hand_ref(a.hand)}"
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    hs_probe = None
    cfg.scene.num_envs = 1                                       # resized below once the slot count is known
    # slot count: build once with 1 env to read the layout, then the probe scene
    from isaacsimenvs.common.robots.multi_hand import HandSet  # noqa: F401  (layout comes from the env)
    cfg.scene.num_envs = 32
    cfg.reset.stagger_initial_episodes = False
    cfg.reset.table_safe = True
    cfg.log.group_stats = False
    cfg.log.design_stats = False
    cfg.seed = 1
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    dev, n = env.device, env.num_envs
    n_all = len(env.canon_joint_names)
    n_hand = env.joint_enabled.shape[1]
    n_tips = env.fingertips().shape[1]
    sizes = field_sizes(n_all, n_hand, n_tips, env.task_onehot().shape[1])
    off, o = {}, 0
    for f in env.cfg.obs.obs_list:
        off[f] = (o, o + sizes[f])
        o += sizes[f]
    hs = env.hand_set
    spec, lay = hs.specs[0], hs.layouts[0]
    slot_name = {s: nm for nm, s in zip(spec.hand_joint_names, lay.slot_of_joint)}
    finger_of_slot = {}
    for fi, joints in enumerate(lay.fingers):
        for j in joints:
            finger_of_slot[lay.slot_of_joint[j]] = fi
    # --- action probe: env 0 baseline, env 1 + i pushes slot i (policy action layout: wrist 6, then the slots) -----
    slots = sorted(slot_name)[: n - 1]
    n_act = env.action_space.shape[-1]
    act = torch.zeros(n, n_act, device=dev)
    for i, k in enumerate(slots):
        act[1 + i, 6 + k] = a.push
    obs, _ = env.reset()
    q0 = env.robot.data.joint_pos.clone()
    for _ in range(a.steps):
        obs, *_ = env.step(act)
    q = env.robot.data.joint_pos
    names = list(env.robot.joint_names)
    palm = env.robot.data.body_state_w[:, env.palm_body]
    rot = quat_mul(palm[:, 3:7], env.palm_frame_quat)            # canonical palm frame (normal, width, length)
    tips = env.fingertips()                                       # (N, F, 3) env frame
    rel = tips - (palm[:, :3] - env.scene.env_origins).unsqueeze(1)
    tips_c = quat_apply_inverse(rot.unsqueeze(1).expand(-1, rel.shape[1], -1).reshape(-1, 4),
                                rel.reshape(-1, 3)).reshape(rel.shape) * 1000
    probe = []
    print(f"\naction probe: +{a.push} on one slot for {a.steps} steps; tip displacement vs env 0 in the palm frame "
          f"(normal, width, length mm)")
    for i, k in enumerate(slots):
        e = 1 + i
        dq = (q[e] - q[0]).abs()
        dq[env.wrist_ids] = 0
        j = int(dq.argmax())
        fi = finger_of_slot.get(k)
        d_tip = (tips_c[e, fi] - tips_c[0, fi]).cpu().numpy() if fi is not None and fi < tips_c.shape[1] else None
        moved = names[j]
        r = {"slot": k, "joint": slot_name[k], "moved": moved, "d_deg": round(math.degrees(float(q[e, j] - q[0, j])), 1),
             "tip_mm": None if d_tip is None else [round(float(t), 1) for t in d_tip]}
        probe.append(r)
        flag = "" if moved.endswith(slot_name[k]) or moved == slot_name[k] else "   <-- MOVED ANOTHER JOINT"
        print(f"slot {k:2d} {slot_name[k][:26]:26s} -> {moved[:30]:30s} {r['d_deg']:7.1f} deg  tip {r['tip_mm']}{flag}")
    x = obs["policy"][0]                                   # env 0: zero action = home, settled

    def seg(f, k=None, w=1):
        lo, hi = off[f]
        v = x[lo:hi]
        return v if k is None else v[k * w:(k + 1) * w]

    rows = []
    for k in sorted(slot_name):
        r = {"slot": k, "joint": slot_name[k], "finger": finger_of_slot.get(k),
             "pos_norm": round(float(seg("joint_pos")[6 + k]), 3) if "joint_pos" in off else None,
             "enabled": float(seg("joint_enabled", k)[0]) if "joint_enabled" in off else None}
        for f in ("joint_origin", "joint_link_vec", "joint_axis"):
            if f in off:
                v = seg(f, k, 3).cpu().numpy()
                r[f] = [round(float(t) * (1000 if f != "joint_axis" else 1), 1 if f != "joint_axis" else 2) for t in v]
        rows.append(r)
    print(f"\n=== {a.hand}: {len(slot_name)} hand joints in {n_hand} slots; obs fields {list(off)}")
    if "palm_extents" in off:
        print("palm_extents (mm):", [round(float(t) * 1000) for t in seg("palm_extents")])
    print(f"{'slot':>4} {'finger':>6} {'joint':28s} {'pos':>6}  {'origin mm (n,w,l)':22s} {'link mm':22s} axis")
    for r in rows:
        print(f"{r['slot']:4d} {r['finger']!s:>6} {r['joint'][:28]:28s} {r['pos_norm']!s:>6}  {str(r.get('joint_origin')):22s} "
              f"{str(r.get('joint_link_vec')):22s} {r.get('joint_axis')}")

    out = Path(__file__).parent / "results/diag" / f"{Path(a.hand).name.replace('.spec.json', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"hand": a.hand, "policy_run": a.policy_run, "tokens": rows, "probe": probe,
                               "obs_fields": list(off)}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
