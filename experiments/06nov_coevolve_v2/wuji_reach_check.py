"""Does the vendor Wuji v2 even reach the pen? One multi:finger: scene with the vendor hand and the controller's best
evolved design in Wuji's niche (4 envs each), the v1 k5_none gen-7 policy with no adapters, every env's first
300 control steps: per hand the palm centre's and fingertips' height above the table at reset, the pen's position,
and the closest any fingertip gets to the pen over the run.

    .venv_isaacsim/bin/python experiments/06nov_coevolve_v2/wuji_reach_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import os as _os
RUN = REPO / "debug_outputs/coevolve" / _os.environ.get("REACH_RUN", "v2/k5_none")     # v2 by default since 7 Nov
GEN = int(_os.environ.get("REACH_GEN", "-1"))
if GEN < 0:
    GEN = max(int(p.parent.name[4:]) for p in RUN.glob("gen_*/policy.pth"))


def main() -> None:
    sys.path.insert(0, str(REPO))
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app  # noqa: F841

    import torch
    import yaml
    from isaaclab.utils.math import quat_mul
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.common.obs_geometry import apply_local_offset
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg

    import os
    ref_spec = next(h["spec"] for h in json.loads(
        (REPO / "debug_outputs/coevolve/studies/wuji_zero_shot/manifest.json").read_text())["hands"]
        if h["id"] == "ref_f5_m20-21_best")
    hands = os.environ.get("REACH_HANDS", f"wuji2_left_uniform_handonly+{ref_spec}").split("+")
    tr = json.loads((RUN / "state.json").read_text())["train_runs"][str(GEN)]
    full = OmegaConf.to_container(OmegaConf.load(next(Path(tr).rglob(".hydra/config.yaml"))), resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    ref = "multi:finger:" + "+".join(hands)
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = 8
    cfg.reset.stagger_initial_episodes = False
    cfg.reset.table_safe = True                             # finger-scene table-safe fixed 7 Nov
    cfg.log.group_stats = False
    cfg.log.design_stats = False
    cfg.seed = 7500
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    acfg = full["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    acfg["params"]["network"]["robot_spec"] = ref
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump({"train": acfg}, fh)
    player = RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], fh.name,
                      str((RUN / f"gen_{GEN:04d}/policy.pth").resolve()), device=env.device, sapg_expl_coef=0.0,
                      num_envs=env.num_envs)
    hand = env.hand_of_env.long()
    top = float(cfg.table.top_height)

    def state():
        origins = env.scene.env_origins
        palm = env.robot.data.body_state_w[:, env.palm_body]
        centre = apply_local_offset(palm[:, 0:3], palm[:, 3:7], env.palm_center_offset, (env.num_envs,)) - origins
        tips = env.fingertips() - origins.unsqueeze(1)
        pen, _ = env.pen_pose()
        pen = pen - origins
        valid = env.fingertip_valid if hasattr(env, "fingertip_valid") else torch.ones(tips.shape[:2], dtype=torch.bool,
                                                                                       device=env.device)
        d = (tips - pen.unsqueeze(1)).norm(dim=-1)
        d = torch.where(valid, d, torch.full_like(d, 1e9))
        return centre, tips, pen, valid, d.min(1).values

    c0, t0, p0, v0, d0 = state()
    best = d0.clone()
    trace = {h: [] for h in range(len(hands))}
    poses = {}
    from isaaclab.utils.math import quat_apply
    def snap(t):
        c, tp, p, v, d = state()
        palm_rot = quat_mul(env.robot.data.body_state_w[:, env.palm_body, 3:7], env.palm_frame_quat)
        nz = quat_apply(palm_rot, torch.tensor([[1.0, 0, 0]], device=env.device).expand(env.num_envs, 3))[:, 2]
        tz = torch.where(v, tp[..., 2], torch.full_like(tp[..., 2], 9.0)).amin(1)
        jp = env.robot.data.joint_pos
        for h in range(len(hands)):
            e0 = int((hand == h).nonzero()[0, 0])
            names = env.robot.real_joint_names(e0)
            cols = env.robot.real_slots(e0)
            pb = env.robot.data.body_state_w[e0, env.palm_body, :7] - torch.cat([env.scene.env_origins[e0], torch.zeros(4, device=env.device)])
            pp, pq = env.pen_pose()
            poses.setdefault(h, {})[t] = {"q": {nm: float(jp[e0, c]) for nm, c in zip(names, cols)},
                                           "target": {nm: float(env.targets[e0, c]) for nm, c in zip(names, cols)},
                                           "palm": pb.tolist(), "pen": pp[e0].tolist(),
                                           "pen_quat": pq[e0].tolist(), "table_top": top}
        for h in range(len(hands)):
            m = hand == h
            trace[h].append((t, 1000 * (c[m][:, 2].mean() - top), float(nz[m].mean()), 1000 * (tz[m].mean() - top),
                             1000 * d[m].mean(), 1000 * (p[m][:, 2].mean() - top)))
    snap(0)
    with torch.no_grad():
        for t in range(1, 301):
            act = player.get_action_and_value(obs["policy"], deterministic_actions=False)[0]
            obs, *_ = env.step(act)
            *_, d = state()
            best = torch.minimum(best, d)
            if t in (5, 10, 15, 20, 25, 35, 50, 75, 100, 200, 300):
                snap(t)
    for h, nm in enumerate(hands):
        print(f"[trace] {Path(nm).name}: step | palm centre z | palm normal . world z | lowest tip z | closest tip-pen | pen z (mm above table)")
        for row in trace[h]:
            print(f"[trace]   {row[0]:4d} | {row[1]:6.0f} | {row[2]:+.2f} | {row[3]:6.0f} | {row[4]:6.0f} | {row[5]:5.0f}")
    out_p = os.environ.get("REACH_POSES")
    if out_p:
        Path(out_p).write_text(json.dumps({"hands": hands, "poses": {str(h): v for h, v in poses.items()}}, indent=1))
    for h, name in enumerate([Path(x).name for x in hands]):
        m = hand == h
        tz = torch.where(v0[m], t0[m][..., 2], torch.full_like(t0[m][..., 2], float("nan")))
        print(f"[reach] {name}:")
        print(f"   palm centre at reset: height above table {1000 * (c0[m][:, 2].mean() - top):.0f} mm, "
              f"xy ({1000 * c0[m][:, 0].mean():.0f}, {1000 * c0[m][:, 1].mean():.0f}) mm")
        print(f"   pen at reset: xy ({1000 * p0[m][:, 0].mean():.0f}, {1000 * p0[m][:, 1].mean():.0f}) mm, "
              f"height {1000 * (p0[m][:, 2].mean() - top):.0f} mm above table")
        print(f"   fingertips at reset: lowest {1000 * (torch.nanquantile(tz, 0.0) - top):.0f} mm, highest "
              f"{1000 * (torch.nanquantile(tz, 1.0) - top):.0f} mm above table; {int(v0[m][0].sum())} valid tips")
        print(f"   closest fingertip to the pen: {1000 * d0[m].mean():.0f} mm at reset, {1000 * best[m].mean():.0f} mm "
              f"best over 300 steps (per env: {[round(1000 * float(x)) for x in best[m]]})")
    sys.stdout.flush()
    import os as _os
    _os._exit(0)


if __name__ == "__main__":
    main()
