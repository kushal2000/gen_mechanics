"""Do long-fingered hands get stuck on the table? Fingertip heights and time to the first goal under a final policy.

    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/diag_table_contact.py \
        --run map_elites --hands g0058_c00 g0058_c06 g0061_c04 g0018_c00 [--envs-per-hand 20] [--seconds 20]

The run's final policy (as final_archive_eval.py / render_archive_videos.py set it up), --envs-per-hand envs per hand,
one episode each from a common reset (no stagger), the 200-goal cap. Per hand: the fingertips' lowest height above the
table top over time (a fingertip body's origin below the top surface = the finger is pushed into the table), the
fraction of envs with a fingertip below the top at 0.5 s / 2 s / 5 s / 10 s, the wrist height, and the time to the
first goal (median, and how many envs had none within 5 s / 10 s). Writes
debug_outputs/evolution/final_archive_eval/table_contact_<run>.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "debug_outputs/evolution"
FINAL = RUNS / "final_archive_eval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="map_elites")
    ap.add_argument("--hands", nargs="+", required=True)
    ap.add_argument("--envs-per-hand", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--finger-pos-fraction", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-table-safe", action="store_true", help="the old reset (no fingertip check)")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO))
    full_man = json.loads((FINAL / f"{args.run}_manifest.json").read_text())
    spec_of = {h["id"]: h["spec"] for h in full_man["hands"]}
    man_path = Path(tempfile.mkdtemp()) / "manifest.json"
    man_path.write_text(json.dumps({"generation": full_man["generation"], "grid": full_man["grid"],
                                    "hands": [{"id": h, "spec": spec_of[h]} for h in args.hands]}))

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app  # noqa: F841
    import numpy as np
    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg

    last = json.loads((FINAL / "summary.json").read_text())[args.run]["policy_iteration"]
    gdir = RUNS / args.run / f"gen_{last:04d}"
    train_run = sorted(glob.glob(str(REPO / "debug_outputs/train_logs/evolution" / args.run / f"gen_{last:04d}" / "*/")))[0]
    full = OmegaConf.to_container(OmegaConf.load(next(Path(train_run).rglob(".hydra/config.yaml"))), resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    ref = f"multi:template:{man_path}"
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = args.envs_per_hand * len(args.hands)
    cfg.reset.stagger_initial_episodes = False
    cfg.goal.max_successes = 200
    if args.finger_pos_fraction is not None:
        cfg.reset.finger_pos_fraction = args.finger_pos_fraction
    cfg.reset.table_safe = not args.no_table_safe
    cfg.seed = 0
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    acfg = full["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    acfg["params"]["network"]["robot_spec"] = ref
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"train": acfg}, f)
    player = RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], f.name, str(gdir / "policy.pth"),
                      device=env.device, sapg_expl_coef=0.0, num_envs=env.num_envs)

    n, dt = env.num_envs, float(env.step_dt)
    steps = int(round(args.seconds / dt))
    top = float(cfg.table.top_height)
    hand = env.hand_of_env.long()
    real_tip = env.tip_body_idx != env.palm_body                      # padded tip slots point at the palm
    first_goal = torch.full((n,), float("nan"), device=env.device)
    tip_min, palm_z, open_ = [], [], torch.ones(n, dtype=torch.bool, device=env.device)
    rows = torch.arange(n, device=env.device).unsqueeze(1)
    with torch.no_grad():
        for t in range(steps):
            act = player.get_normalized_action(obs["policy"], deterministic_actions=False)
            obs, _, term, trunc, _ = env.step(act)
            z = env.robot.data.body_pos_w[rows, env.tip_body_idx, 2] - env.scene.env_origins[:, 2:3] - top
            tip_min.append(torch.where(real_tip, z, torch.full_like(z, 9.9)).min(1).values.cpu())
            palm_z.append((env.robot.data.body_pos_w[:, env.palm_body, 2] - env.scene.env_origins[:, 2] - top).cpu())
            hit = open_ & torch.isnan(first_goal) & (env.successes > 0)
            first_goal[hit] = (t + 1) * dt
            open_ &= ~(term | trunc)
    tip_min, palm_z = torch.stack(tip_min).numpy(), torch.stack(palm_z).numpy()      # (steps, n)
    fg = first_goal.cpu().numpy()
    out = {}
    for h, hid in enumerate(args.hands):
        m = (hand == h).cpu().numpy()
        at = lambda s: int(round(s / dt)) - 1
        r = {"envs": int(m.sum()),
             "fingertip_below_table_frac": {f"{s}s": float((tip_min[at(s), m] < 0).mean()) for s in (0.5, 2, 5, 10)},
             "lowest_fingertip_mm_median": {f"{s}s": float(np.median(tip_min[at(s), m]) * 1000) for s in (0.5, 2, 5, 10)},
             "palm_height_mm_median": {f"{s}s": float(np.median(palm_z[at(s), m]) * 1000) for s in (0.5, 2, 5, 10)},
             "first_goal_s_median": float(np.nanmedian(fg[m])) if np.isfinite(fg[m]).any() else None,
             "no_goal_by_5s": float((~(fg[m] <= 5)).mean()), "no_goal_by_10s": float((~(fg[m] <= 10)).mean())}
        out[hid] = r
        print(f"[diag] {hid}: fingertip below table {r['fingertip_below_table_frac']}, lowest fingertip mm "
              f"{ {k: round(v) for k, v in r['lowest_fingertip_mm_median'].items()} }, first goal median "
              f"{r['first_goal_s_median']} s, no goal by 5 s {r['no_goal_by_5s']:.0%}, by 10 s {r['no_goal_by_10s']:.0%}",
              flush=True)
    print(f"[diag] table_safe={cfg.reset.table_safe} finger_pos_fraction={cfg.reset.finger_pos_fraction}: "
          f"{getattr(env, 'reset_redraws', 0)} redraws, {getattr(env, 'reset_fallbacks', 0)} home-pose fallbacks over {n} envs", flush=True)
    (FINAL / f"table_contact_{args.run}{args.tag}.json").write_text(json.dumps(out, indent=1))
    os._exit(0)


if __name__ == "__main__":
    main()
