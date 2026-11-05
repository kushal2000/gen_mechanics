"""How often does a reset still start a fingertip inside the table? (6 Nov: the table-safe check reads each finger's
last link's FRAME, i.e. its last joint, not the fingertip pad.)

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/diag_reset_table.py [--every 16] [--envs 4] [--resets 5]

Every --every-th hand of the iteration-1 population, --envs envs each, the training env's reset (table_safe on), reset
--resets times; after each reset (no physics step), per real finger: the height above the table top of (a) the point the
check reads (the last link's frame) and (b) the fingertip pad (env.fingertips()). Prints the fraction of resets with a
pad below the top, by finger count, and the worst cases.
"""
import argparse, json, sys, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--resets", type=int, default=5)
    ap.add_argument("--clearance", type=float, default=None, help="reset.table_clearance (default: the run's saved value)")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO))
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([]); a.headless = True
    app = AppLauncher(a).app  # noqa: F841
    import numpy as np, torch
    from omegaconf import OmegaConf
    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    g = REPO / "debug_outputs/evolution/noevolution_4096/gen_0001"
    run = sorted((REPO / "debug_outputs/train_logs/evolution/noevolution_4096/gen_0001").glob("*/.hydra/config.yaml"))[0]
    env_d = OmegaConf.to_container(OmegaConf.load(run), resolve=True)["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    man = json.loads((g / "manifest.json").read_text())
    man["hands"] = man["hands"][::args.every]
    mf = tempfile.NamedTemporaryFile("w", suffix="_manifest.json", delete=False); json.dump(man, mf); mf.close()
    cfg = InHandOlympicsEnvCfg(); cfg.from_dict(env_d)
    cfg.hand.robot_spec = f"multi:template:{mf.name}"
    cfg.scene.num_envs = args.envs * len(man["hands"])
    cfg.reset.stagger_initial_episodes = False
    if args.clearance is not None:
        cfg.reset.table_clearance = args.clearance
    cfg.log.group_stats = False
    env = InHandOlympicsEnv(cfg)
    top = cfg.table.top_height
    real = (env.tip_body_idx != env.palm_body)
    hand = env.hand_of_env.long().cpu().numpy()
    nf = np.array([len(json.loads((REPO / h["spec"]).read_text())["design"]["fingers"]) if isinstance(json.loads((REPO / h["spec"]).read_text())["design"], dict) else 0 for h in man["hands"]])
    pad_min, frame_min, link_min = [], [], []
    for _ in range(args.resets):
        env.reset()
        env.sim.physics_sim_view.update_articulations_kinematic()
        env.robot.update(0.0)
        rows = torch.arange(env.num_envs, device=env.device).unsqueeze(1)
        fz = env.robot.data.body_pos_w[rows, env.tip_body_idx, 2] - env.scene.env_origins[:, 2:3] - top
        pz = env.fingertips()[..., 2] - top
        big = torch.full_like(fz, 9.0)
        frame_min.append(torch.where(real, fz, big).amin(1).cpu().numpy())
        pad_min.append(torch.where(real, pz, big).amin(1).cpu().numpy())
        if hasattr(env, "_table_margin"):
            mm = env._table_margin(torch.arange(env.num_envs, device=env.device))
            e = int(torch.where(real, pz, big).amin(1).argmin())
            print(f"  [check] env {e}: diag pad {float(torch.where(real, pz, big).amin(1)[e]) * 1000:.1f} mm, "
                  f"fix margin {float(mm[e]) * 1000:.1f} mm (pad clearance {cfg.reset.table_clearance * 1000:.0f}); "
                  f"envs with margin < -1 mm: {int((mm < -0.001).sum())}", flush=True)
        if hasattr(env, "reset_link_idx"):
            lzz = env.robot.data.body_pos_w[:, env.reset_link_idx, 2] - env.scene.env_origins[:, 2:3] - top
            link_min.append(torch.where(env.reset_link_valid, lzz, torch.full_like(lzz, 9.0)).amin(1).cpu().numpy())
    pad, frm = np.stack(pad_min), np.stack(frame_min)          # (resets, envs)
    print(f"\n[diag] {len(man['hands'])} hands x {args.envs} envs x {args.resets} resets; redraws {env.reset_redraws}, "
          f"lifted {getattr(env, 'reset_lifts', 0)} (mean lift {getattr(env, 'reset_lift_sum', 0.0) / max(getattr(env, 'reset_lifts', 0), 1) * 1000:.0f} mm), "
          f"could not clear {env.reset_fallbacks}", flush=True)
    print(f"  checked point (last link frame) below top+1cm: {(frm < 0.01).mean():.1%} (should be ~0 after the redraws)")
    print(f"  fingertip PAD below the table top: {(pad < 0).mean():.1%} of resets; below -1 cm: {(pad < -0.01).mean():.1%}; "
          f"below -3 cm: {(pad < -0.03).mean():.1%}")
    if link_min:
        lk = np.stack(link_min)
        print(f"  any finger JOINT below the table top: {(lk < 0).mean():.1%}; below top + 1 cm: {(lk < 0.01).mean():.1%}")
    for k in sorted(set(nf.tolist())):
        m = nf[hand] == k
        print(f"    {k} fingers: pad below top {(pad[:, m] < 0).mean():.1%}, median lowest pad {np.median(pad[:, m]) * 1000:.0f} mm")
    worst = np.argsort(pad.min(0))[:5]
    for e in worst:
        print(f"  worst: env {e} hand {man['hands'][hand[e]]['id']}: pad {pad[:, e].min() * 1000:.0f} mm, checked point {frm[:, e].min() * 1000:.0f} mm")
    sys.stdout.flush()
    import os; os._exit(0)

if __name__ == "__main__":
    main()
