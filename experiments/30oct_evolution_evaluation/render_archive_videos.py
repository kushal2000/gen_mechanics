"""Film evolved hands in sim under their run's final policy: one clip per hand, plus a grid video per call.

    GENMECH_KEEP_VISUALS=1 .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/render_archive_videos.py \
        --run map_elites --top 4 [--hands g0058_c06 ...] [--seconds 20] [--frame-only]

Hands: the --top best niches of the run's final archive re-evaluation (final_archive_eval.py's summary.json: each
niche's best hand under the final policy), or --hands by id. One env per hand in a template scene of just those hands
(the run's own env config, as eval_hands_cap.py builds it), the run's last policy (SAPG leader, sampled actions as in
every evaluation), the goal cap raised to 200 so a clip is not cut at 50. Each hand is filmed in its own pass (one
camera), following its palm; tiles are captioned with run, hand, niche, joints and goals so far.
Writes debug_outputs/evolution/videos/final_archive/<run>/<hand>.mp4 (or .png with --frame-only) and grid.mp4.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "debug_outputs/evolution"
FINAL = RUNS / "final_archive_eval"
LABEL = {"baseline": "Greedy evolution", "map_elites": "MAP-Elites"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=sorted(LABEL))
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--hands", nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", default="640,480")
    ap.add_argument("--eye", default="0.42,-0.42,0.16", help="camera offset from the palm-pen midpoint (world, m)")
    ap.add_argument("--look", default="0.0,0.0,0.0", help="look-at offset from the palm-pen midpoint (world, m)")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--frame-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-tag", default="", help="final_archive_eval.py output tag, e.g. _it73_100env")
    args = ap.parse_args()
    os.environ.setdefault("GENMECH_KEEP_VISUALS", "1")
    sys.path.insert(0, str(REPO))

    summary = json.loads((FINAL / f"summary{args.summary_tag}.json").read_text())[args.run]
    best = {n: d["best"] for n, d in summary["per_niche"].items() if d["best"]}
    if args.hands:
        niche_of = {b["hand"]: n for n, b in best.items()}
        chosen = [(niche_of.get(h, "?"), h) for h in args.hands]
    else:
        chosen = [(n, b["hand"]) for n, b in sorted(best.items(), key=lambda kv: -kv[1]["final_goals"])[:args.top]]
    full_man = json.loads((FINAL / (f"{args.run}{args.summary_tag}_manifest.json" if args.summary_tag else f"{args.run}_manifest.json")).read_text())
    spec_of = {h["id"]: h["spec"] for h in full_man["hands"]}
    man = {"generation": full_man["generation"], "grid": full_man["grid"],
           "hands": [{"id": h, "spec": spec_of[h]} for _, h in chosen]}
    out = Path(args.out or RUNS / "videos/final_archive" / args.run)
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    man_path.write_text(json.dumps(man, indent=1))

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless, a.enable_cameras = True, True
    app = AppLauncher(a).app  # noqa: F841

    import imageio
    import numpy as np
    import torch
    import yaml
    from omegaconf import OmegaConf
    from PIL import Image, ImageDraw, ImageFont

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg

    W, H = (int(v) for v in args.res.split(","))
    EYE = np.array([float(v) for v in args.eye.split(",")])
    LOOK = np.array([float(v) for v in args.look.split(",")])
    last = summary["policy_iteration"]
    gdir = RUNS / args.run / f"gen_{last:04d}"
    train_run = sorted(glob.glob(str(REPO / "debug_outputs/train_logs/evolution" / args.run / f"gen_{last:04d}" / "*/")))[0]
    hydra = next(Path(train_run).rglob(".hydra/config.yaml"))
    full = OmegaConf.to_container(OmegaConf.load(hydra), resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    ref = f"multi:template:{man_path}"
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = len(chosen)
    cfg.reset.stagger_initial_episodes = False
    cfg.goal.max_successes = 200
    cfg.seed = 0
    cfg.viewer.resolution = (W, H)
    env = InHandOlympicsEnv(cfg, render_mode="rgb_array")
    obs, _ = env.reset()
    acfg = full["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    acfg["params"]["network"]["robot_spec"] = ref
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"train": acfg}, f)
    player = RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], f.name, str(gdir / "policy.pth"),
                      device=env.device, sapg_expl_coef=0.0, num_envs=env.num_envs)
    hand_of_env = env.hand_of_env.long().tolist()
    try:
        big, small = ImageFont.truetype("DejaVuSans-Bold.ttf", 24), ImageFont.truetype("DejaVuSans.ttf", 17)
    except OSError:
        big = small = ImageFont.load_default()

    def caption(frame, title, sub):
        im = Image.fromarray(np.asarray(frame)[..., :3])
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, 62], fill=(20, 20, 20))
        d.text((12, 5), title, font=big, fill=(255, 255, 255))
        d.text((12, 37), sub, font=small, fill=(205, 205, 205))
        return np.asarray(im)

    steps = 1 if args.frame_only else int(round(args.seconds / float(env.step_dt)))
    stride = max(1, int(round(1.0 / (args.fps * float(env.step_dt)))))
    print(f"[render] {args.run}: {[h for _, h in chosen]}, {steps} steps, every {stride}th frame", flush=True)
    def aim(e):     # the camera on the midpoint of palm and pen, so both stay in shot
        c = 0.5 * (env.robot.data.body_pos_w[e, env.palm_body] + env.pen.data.root_pos_w[e]).cpu().numpy()
        env.sim.set_camera_view(eye=(c + EYE).tolist(), target=(c + LOOK).tolist())

    clips = []
    for h, (niche, hid) in enumerate(chosen):
        e = hand_of_env.index(h)
        n_joints = env.hand_set.specs[h].num_hand_joints if hasattr(env.hand_set.specs[h], "num_hand_joints") else "?"
        obs, _ = env.reset()
        player.reset() if hasattr(player, "reset") else None
        aim(e)
        for _ in range(4):          # the renderer returns black frames until it has warmed up
            env.render()
        frames, t0 = [], time.time()
        for t in range(steps):
            with torch.no_grad():
                act = player.get_normalized_action(obs["policy"], deterministic_actions=False)
            obs, *_ = env.step(act)
            if t % stride == 0 or args.frame_only:
                aim(e)
                goals = int(env.successes[e].item())
                frames.append(caption(env.render(), f"{LABEL[args.run]}: {hid}",
                                      f"niche {niche}   {n_joints} joints   goals: {goals}"))
        clips.append(frames)
        if args.frame_only:
            imageio.imwrite(out / f"{hid}.png", frames[-1])
        else:
            imageio.mimwrite(out / f"{hid}.mp4", frames, fps=args.fps, quality=8, macro_block_size=None)
        print(f"[render] {hid} ({niche}): {int(env.successes[e].item())} goals shown, {time.time() - t0:.0f} s wall", flush=True)

    def grid(tiles):
        rows = [np.concatenate(tiles[i:i + args.cols], axis=1) for i in range(0, len(tiles), args.cols)]
        if len(rows) > 1 and rows[-1].shape != rows[0].shape:
            pad = np.zeros((rows[0].shape[0], rows[0].shape[1] - rows[-1].shape[1], 3), dtype=rows[0].dtype)
            rows[-1] = np.concatenate([rows[-1], pad], axis=1)
        return np.concatenate(rows, axis=0)

    if args.frame_only:
        imageio.imwrite(out / "grid.png", grid([c[-1] for c in clips]))
    else:
        n = min(len(c) for c in clips)
        imageio.mimwrite(out / "grid.mp4", [grid([c[i] for c in clips]) for i in range(n)], fps=args.fps, quality=8,
                         macro_block_size=None)
    print(f"[render] wrote {out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
