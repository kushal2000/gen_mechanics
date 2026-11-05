"""What a training run would log per design, without training: the iteration's policy rolled out exactly as during
training -- all 24,576 envs (6 per hand, dealt round-robin so each SAPG block holds each hand once), each env with
its own block's exploration coefficient (linspace(50, 0, 6): followers 50..10, the leader 0), sampled actions,
staggered episodes, every episode counted -- and no updates, so the numbers belong to that checkpoint.

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/eval_training_like.py --manifest <gen manifest> \
        --checkpoint <policy.pth> --train-run <run dir> --out-dir <dir> [--steps 8000]

Bookkeeping is the env's own per-design logger (cfg.log.design_stats): per-env sums every 1,600 steps in
<out-dir>/step_<N>.npz, meta.npz, hands.json -- the files a training run with env.log.design_stats=True writes.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--envs", type=int, default=24576)
    ap.add_argument("--block", type=int, default=4096, help="expl_coef_block_size")
    ap.add_argument("--interval", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--table-clearance", type=float, default=None,
                    help="reset.table_clearance (default: the run's saved value; the 6 Nov fix uses 0.02)")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO))

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app  # noqa: F841

    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg

    hydra = next(Path(args.train_run).rglob(".hydra/config.yaml"))
    full = OmegaConf.to_container(OmegaConf.load(hydra), resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    ref = f"multi:template:{args.manifest}"
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = args.envs
    cfg.reset.stagger_initial_episodes = True            # as in training
    if args.table_clearance is not None:
        cfg.reset.table_clearance = args.table_clearance
    cfg.log.design_stats = True
    cfg.log.design_interval_steps = args.interval
    cfg.log.design_dir = args.out_dir
    cfg.seed = args.seed
    t0 = time.time()
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    acfg = full["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    acfg["params"]["network"]["robot_spec"] = ref
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"train": acfg}, f)
    n, dev = env.num_envs, env.device
    player = RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], f.name, args.checkpoint, device=dev,
                      sapg_expl_coef=0.0, num_envs=n)      # 0.0 only sizes the obs; the column is set per env below
    blocks = n // args.block
    coef = torch.linspace(50.0, 0.0, blocks, device=dev).repeat_interleave(args.block).reshape(-1, 1)
    p = player.player
    print(f"[training-like] {n} envs, {blocks} SAPG blocks, ready in {time.time() - t0:.0f} s", flush=True)
    t1 = time.time()
    with torch.no_grad():
        for step in range(args.steps):
            x = p._preproc_obs(torch.cat([obs["policy"], coef.to(obs["policy"].dtype)], dim=1))
            res = p.model({"is_train": False, "prev_actions": None, "obs": x, "rnn_states": p.states})
            p.states = res["rnn_states"]
            obs, *_ = env.step(res["actions"].reshape(n, -1))
            if (step + 1) % 1600 == 0:
                print(f"[training-like] {step + 1} steps, {time.time() - t1:.0f} s", flush=True)
    (Path(args.out_dir) / "done.json").write_text(json.dumps({"checkpoint": args.checkpoint, "steps": args.steps,
                                                               "envs": n, "blocks": blocks, "seed": args.seed}))
    print(f"[training-like] done in {time.time() - t1:.0f} s -> {args.out_dir}", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
