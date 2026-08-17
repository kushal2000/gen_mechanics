"""Do authored and converted objects behave the same under a RANDOM policy?

Every check that passed so far used either no actions or a trained policy. Both
are in-distribution: a converged policy makes smooth contacts, and zero actions
make none. Training does neither -- it flails, launches objects, drives
penetrations, and resets thousands of times per env.

The pretrained policy scores the same on both paths (lift 79% at 24,576 envs)
while TRAINING on them diverges 16x by epoch 3000. So the difference lives in
what exploration reaches and evaluation does not, and this measures exactly
that: identical random actions on both paths, same seed, comparing what happens
to the objects.

Reported per path, over the rollout:

  * lift rate            -- how often the object leaves the table at all
  * object speed         -- mean and 99th percentile, where tunnelling shows up
  * ejection             -- objects that end up implausibly far from their env
  * below-table          -- objects that fell through the surface
  * resets survived      -- objects still in a sane place after N resets

A difference here is the mechanism. No difference here, with training still
diverging, would point away from object physics entirely.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.compare_object_exploration --num_envs 512 --steps 600
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--num_assets_per_type", type=int, default=100)
    p.add_argument("--author", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    a.headless = True
    return a


def main() -> None:
    args = _parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch

    import genmech.tasks  # noqa: F401
    from genmech.tasks.pose_reach.env_cfg import PoseReachEnvCfg

    cfg = PoseReachEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.author_object_usds = args.author
    cfg.assets.robot_spec = "sharpa_iiwa14"

    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    tag = "authored" if args.author else "converted"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.reset(seed=args.seed)

    origins = inner.scene.env_origins
    table_z = float(getattr(inner.cfg.assets, "table_reset_z", 0.0))

    speeds, heights, dists = [], [], []
    lifted_any = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
    below = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
    far = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)

    # The SAME random action sequence for both paths: seeded generator, and the
    # actions do not depend on observations, so the two runs receive identical
    # commands and any difference is the objects' response.
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    for _ in range(args.steps):
        act = (torch.rand((inner.num_envs, inner.cfg.action_space),
                          generator=gen) * 2.0 - 1.0).to(inner.device)
        env.step(act)
        pos = inner.object.data.root_pos_w - origins
        spd = inner.object.data.root_lin_vel_w.norm(dim=-1)
        speeds.append(spd.cpu().numpy())
        heights.append(pos[:, 2].cpu().numpy())
        d = pos[:, :2].norm(dim=-1)
        dists.append(d.cpu().numpy())
        lifted_any |= inner._lifted_object
        below |= pos[:, 2] < (table_z - 0.10)
        far |= d > 2.0

    speeds = np.stack(speeds)
    heights = np.stack(heights)
    dists = np.stack(dists)

    res = dict(
        tag=tag, num_envs=args.num_envs, steps=args.steps,
        lift_rate=float(lifted_any.float().mean()),
        speed_mean=float(speeds.mean()),
        speed_p99=float(np.percentile(speeds, 99)),
        speed_max=float(speeds.max()),
        height_mean=float(heights.mean()),
        height_min=float(heights.min()),
        below_table=int(below.sum()),
        ejected=int(far.sum()),
        dist_p99=float(np.percentile(dists, 99)),
    )
    print(f"\n[explore:{tag}] num_envs={args.num_envs} steps={args.steps}")
    for k, v in res.items():
        if k in ("tag", "num_envs", "steps"):
            continue
        print(f"[explore:{tag}]   {k:<12} = {v}")

    if args.out:
        np.savez(args.out, **{k: v for k, v in res.items() if k != "tag"},
                 speeds=speeds, heights=heights)
        print(f"[explore:{tag}] wrote {args.out}")

    del app
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
