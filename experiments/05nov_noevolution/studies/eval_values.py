"""eval_hands.py's evaluation, also recording the critic: per hand, the policy's goals AND what its value function
predicted along the way (study 3).

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/eval_values.py --manifest <m> --checkpoint <pth> \
        --train-run <run dir> --out <fitness.json> [--envs-per-hand 4] [--seed 3]

The loop is coevolution/eval/eval_hands.py's (one template scene, sampled actions, the leader's SAPG coefficient,
every env's first episode), copied rather than edited because the running no-evolution chain calls that file. The
action and V(s) come from one forward pass (RlPlayer.get_action_and_value), so the trajectory is the one a plain eval
would produce. Per hand, besides eval_hands.py's fields:

    v0          the critic at the common reset state (no rollout needed beyond the scene's reset)
    v_1s, v_4s  the critic averaged over the first 60 / 240 control steps (1 s / 4 s; envs that end earlier count
                their steps so far)
    v_episode   the critic averaged over the first episode
    goals_1s, goals_4s   goals reached by 1 s / 4 s (a truncated explicit eval, for comparison)

V is the denormalised critic output (rl_games' model returns denorm_value in eval mode), in the training reward's
units: shaping included, so it is a predictor to validate, not a fitness.
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs-per-hand", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--no-finger-table-safe", action="store_true", help="the pre-7-Nov workaround")
    ap.add_argument("--wrist-z", type=float, default=None, help="override the wrist start height (m)")
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
    manifest = json.loads(Path(args.manifest).read_text())
    # grammar specs share one template articulation; a registered robot (a vendor URDF with mesh colliders) cannot be
    # placed on the template, so such a manifest runs as multi:finger: -- the same finger-major 6 x 5 token grid,
    # an articulation per hand (6 Nov, the Wuji v2 zero-shot test)
    def _grammar(spec):                                  # a grammar design (template-able), not a vendor spec JSON
        if not spec.endswith(".spec.json"):
            return False
        p = Path(spec) if Path(spec).is_absolute() else REPO / spec
        return "design" in json.loads(p.read_text())
    if all(_grammar(h["spec"]) for h in manifest["hands"]):
        ref = f"multi:template:{args.manifest}"
    else:
        ref = "multi:finger:" + "+".join(h["spec"] for h in manifest["hands"])
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = args.envs_per_hand * len(manifest["hands"])
    cfg.reset.stagger_initial_episodes = False
    if ref.startswith("multi:finger:") and args.no_finger_table_safe:   # fixed 7 Nov (slot_link_transforms)
        cfg.reset.table_safe = False
    if args.table_clearance is not None:
        cfg.reset.table_clearance = args.table_clearance
    if args.wrist_z is not None:
        pose = list(cfg.wrist.default_pose)
        pose[2] = args.wrist_z
        cfg.wrist.default_pose = tuple(pose)
    cfg.seed = args.seed
    t0 = time.time()
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    specs = [h["spec"] for h in manifest["hands"]]
    if [s.name for s in env.hand_set.specs] != specs:
        raise RuntimeError("the scene's hands are not the manifest's, in order")
    acfg = full["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    acfg["params"]["network"]["robot_spec"] = ref
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"train": acfg}, f)
    player = RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], f.name, args.checkpoint,
                      device=env.device, sapg_expl_coef=0.0, num_envs=env.num_envs)
    print(f"[eval_values] {len(specs)} hands x {args.envs_per_hand} envs ready in {time.time() - t0:.0f} s", flush=True)

    n, dev = env.num_envs, env.device
    hand = env.hand_of_env.long()
    z = lambda: torch.zeros(n, device=dev)  # noqa: E731
    open_ = torch.ones(n, dtype=torch.bool, device=dev)
    goals, length, reward_sum = z(), z(), z()
    v0 = None
    v_sum = {"1s": z(), "4s": z(), "episode": z()}
    v_len = {"1s": z(), "4s": z(), "episode": z()}
    goals_at = {"1s": z(), "4s": z()}
    horizon = {"1s": 60, "4s": 240}
    t1, steps = time.time(), 0
    with torch.no_grad():
        while steps < args.max_steps:
            act, v = player.get_action_and_value(obs["policy"], deterministic_actions=False)
            v = v.float()
            if v0 is None:
                v0 = v.clone()
            o = open_.float()
            for k in v_sum:
                w = o if k == "episode" else o * float(steps < horizon[k])
                v_sum[k] += w * v
                v_len[k] += w
            obs, rew, term, trunc, extras = env.step(act)
            steps += 1
            reward_sum += o * rew
            length += o
            done = (term | trunc) & open_
            if done.any():
                goals = torch.where(done, extras["episode_final"]["successes"].float(), goals)
                open_ &= ~done
            for k, hz in horizon.items():
                if steps == hz:      # goals so far: finished envs' final count, open envs' running count
                    goals_at[k] = torch.where(open_, env.successes.float(), goals)
            if steps % 60 == 0 and not bool(open_.any()):
                break
    goals = torch.where(open_, env.successes.float(), goals)
    for k, hz in horizon.items():
        if steps < hz:
            goals_at[k] = goals
    out = {}
    for h, spec in enumerate(specs):
        m = hand == h
        k = float(m.sum())
        out[spec] = {"envs": int(k), "goals_per_episode": float(goals[m].sum()) / k,
                     "reward_per_step": float(reward_sum[m].sum()) / max(float(length[m].sum()), 1.0),
                     "episode_steps": float(length[m].sum()) / k, "v0": float(v0[m].mean()),
                     **{f"v_{w}": float(v_sum[w][m].sum()) / max(float(v_len[w][m].sum()), 1.0) for w in v_sum},
                     **{f"goals_{w}": float(goals_at[w][m].mean()) for w in goals_at}}
    Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "envs_per_hand": args.envs_per_hand,
                                          "seed": args.seed, "steps": steps, "hands": out}, indent=1))
    print(f"[eval_values] {n} envs done in {steps} steps, {time.time() - t1:.0f} s; wrote {args.out}", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
