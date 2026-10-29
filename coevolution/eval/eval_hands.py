"""Per-hand fitness of a shared policy over a hand set: the evolution loop's evaluation (coevolution/evolution).

    .venv_isaacsim/bin/python coevolution/eval/eval_hands.py --manifest <gen_N/eval_manifest.json> \
        --checkpoint <model.pth> --train-run <a training run dir (its .hydra/config.yaml)> --out <fitness.json>

One olympics scene of every hand in the manifest (``multi:template:``, the manifest's pinned grid), ``--envs-per-hand``
parallel envs each (10 by default), the env and the agent exactly as the training run was configured, the
checkpoint's policy with SAMPLED actions (stochastic evals, memory: greedy stalls), every env pinned to the training
task. All envs start from a common reset; each env contributes its FIRST episode only (the user's call, 29 Oct), and
the eval stops once every env has finished one (or at ``--max-steps``: 50 goals x 10 s at 60 Hz = 30,000 bounds an
episode). Per hand:

    goals_per_episode   the mean goals of the hand's envs' first episodes: the selection scalar (the training
                        metric, as wandb's episode_final/successes)
    reward_per_step     the training reward per step over those episodes (shaping included): the tie-breaker when
                        hands score no goals (early generations, hard hands), where goal counts cannot rank them
    done_<reason>       the fraction of those episodes that ended by each reason (max_successes, timeout, dropped,
                        fell, away, off_course); ``unfinished`` the fraction cut off by --max-steps (scored as far as
                        they got)
    episode_steps       their mean length; envs

Output JSON: ``{"checkpoint", "envs_per_hand", "steps", "hands": {<spec path as in the manifest>: {...}}}``.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-run", required=True, help="a training run dir holding .hydra/config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs-per-hand", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
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

    hydra = next(Path(args.train_run).rglob(".hydra/config.yaml"), None)
    if hydra is None:
        raise FileNotFoundError(f"no .hydra/config.yaml under {args.train_run}")
    full = OmegaConf.to_container(OmegaConf.load(hydra), resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    manifest = json.loads(Path(args.manifest).read_text())
    ref = f"multi:template:{args.manifest}"
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = args.envs_per_hand * len(manifest["hands"])
    cfg.reset.stagger_initial_episodes = False
    cfg.seed = args.seed
    t0 = time.time()
    env = InHandOlympicsEnv(cfg)
    if len(cfg.task.names) != 1:
        raise ValueError(f"fitness is defined on one task; the run trains {cfg.task.names}")
    obs, _ = env.reset()
    hs = env.hand_set
    specs = [h["spec"] for h in manifest["hands"]]
    if [s.name for s in hs.specs] != specs:
        raise RuntimeError("the scene's hands are not the manifest's, in order")

    acfg = full["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    acfg["params"]["network"]["robot_spec"] = ref        # the eval set's template (same pinned grid, same shapes)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"train": acfg}, f)
    n_act = env.action_space.shape[-1]
    player = RlPlayer(obs["policy"].shape[-1], n_act, f.name, args.checkpoint, device=env.device,
                      sapg_expl_coef=0.0, num_envs=env.num_envs)
    print(f"[eval_hands] {len(specs)} hands x {args.envs_per_hand} envs ready in {time.time() - t0:.0f} s", flush=True)

    n, dev = env.num_envs, env.device
    hand = env.hand_of_env.long()
    reasons = ("max_successes", "timeout", "dropped", "fell", "away", "off_course")
    open_ = torch.ones(n, dtype=torch.bool, device=dev)       # still in its first episode
    goals = torch.zeros(n, device=dev)
    length = torch.zeros(n, device=dev)
    reward_sum = torch.zeros(n, device=dev)
    reason = {r: torch.zeros(n, device=dev) for r in reasons}
    t1, steps = time.time(), 0
    with torch.no_grad():
        while steps < args.max_steps:
            act = player.get_normalized_action(obs["policy"], deterministic_actions=False)
            obs, rew, term, trunc, extras = env.step(act)
            steps += 1
            o = open_.float()
            reward_sum += o * rew
            length += o
            done = (term | trunc) & open_
            if done.any():
                final = extras["episode_final"]
                goals = torch.where(done, final["successes"].float(), goals)
                for r in reasons:
                    reason[r] = torch.where(done, final[f"done_{r}"].float(), reason[r])
                open_ &= ~done
            if steps % 60 == 0 and not bool(open_.any()):
                break
    goals = torch.where(open_, env.successes.float(), goals)   # cut off by --max-steps: as far as it got
    out = {}
    for h, spec in enumerate(specs):
        m = hand == h
        k = float(m.sum())
        out[spec] = {"envs": int(k), "goals_per_episode": float(goals[m].sum()) / k,
                     "reward_per_step": float(reward_sum[m].sum()) / max(float(length[m].sum()), 1.0),
                     "episode_steps": float(length[m].sum()) / k, "unfinished": float(open_[m].float().sum()) / k,
                     **{f"done_{r}": float(reason[r][m].sum()) / k for r in reasons}}
    Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "envs_per_hand": args.envs_per_hand,
                                          "steps": steps, "hands": out}, indent=1))
    best = max(out.values(), key=lambda v: v["goals_per_episode"])
    print(f"[eval_hands] first episodes of {n} envs done in {steps} steps, {time.time() - t1:.0f} s; best "
          f"{best['goals_per_episode']:.2f} goals/episode; wrote {args.out}", flush=True)
    import os
    os._exit(0)          # Kit can hang on a clean close (seen 29 Oct); everything is written


if __name__ == "__main__":
    main()
