"""Per-task scores of one in-hand olympics checkpoint: goals in each env's first episode, and how that episode ended,
split by the task the env drew at reset. Sampled actions (SAPG leader block, coefficient 0), as every eval since
12 Oct; the env config is the run's own (its saved .hydra/config.yaml), so the task mix is the run's.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/12oct_inhand_olympics/analysis/eval_tasks.py \\
        --run debug_outputs/train_logs/12oct_inhand_olympics/0_<run> [--envs 768] [--checkpoint <.pth>]
        # -> <run>/eval_tasks_ep<N>.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the run directory (holds .hydra/config.yaml)")
    ap.add_argument("--checkpoint", default="", help="default: the run's highest-epoch last_*.pth")
    ap.add_argument("--envs", type=int, default=768, help="a multiple of 6 (the SAPG blocks)")
    ap.add_argument("--max-seconds", type=float, default=600.0)
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app

    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg

    run = pathlib.Path(args.run)
    ckpt = pathlib.Path(args.checkpoint) if args.checkpoint else max(
        run.glob("*/nn/last_*_ep_*.pth"), key=lambda p: int(re.search(r"_ep_(\d+)_", p.name).group(1)))
    epoch = re.search(r"_ep_(\d+)_", ckpt.name).group(1)
    saved = OmegaConf.load(run / ".hydra/config.yaml")
    env_d = OmegaConf.to_container(saved.env, resolve=True)
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.scene.num_envs = args.envs
    cfg.seed = 0
    cfg.reset.stagger_initial_episodes = False    # the first episode is a whole one in every env
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()

    acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump({"train": acfg}, f)
    f.close()
    player = RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], f.name, str(ckpt),
                      device=env.device, sapg_expl_coef=0.0, num_envs=env.num_envs)

    n, dev = env.num_envs, env.device
    task_of = env.task_id.clone()                 # the task each env drew for its first episode
    first = torch.full((n,), -1.0, device=dev)
    how = [""] * n
    k, max_steps = 0, int(args.max_seconds / env.step_dt)
    while k < max_steps and bool((first < 0).any()):
        with torch.no_grad():
            act = player.get_normalized_action(obs["policy"], deterministic_actions=False)
        before = env.successes.clone()
        obs, _, term, trunc, _ = env.step(act)
        k += 1
        ended = (term | trunc) & (first < 0)
        if ended.any():
            # Successes were zeroed by the reset; the episode's count is the one before it plus this step's hits.
            first[ended] = before[ended] + env.is_success[ended].float()
            for i in ended.nonzero().flatten().tolist():
                how[i] = next((r for r, v in env.termination_reasons.items() if bool(v[i])), "?")
        if k % 600 == 0:
            print(f"[eval_tasks] {k * env.step_dt:.0f} s sim: {int((first < 0).sum())} of {n} envs in their first "
                  f"episode", flush=True)

    out = {"run": str(run), "checkpoint": str(ckpt), "envs": n, "tasks": {}}
    # A multi-hand scene: every task again per hand, as "<task>/<hand>".
    groups = [(task.name, task_of == t) for t, task in enumerate(env.tasks)]
    if env.hand_set is not None:
        groups += [(f"{task.name}/{label}", (task_of == t) & (env.hand_of_env == h))
                   for t, task in enumerate(env.tasks) for h, label in enumerate(env.hand_set.labels)]
    for name, of in groups:
        m = of & (first >= 0)
        ends = [how[i] for i in m.nonzero().flatten().tolist()]
        g = first[m]
        q = torch.quantile(g, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=g.device)) if m.any() else None
        out["tasks"][name] = {
            "envs": int(of.sum()),
            "goals_per_episode_first": float(g.mean()) if m.any() else float("nan"),
            # How the goals spread over envs: are they from a few envs or most of them?
            "goals_quantiles_10_25_50_75_90": [round(float(v), 2) for v in q] if q is not None else [],
            "frac_envs_with_goals": {f">={k}": float((g >= k).float().mean()) for k in (1, 2, 10, 50)} if m.any() else {},
            "ended": {r: ends.count(r) / max(len(ends), 1) for r in sorted(set(ends))},
        }
        print(f"[eval_tasks] {name:14s} {out['tasks'][name]}", flush=True)
    path = run / f"eval_tasks_ep{epoch}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"[eval_tasks] wrote {path}", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
