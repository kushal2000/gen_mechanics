"""Zero-shot transfer of v4 policies to the commercial hands (14 Nov 2026): the UNCHANGED vendor URDFs
(<hand>_left_uniform_handonly: uniform dynamics, 5 rad/s), no adapters. A v4 policy sees no joint limits and reads
joint angles and acts on one fixed range (-30..90 deg) on every hand -- its training config, replayed exactly from
the policy's training run -- so a vendor hand gets the same treatment as a training design; targets past a vendor
joint's own limits are clamped to them.

One Kit process per hand (a scene holds one hand): --envs envs (100), SAMPLED actions from the SAPG leader (stochastic
evals), every env's FIRST episode only, all envs from a common reset, table-safe finger resets. Per hand and policy:
goals / episode, the share of episodes with at least one goal, how episodes ended, mean length.

    .venv_isaacsim/bin/python experiments/14nov_analyze_v4/zero_shot_commercial.py \
        [--hands wuji2 tesollo xhand leap dex3 allegro shadow] [--policies k5_none:48 k5_global:48 hod:48] [--envs 100]

Results: experiments/14nov_analyze_v4/results/zero_shot_commercial.json (merged across calls), and a table printed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "debug_outputs/coevolve/v4"
HANDS = ("wuji2", "tesollo", "xhand", "leap", "dex3", "allegro", "shadow")
REASONS = ("max_successes", "timeout", "dropped", "fell", "away", "off_course")
RESULTS = Path(__file__).resolve().parent / "results/zero_shot_commercial.json"


def resolve(policy: str) -> dict:
    if policy.startswith("ckpt="):                        # ckpt=<label>=<policy.pth>=<training run dir> (17 Nov)
        _, label, pth, tr = policy.split("=", 3)
        return {"label": label, "policy": str(REPO / pth), "train_run": str(REPO / tr)}
    return _resolve_run(policy)


def _resolve_run(policy: str) -> dict:
    """[campaign/]run:gen (gen 'latest' = the newest with a policy) -> {label, policy path, training run dir};
    campaign default v4 (CAMPAIGN), e.g. v5/k5_none:12."""
    run, g = policy.split(":")
    rd = (CAMPAIGN.parent / run) if "/" in run else (CAMPAIGN / run)
    if g == "latest":
        g = str(max(int(p.parent.name[4:]) for p in rd.glob("gen_*/policy.pth")))
    st = json.loads((rd / "state.json").read_text())
    return {"label": f"{run}:{int(g)}", "policy": str(rd / f"gen_{int(g):04d}/policy.pth"),
            "train_run": str(REPO / st["train_runs"][str(int(g))])}


def hand_ref(h: str) -> str:
    return h if (h.endswith(".spec.json") or "_handonly" in h) else f"{h}_left_uniform_handonly"


# --- one hand, in its own Kit process ----------------------------------------------------------------------------
def worker(args) -> None:
    global RESULTS
    if args.results:
        RESULTS = Path(args.results).resolve()
    sys.path.insert(0, str(REPO))
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app  # noqa: F841
    out = {}
    try:
        out = _evaluate(args)
    except Exception:
        import traceback
        out = {"error": traceback.format_exc()[-3000:]}
    Path(args.result).write_text(json.dumps(out))
    os._exit(0)                                         # app.close() can hang


def _evaluate(args) -> dict:
    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg

    pols = [resolve(p) for p in args.policies]
    full = OmegaConf.to_container(OmegaConf.load(Path(pols[0]["train_run"]) / ".hydra/config.yaml"), resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    ref = f"multi:finger:{hand_ref(args.hand)}"
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = args.envs
    cfg.reset.stagger_initial_episodes = False
    cfg.reset.table_safe = True
    cfg.log.group_stats = False
    cfg.log.design_stats = False
    cfg.seed = args.seed
    if args.collider:
        cfg.hand.collider_type = args.collider
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    n, n_act = env.num_envs, env.action_space.shape[-1]
    res = {"hand": args.hand, "ref": ref, "envs": n, "hand_joints": int(env.joint_enabled[0].sum()), "policies": {}}
    for p in pols:
        acfg = OmegaConf.to_container(OmegaConf.load(Path(p["train_run"]) / ".hydra/config.yaml"),
                                      resolve=True)["agent"]
        acfg["params"]["config"]["multi_gpu"] = False
        acfg["params"]["network"]["robot_spec"] = ref
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump({"train": acfg}, f)
        f.close()
        player = RlPlayer(obs["policy"].shape[-1], n_act, f.name, p["policy"], device=env.device,
                          sapg_expl_coef=0.0, num_envs=n)
        obs, _ = env.reset()
        goals = torch.full((n,), float("nan"), device=env.device)
        # behaviour over each env's first episode (15 Nov): the nearest fingertip to the pen, the pen's lift above its
        # resting height, the pen's distance to its goal -- which stage fails: reach, grasp/lift or reorient
        tip_min = torch.full((n,), float("inf"), device=env.device)
        lift_max = torch.zeros(n, device=env.device)
        goal_min = torch.full((n,), float("inf"), device=env.device)
        live = torch.ones(n, dtype=torch.bool, device=env.device)
        rec_steps = {int(x) for x in args.record.split(",")} if args.record else set()
        poses = {}

        def snap(t):                                      # env 0's pose (render_rollout.py's format), 17 Nov
            names, cols = env.robot.real_joint_names(0), env.robot.real_slots(0)
            jp = env.robot.data.joint_pos[0]
            pb = env.robot.data.body_state_w[0, env.palm_body, :7].clone()
            pb[:3] -= env.scene.env_origins[0]
            pp, pq = env.pen_pose()
            poses[str(t)] = {"q": {nm: float(jp[c]) for nm, c in zip(names, cols)}, "palm": pb.tolist(),
                             "pen": pp[0].tolist(), "pen_quat": pq[0].tolist(), "table_top": float(cfg.table.top_height)}
        if 0 in rec_steps:
            snap(0)
        length = torch.zeros(n, device=env.device)
        why = {r: torch.zeros(n, dtype=torch.bool, device=env.device) for r in REASONS}
        t0, steps = time.time(), 0
        while bool(goals.isnan().any()) and steps < args.max_steps:
            with torch.no_grad():
                act = player.get_normalized_action(obs["policy"], deterministic_actions=False)
            obs, _, term, trunc, extras = env.step(act)
            pen_pos, _ = env.pen_pose()
            tips = env.fingertips()
            d_tip = (tips - pen_pos.unsqueeze(1)).norm(dim=-1).amin(dim=1)
            tip_min = torch.where(live, torch.minimum(tip_min, d_tip), tip_min)
            lift_max = torch.where(live, torch.maximum(lift_max, pen_pos[:, 2] - env.pen_rest_height), lift_max)
            goal_min = torch.where(live, torch.minimum(goal_min, env.goal_dist), goal_min)
            live &= ~(term | trunc)
            if steps in rec_steps and bool(live[0]):
                snap(steps)
            steps += 1
            done = (term | trunc) & goals.isnan()
            if bool(done.any()):
                fin = extras["episode_final"]
                goals = torch.where(done, fin["successes"].float(), goals)
                length = torch.where(done, torch.full_like(length, steps), length)
                for r in REASONS:
                    why[r] |= done & fin[f"done_{r}"].bool()
        if rec_steps:
            rd = RESULTS.parent / "rollouts"
            rd.mkdir(parents=True, exist_ok=True)
            tag = f"{Path(args.hand).name.replace('.spec.json', '')}__{p['label'].replace('/', '_').replace(':', '_g')}"
            (rd / f"{tag}.json").write_text(json.dumps({"hands": [hand_ref(args.hand)], "poses": {"0": poses}}))
        unfinished = goals.isnan()
        goals = torch.where(unfinished, env.successes.float(), goals)      # scored as far as they got
        g = goals.cpu()
        res["policies"][p["label"]] = {
            "goals_per_episode": float(g.mean()), "goals_sem": float(g.std() / max(1, n) ** 0.5),
            "share_with_goal": float((g >= 1).float().mean()), "max_goals": float(g.max()),
            "episode_steps": float(torch.where(unfinished, float(steps), length).mean()),
            "unfinished": float(unfinished.float().mean()),
            **{f"done_{r}": float(why[r].float().mean()) for r in REASONS},
            "tip_to_pen_min_mm": float(tip_min.mean()) * 1000, "pen_lift_max_mm": float(lift_max.mean()) * 1000,
            "share_lifted_2cm": float((lift_max > 0.02).float().mean()), "goal_dist_min_mm": float(goal_min.mean()) * 1000,
            "steps": steps, "seconds": round(time.time() - t0, 1)}
        print(f"[zero-shot] {args.hand} {p['label']}: {float(g.mean()):.2f} goals/ep "
              f"({100 * float((g >= 1).float().mean()):.0f}% >= 1), {steps} steps", flush=True)
    return res


# --- the driver ---------------------------------------------------------------------------------------------------
def main(args) -> None:
    global RESULTS
    if args.results:
        RESULTS = Path(args.results).resolve()
    pols = [resolve(p)["label"] for p in args.policies]
    print(f"[zero-shot] policies {pols}; hands {args.hands}", flush=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    all_res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for h in args.hands:
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        cmd = [sys.executable, __file__, "--worker", "--hand", h, "--result", tmp.name, "--envs", str(args.envs),
               "--max-steps", str(args.max_steps), "--seed", str(args.seed), "--policies", *args.policies]
        if args.collider:
            cmd += ["--collider", args.collider]
        if args.record:
            cmd += ["--record", args.record, "--results", str(RESULTS)]
        t0 = time.time()
        log = RESULTS.parent / f"logs/{RESULTS.stem}_{Path(h).name.replace('.spec.json', '')}.log"
        log.parent.mkdir(exist_ok=True)
        with open(log, "w") as fh:
            subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                           env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, timeout=args.timeout)
        r = json.loads(Path(tmp.name).read_text() or "{}")
        if "error" in r or not r:
            print(f"[zero-shot] {h}: FAILED ({time.time() - t0:.0f} s; {log})\n{r.get('error', '')[-800:]}", flush=True)
            continue
        all_res.setdefault(h, {"policies": {}})
        all_res[h].update(ref=r["ref"], hand_joints=r["hand_joints"], envs=r["envs"])
        all_res[h]["policies"].update(r["policies"])
        RESULTS.write_text(json.dumps(all_res, indent=1))
        print(f"[zero-shot] {h} done ({time.time() - t0:.0f} s)", flush=True)
    print(f"\n{'hand':10s}" + "".join(f"{p:>16s}" for p in pols))
    for h in args.hands:
        if h in all_res:
            row = all_res[h]["policies"]
            print(f"{h:10s}" + "".join(f"{row[p]['goals_per_episode']:9.2f} ({100 * row[p]['share_with_goal']:3.0f}%)"
                                       if p in row else f"{'-':>16s}" for p in pols))
    print(f"(goals / episode, share of episodes with >= 1 goal) -> {RESULTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", nargs="+", default=list(HANDS))
    ap.add_argument("--policies", nargs="+", default=["k5_none:latest", "k5_global:latest", "hod:latest"])
    ap.add_argument("--envs", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=30000, help="50 goals x 10 s at 60 Hz bounds an episode")
    ap.add_argument("--seed", type=int, default=7400)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--record", default="", help="comma-separated steps at which env 0's pose is saved (rollouts/)")
    ap.add_argument("--collider", default="", help="hand.collider_type override (convex_hull / convex_decomposition)")
    ap.add_argument("--results", default="", help="results JSON (default results/zero_shot_commercial.json)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--hand", default="", help=argparse.SUPPRESS)
    ap.add_argument("--result", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    worker(a) if a.worker else main(a)
