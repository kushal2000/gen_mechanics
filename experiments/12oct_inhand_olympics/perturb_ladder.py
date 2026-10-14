"""Perturbation ladder: raise pen_lift's push and twist scales (env.perturb.force_scale / torque_scale) 10x at a time
on one run's policy until it fails, i.e. saturates at a low goal count. Each stage fine-tunes the last stage's latest
checkpoint (weights only) with 10x the scales; a stage is saturated when the leader block's goals per episode stop
rising. Stops at the first stage that saturates at or below --stop goals per episode (pen_lift's pushes start after
two goals, the pickup and one full hold, so a policy that cannot hold against them sits near 2).

    nohup .venv_isaacsim/bin/python experiments/12oct_inhand_olympics/perturb_ladder.py --base-job 255933 \\
        --seed 103 --hand wuji2_left_uniform_handonly_only_thumb_index > <out>/ladder.log 2>&1 &
    # state and log: debug_outputs/train_logs/12oct_inhand_olympics/perturb_ladder_s<seed>/

Only cancels stage jobs it submitted itself, once the next stage is submitted; the base run is left alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import subprocess
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
LOGS = REPO / "debug_outputs/train_logs/12oct_inhand_olympics"
SUB = REPO / "experiments/12oct_inhand_olympics/olympics_sapg_mlp.sub"
LEADER = "episode_final/successes_per_block/block_5"


def log(msg: str) -> None:
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def run_dir(job: str) -> pathlib.Path | None:
    d = glob.glob(str(LOGS / f"0_*_{job}"))
    return pathlib.Path(d[0]) if d else None


def leader_goals(job: str) -> list[float]:
    """The leader block's goals per finished episode, one value per epoch."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    d = run_dir(job)
    ev = glob.glob(str(d / "0_*/summaries/events*")) if d else []
    if not ev:
        return []
    ea = EventAccumulator(ev[0], size_guidance={"scalars": 0})
    ea.Reload()
    if LEADER not in ea.Tags()["scalars"]:
        return []
    return [e.value for e in sorted(ea.Scalars(LEADER), key=lambda e: e.step)]


def latest_checkpoint(job: str) -> str:
    nn = glob.glob(str(run_dir(job) / "0_*/nn"))[0]
    last = glob.glob(f"{nn}/last_*_ep_*.pth")
    if last:
        return max(last, key=lambda p: int(re.search(r"_ep_(\d+)_", p).group(1)))
    return max(glob.glob(f"{nn}/*.pth"), key=os.path.getmtime)


def job_alive(job: str) -> bool:
    out = subprocess.run(["squeue", "-j", job, "-h", "-o", "%T"], capture_output=True, text=True).stdout.strip()
    return out in ("PENDING", "RUNNING", "CONFIGURING")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-job", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--force", type=float, default=20.0, help="the base run's force_scale")
    ap.add_argument("--torque", type=float, default=2.0, help="the base run's torque_scale")
    ap.add_argument("--factor", type=float, default=10.0)
    ap.add_argument("--stop", type=float, default=2.0, help="stop once a stage saturates at or below this")
    ap.add_argument("--max-stages", type=int, default=4, help="raises after the base run")
    ap.add_argument("--stage-epochs", type=int, default=2000)
    ap.add_argument("--min-epochs", type=int, default=500, help="before a stage can count as saturated")
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--poll", type=int, default=600, help="seconds")
    ap.add_argument("--project", default="gen_mechanics_inhand_olympics_carried")
    args = ap.parse_args()

    out = LOGS / f"perturb_ladder_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "stages": [{"job": args.base_job, "force": args.force, "torque": args.torque, "plateau": None}]}
    save = lambda: state_path.write_text(json.dumps(state, indent=1))  # noqa: E731
    save()
    w = args.window
    while True:
        st = state["stages"][-1]
        g, alive = leader_goals(st["job"]), job_alive(st["job"])
        n = len(g)
        cur = sum(g[-w:]) / len(g[-w:]) if n >= w else float("nan")
        prev = sum(g[-2 * w:-w]) / w if n >= 2 * w else float("nan")
        log(f"stage {len(state['stages']) - 1} job {st['job']} force {st['force']:g} torque {st['torque']:g}: "
            f"{n} epochs, goals/ep last {w} {cur:.2f}, previous {w} {prev:.2f}, alive {alive}")
        if not alive and n < 2 * w:
            log("the stage's job ended before it could saturate; stopping")
            break
        saturated = (not alive) or (n >= max(args.min_epochs, 2 * w) and (cur >= 45 or cur - prev < max(0.05 * prev, 0.3)))
        if saturated:
            st["plateau"] = cur
            save()
            log(f"stage saturated at {cur:.2f} goals/ep")
            if len(state["stages"]) > 1 and cur <= args.stop:
                log(f"thumb+index fails at force_scale {st['force']:g} / torque_scale {st['torque']:g}: "
                    f"saturates at {cur:.2f} goals/ep; done")
                break
            if len(state["stages"]) > args.max_stages:
                log("max stages reached; done")
                break
            ckpt = latest_checkpoint(st["job"])
            f, t = st["force"] * args.factor, st["torque"] * args.factor
            env = dict(os.environ, TASKS="pen_lift", SEED=str(args.seed), PEN_SHAPE="box", HAND=args.hand,
                       WANDB_PROJECT=args.project, EPOCHS=str(args.stage_epochs), CHECKPOINT=ckpt,
                       CHECKPOINT_MODE="weights", RUN_TAG=f"stagger_f{f:g}_t{t:g}",
                       EXTRA_HYDRA=f"env.reset.stagger_initial_episodes=True env.perturb.force_scale={f} "
                                   f"env.perturb.torque_scale={t}")
            job = subprocess.run(["sbatch", "--parsable", f"--job-name=OC_lift_s{args.seed}_f{f:g}", str(SUB)],
                                 env=env, capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()
            log(f"submitted {job}: force {f:g} torque {t:g} from {ckpt}")
            if len(state["stages"]) > 1 and alive:
                subprocess.run(["scancel", st["job"]])
                log(f"cancelled the saturated stage job {st['job']}")
            state["stages"].append({"job": job, "force": f, "torque": t, "plateau": None, "from": ckpt})
            save()
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
