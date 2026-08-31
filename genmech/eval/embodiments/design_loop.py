"""Training-free design loop: evaluate, select, mutate, repeat.

    python -m genmech.eval.embodiments.design_loop --iters 10 --seeds 3

Per iteration and per fitness arm:

    1. score every design in the current population, --seeds independent env
       seeds x 6,000 steps, in parallel SLURM jobs
    2. keep the top 4,096 by the seed-averaged score
    3. give each elite one child per operator (5 of them)
    4. the next population is 4,096 elites + 20,480 children = 24,576

THE POLICY NEVER CHANGES. That is the point -- it isolates how far the design
space alone can be pushed against a fixed controller -- and it is also the main
threat to the result: each generation drifts further from what the policy was
trained on, so a gain can be the population learning to suit this controller
rather than becoming better hardware. The compounding is invisible to seed
averaging, because it is bias, not variance. Read the trajectory of the arms
against each other, not the absolute numbers.

TWO ARMS RUN IN LOCKSTEP, selecting on different fitness:

    nominal   DR off -- raw competence
    wrench    dr_sweep/wrench_p0.3 -- competence under disturbance

They share iteration boundaries so the comparison is paired. At 3 seeds the
seed-averaged score has reliability ~0.76 (Spearman-Brown from a measured
single-run 0.51), so selection response is ~0.76 of the achievable maximum.

Layout, one directory per iteration so nothing is overwritten:

    assets/urdf/generated/population/loop/<arm>/iter<NN>/manifest.json
    results/loop/<arm>/iter<NN>/seed<K>/       raw eval output
    results/loop/<arm>/iter<NN>/fitness.csv    seed-averaged score
    results/loop/<arm>/summary.json            per-iteration statistics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path("/share/portal/kk837/gen_mechanics")
SUB = REPO / "experiments/eval_population.sub"
SOURCE_RUN = REPO / "train_dir/gen_mechanics/multi_embodiment_control/mec_population24k_ft_seed0_2026-08-18_16-06-04"
CKPT = SOURCE_RUN / "0_pose_reach_sapg/nn/last_0_pose_reach_sapg_ep_120000_rew__6439.6465_.pth"
BASE_POP = REPO / "assets/urdf/generated/population/seed_0003/manifest.json"
OPERATORS = ("palm", "scale", "mounting", "num_joints_up", "num_joints_down")
ARMS = {"nominal": "", "wrench": "dr_sweep/wrench_p0.3"}
TOP, TOTAL, FAM = 4096, 24576, 6
ALPHA = 0.1


# --- SLURM ----------------------------------------------------------------

def submit_eval(pop: Path, seed: int, out: Path, condition: str) -> str:
    env = dict(os.environ,
               RUN_DIR=str(SOURCE_RUN), CKPT=str(CKPT), OUT_DIR=str(out),
               POPULATION_PATH=str(pop), ENV_SEED=str(seed),
               STEP_BUDGET="6000", GOALS="50", DR="off",
               SUCCESS_TOLERANCE="0.01", TAG=f"loop_s{seed}")
    if condition:
        env["CONDITION"] = condition
    jid = subprocess.run(["sbatch", "--parsable", str(SUB)], env=env,
                         capture_output=True, text=True, check=True).stdout.strip()
    return jid.split(";")[0]


def wait_for(jids: list[str], poll: int = 60, limit_h: float = 4.0) -> None:
    t0 = time.time()
    while time.time() - t0 < limit_h * 3600:
        out = subprocess.run(["squeue", "-j", ",".join(jids), "-h", "-o", "%i"],
                             capture_output=True, text=True).stdout.split()
        if not out:
            return
        print(f"    waiting on {len(out)}/{len(jids)} "
              f"({(time.time()-t0)/60:.0f} min)", flush=True)
        time.sleep(poll)
    raise TimeoutError(f"jobs {jids} did not finish within {limit_h} h")


# --- fitness ---------------------------------------------------------------

def read_goals(d: Path) -> list[float]:
    rows = list(csv.DictReader(open(d / "per_design.csv")))
    rows.sort(key=lambda r: int(r["design_index"]))
    return [float(r["total_goals"]) for r in rows]


def aggregate(dirs: list[Path], out: Path) -> list[float]:
    cols = [read_goals(d) for d in dirs]
    n = len(cols[0])
    mean = [sum(c[i] for c in cols) / len(cols) for i in range(n)]
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["design_index", "mean_goals", "n_seeds"])
        for i, v in enumerate(mean):
            w.writerow([i, v, len(cols)])
    return mean


# --- mutation (in-process; 20,480 children is ~30 min serial, ~2 min on 16) --

_MUT = None


def _init_worker():
    global _MUT
    from genmech.robots.generated import mutate as M
    _MUT = M


def _mutate_one(args):
    parent_json, op, name, key = args
    from genmech.robots.generated.population import hand_from_json, hand_to_json
    h = hand_from_json(parent_json)
    child, failed, tries = _MUT.mutate_one(
        h, op, ALPHA, random.Random(key), name=name)
    return (hand_to_json(child), failed, tries,
            _MUT._displacement(h, child) if not failed else {})


def build_next(elites: list[dict], it: int, arm: str, pool: ProcessPoolExecutor,
               ) -> list[dict]:
    slots: list[dict | None] = [None] * TOTAL
    for j, e in enumerate(elites):
        slots[j * FAM] = {**e, "source": "elite"}
    for k, op in enumerate(OPERATORS, start=1):
        tasks = [(e["params"], op, f"loop_{arm}_i{it+1:02d}_{op}_{j:05d}",
                  f"{arm}:{it}:{op}:{j}") for j, e in enumerate(elites)]
        nfail = 0
        for j, (params, failed, tries, disp) in enumerate(pool.map(_mutate_one, tasks, chunksize=32)):
            nfail += failed
            slots[j * FAM + k] = {
                "name": tasks[j][2], "params": params, "source": op,
                "parent_name": elites[j]["name"], "rank": j,
                "mutation_failed": bool(failed), "attempts": tries,
                "displacement": disp,
            }
        print(f"    {op:16s} {len(elites)} children, {nfail} failed", flush=True)
    assert all(s is not None for s in slots)
    return slots


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()

    pop_root = REPO / "assets/urdf/generated/population/loop"
    res_root = REPO / "results/loop"
    for arm in ARMS:
        d = pop_root / arm / "iter00"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "manifest.json").exists():
            shutil.copy(BASE_POP, d / "manifest.json")
            print(f"[loop] seeded {arm}/iter00 from the original seed-3 population")

    summaries = {arm: [] for arm in ARMS}
    with ProcessPoolExecutor(a.workers, initializer=_init_worker) as pool:
        for it in range(a.start, a.start + a.iters):
            print(f"\n===== iteration {it} =====", flush=True)
            jobs = {}
            for arm, cond in ARMS.items():
                pop = pop_root / arm / f"iter{it:02d}"
                jobs[arm] = []
                for s in range(1, a.seeds + 1):
                    out = res_root / arm / f"iter{it:02d}" / f"seed{s}"
                    out.mkdir(parents=True, exist_ok=True)
                    jobs[arm].append((submit_eval(pop, s, out, cond), out))
                print(f"  {arm}: submitted {[j for j,_ in jobs[arm]]}", flush=True)
            wait_for([j for arm in jobs for j, _ in jobs[arm]])

            for arm in ARMS:
                dirs = [o for _, o in jobs[arm]]
                missing = [d for d in dirs if not (d / "per_design.csv").exists()]
                if missing:
                    raise SystemExit(f"{arm} iter {it}: eval produced no CSV in {missing}")
                fit = aggregate(dirs, res_root / arm / f"iter{it:02d}" / "fitness.csv")
                cur = json.loads((pop_root / arm / f"iter{it:02d}" / "manifest.json").read_text())["hands"]
                order = sorted(range(len(fit)), key=lambda i: -fit[i])[:TOP]
                elites = [{"name": cur[i]["name"], "params": cur[i]["params"],
                           "fitness": fit[i]} for i in order]
                srt = sorted(fit, reverse=True)
                rec = {"iteration": it, "mean": sum(fit)/len(fit),
                       "elite_mean": sum(srt[:TOP])/TOP, "best": srt[0],
                       "cutoff": srt[TOP-1], "n_seeds": a.seeds,
                       "distinct_designs": len({h["name"] for h in cur})}
                summaries[arm].append(rec)
                print(f"  {arm}: pop mean {rec['mean']:.2f}  elite mean "
                      f"{rec['elite_mean']:.2f}  best {rec['best']:.1f}  "
                      f"cutoff {rec['cutoff']:.1f}", flush=True)

                nxt = pop_root / arm / f"iter{it+1:02d}"
                nxt.mkdir(parents=True, exist_ok=True)
                slots = build_next(elites, it, arm, pool)
                (nxt / "manifest.json").write_text(json.dumps({
                    "version": 1, "kind": "loop_generation", "iteration": it + 1,
                    "arm": arm, "fitness": ARMS[arm] or "nominal (DR off)",
                    "seeds": a.seeds, "top": TOP, "alpha": ALPHA,
                    "operators": list(OPERATORS), "count": len(slots),
                    "parent_iteration": it, "seed": 3,
                    "slot_layout": "family j at slots 6j..6j+5; 6j is the elite",
                    "hands": slots}))
                (res_root / arm / "summary.json").write_text(json.dumps(summaries[arm], indent=2))
                print(f"  {arm}: wrote {nxt/'manifest.json'}", flush=True)
    print("\n[loop] done")


if __name__ == "__main__":
    main()
