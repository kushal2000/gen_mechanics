"""Collect eval results into one table.

Reads every ``result.json`` under a sweep directory and emits a tidy CSV plus a
printed summary.

The headline number is **retention** — a condition's score divided by the same
hand's nominal score — computed on **mean goals reached**. That uses every
episode's full information instead of thresholding it away (an episode reaching
9 goals and one reaching 0 are identical under a completion rate), and it needs
no calibration: with a threshold metric the goal ceiling has to be tuned so the
reference policy lands mid-range, and that tuning moves the headline. Absolute scores are not comparable across hands: the
reward is tuned for SHARPA and sums over fingertips and hand DoFs, so a hand can
be uniformly weaker on this task without being worse at *generalizing*
(docs/methodology.md §2, §4). Retention removes that offset by scoring each hand
against itself.

    .venv_isaacsim/bin/python -m genmech.eval.aggregate results/<sweep_id>

No Isaac Sim needed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_results(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("result.json")):
        with open(path) as f:
            r = json.load(f)
        rows.append({
            "robot_spec": r["robot_spec"],
            "hand": r["hand"],
            "axis": r["axis"],
            "condition": r["condition"],
            "condition_id": r["condition_id"],
            "seed": r["seed"],
            "num_envs": r["num_envs"],
            # Results written before goals_reached_* existed carry the same
            # quantity as a percentage of the ceiling, so backfill exactly
            # rather than dropping them.
            "goals_reached_mean": r.get(
                "goals_reached_mean", r["goal_pct_mean"] / 100.0 * r["n_goals"]
            ),
            "goals_reached_sem": r.get(
                "goals_reached_sem", r["goal_pct_sem"] / 100.0 * r["n_goals"]
            ),
            "goals_reached_max": r.get("goals_reached_max", r["n_goals"]),
            "ceiling_rate": r.get("ceiling_rate", float("nan")),
            "goal_pct_mean": r["goal_pct_mean"],
            "goal_pct_sem": r["goal_pct_sem"],
            "full_completion_rate": r["full_completion_rate"],
            "zero_goal_rate": r.get("zero_goal_rate", float("nan")),
            "bimodality": r.get("bimodality", float("nan")),
            "any_goal_rate": r["any_goal_rate"],
            "lift_rate": r["lift_rate"],
            "time_to_first_goal_sec": r["time_to_first_goal_sec"],
            "mean_episode_sec": r["mean_episode_sec"],
            "reward_mean": r["reward_mean"],
            "term_complete": r["termination_counts"]["trajectory_complete"],
            "term_fall": r["termination_counts"]["fall_or_hand_far"],
            "term_timeout": r["termination_counts"]["timeout"],
            "term_unfinished": r["termination_counts"]["unfinished"],
            "checkpoint_sha256": r["policy"]["checkpoint_sha256"][:12],
            "path": str(path.parent.relative_to(root)),
        })
    return rows


def add_retention(rows: list[dict]) -> list[dict]:
    """Attach retention and gap, relative to each robot's own nominal run."""
    # Retention is on mean goals reached: it uses every episode's information
    # rather than thresholding it, and needs no calibration of n_goals.
    nominal = {
        r["robot_spec"]: r["goals_reached_mean"]
        for r in rows if r["axis"] == "nominal"
    }
    # Completion rate is the more robust headline when the chain-length
    # distribution is bimodal, so retention is computed on both.
    nominal_complete = {
        r["robot_spec"]: r["full_completion_rate"]
        for r in rows if r["axis"] == "nominal"
    }
    for r in rows:
        base = nominal.get(r["robot_spec"])
        if base is None:
            r["retention"] = float("nan")
            r["gap"] = float("nan")
        base_c = nominal_complete.get(r["robot_spec"])
        r["retention_complete"] = (
            r["full_completion_rate"] / base_c
            if base_c not in (None, 0.0) else float("nan")
        )
        if base is None:
            pass
        elif base <= 0.0:
            # A hand that scores zero nominally has no meaningful baseline; a
            # ratio here would be an artifact, so leave it undefined rather than
            # print an infinity.
            r["retention"] = float("nan")
            r["gap"] = r["goals_reached_mean"] - base
        else:
            r["retention"] = r["goals_reached_mean"] / base
            r["gap"] = r["goals_reached_mean"] - base
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    import csv

    if not rows:
        return
    cols = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    robots = sorted({r["robot_spec"] for r in rows})
    by = {(r["robot_spec"], r["condition_id"]): r for r in rows}
    conditions = sorted({r["condition_id"] for r in rows},
                        key=lambda c: (c.split("/")[0] != "nominal", c))

    width = max(len(c) for c in conditions) + 2
    header = f"{'condition':{width}}" + "".join(f"{rb:>26}" for rb in robots)
    print("\n" + header)
    print("-" * len(header))
    for cid in conditions:
        line = f"{cid:{width}}"
        for rb in robots:
            r = by.get((rb, cid))
            if r is None:
                line += f"{'-':>26}"
            else:
                ret = r["retention"]
                ret_s = "  n/a" if math.isnan(ret) else f"{ret * 100:5.0f}%"
                line += (f"{r['goals_reached_mean']:>8.2f}"
                         f"±{r['goals_reached_sem']:<5.2f}"
                         f"{ret_s:>11}")
        print(line)
    print("-" * len(header))
    print(f"{'':{width}}" + "".join(f"{'goals reached   retention':>26}" for _ in robots))

    sat = [r for r in rows if r.get("ceiling_rate", 0) > 0.4]
    if sat:
        print(f"\nNOTE: {len(sat)} condition(s) have >40% of episodes at the "
              f"{rows[0].get('goals_reached_max')}-goal ceiling; the mean is "
              f"compressed there and differences are understated.")

    unfinished = sum(r["term_unfinished"] for r in rows)
    if unfinished:
        print(f"\nWARNING: {unfinished} env-episodes hit the step cap without "
              f"terminating; their goal_pct is a lower bound.")

    thin = [r for r in rows if r["goals_reached_sem"] > 0.5]
    if thin:
        print(f"\nNOTE: {len(thin)} condition(s) have SEM > 0.5 goals. Goals are "
              f"sampled live and the object spawns at a random orientation, so "
              f"each hand draws its own episodes; raise --num_envs before "
              f"trusting differences that small.")
        for r in thin[:6]:
            print(f"  {r['robot_spec']:20s} {r['condition_id']:28s} "
                  f"{r['goals_reached_mean']:.2f} ± {r['goals_reached_sem']:.2f} "
                  f"goals (n={r['num_envs']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Sweep directory containing result.json files")
    parser.add_argument("--out", default=None, help="CSV path (default <root>/summary.csv)")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"no such directory: {root}")

    rows = load_results(root)
    if not rows:
        raise SystemExit(f"no result.json files under {root}")
    rows = add_retention(rows)

    out = Path(args.out) if args.out else root / "summary.csv"
    write_csv(rows, out)
    print(f"[aggregate] {len(rows)} results from "
          f"{len({r['robot_spec'] for r in rows})} robot(s) -> {out}")
    print_summary(rows)


if __name__ == "__main__":
    main()
