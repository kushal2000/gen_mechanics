"""Collect eval results into one table.

Reads every ``result.json`` under a sweep directory and emits a tidy CSV plus a
printed summary.

The headline number is **retention** — a condition's score divided by the same
hand's nominal score — computed on **full-completion rate** rather than the
chain-length mean, because the chain-length distribution is bimodal (episodes
mostly score either zero goals or all of them) and its mean therefore describes
almost no individual episode. Absolute scores are not comparable across hands: the
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
    nominal = {
        r["robot_spec"]: r["goal_pct_mean"]
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
            r["gap"] = r["goal_pct_mean"] - base
        else:
            r["retention"] = r["goal_pct_mean"] / base
            r["gap"] = r["goal_pct_mean"] - base
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
                ret = r["retention_complete"]
                ret_s = "  n/a" if math.isnan(ret) else f"{ret * 100:5.0f}%"
                line += (f"{r['full_completion_rate'] * 100:>8.0f}%"
                         f"{r['zero_goal_rate'] * 100:>7.0f}%"
                         f"{ret_s:>10}")
        print(line)
    print("-" * len(header))
    print(f"{'':{width}}" + "".join(f"{'complete   zero  retention':>26}" for _ in robots))
    print("\n(retention is on completion rate, not the chain-length mean: the "
          "chain-length\n distribution is bimodal, so its mean describes almost no "
          "individual episode.)")

    unfinished = sum(r["term_unfinished"] for r in rows)
    if unfinished:
        print(f"\nWARNING: {unfinished} env-episodes hit the step cap without "
              f"terminating; their goal_pct is a lower bound.")

    thin = [r for r in rows if r["goal_pct_sem"] > 5.0]
    if thin:
        print(f"\nNOTE: {len(thin)} condition(s) have SEM > 5 goal_pct points. "
              f"Goals are sampled live, so each hand draws its own sequences; "
              f"raise --num_envs before trusting differences that small.")
        for r in thin[:5]:
            print(f"  {r['robot_spec']:20s} {r['condition_id']:28s} "
                  f"{r['goal_pct_mean']:.1f} ± {r['goal_pct_sem']:.1f} "
                  f"(n={r['num_envs']})")


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
