"""Phase 1 tables: per-metric ranks, robustness retention, Pareto fronts, and a JSON for the page.

    .venv_isaacsim/bin/python experiments/01oct_embodiment_niches/analysis/aggregate.py
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
D = REPO / "debug_outputs/embodiment_niches"

HANDS = ["sharpa", "gen_sharpa", "allegro", "leap", "shadow", "tesollo", "dex3", "xhand", "wuji2"]
CONDS = ["nominal", "cube40", "cube55", "cube65", "light", "heavy", "slippery", "push", "push_hard"]

# metric -> (label, higher is better?, getter)
def _g(*ks):
    def f(d):
        for k in ks:
            d = d.get(k) if isinstance(d, dict) else None
            if d is None:
                return None
        return d
    return f

METRICS = {
    "goals_per_min":      ("goals / min", True, _g("goals_per_min")),
    "goals_per_drop":     ("goals per drop", True, _g("goals_per_drop")),
    "time_per_goal":      ("median s / goal", False, _g("time_per_goal_s", "median")),
    "path_efficiency":    ("path efficiency", True, _g("path_efficiency")),
    "work_per_goal":      ("work / goal (J)", False, _g("work_per_goal_J")),
    "power":              ("mean power (W)", False, _g("mean_power_W_per_env")),
    "target_rate":        ("action rate (rad/s/joint)", False, _g("target_rate_rad_s_per_joint")),
    "speed_capped":       ("time at speed cap", False, _g("joint_speed_saturated_frac")),
    "obj_speed":          ("cube speed (m/s)", False, _g("obj_speed_mean")),
    "obj_acc":            ("cube accel (m/s^2)", False, _g("obj_acc_mean")),
    "palm_dist_std":      ("cube wander (mm, std)", False, lambda d: 1000 * d["palm_dist_std"]),
    "participation":      ("joints used (PR / J)", True, _g("participation_frac")),
}


def load():
    R = {}
    for h in HANDS:
        for c in CONDS:
            f = D / f"{h}__{c}.json"
            if f.is_file() and f.stat().st_size:
                R[(h, c)] = json.loads(f.read_text())
    return R


def pareto(points: dict, hi_a: bool, hi_b: bool) -> list[str]:
    """Non-dominated hands for two metrics."""
    sa, sb = (1 if hi_a else -1), (1 if hi_b else -1)
    out = []
    for h, (a, b) in points.items():
        dom = any(sa * a2 >= sa * a and sb * b2 >= sb * b and (sa * a2 > sa * a or sb * b2 > sb * b)
                  for h2, (a2, b2) in points.items() if h2 != h)
        if not dom:
            out.append(h)
    return out


def main():
    R = load()
    have = sorted({h for h, c in R if c == "nominal"}, key=HANDS.index)
    print(f"{len(R)} results; nominal for {len(have)} hands: {have}")
    if not have:
        sys.exit(0)

    nominal = {h: {m: g(R[(h, "nominal")]) for m, (_, _, g) in METRICS.items()} for h in have}
    ranks = {}
    print("\n== nominal, rank in brackets (1 = best)")
    print(f"{'metric':28s}" + "".join(f"{h:>13s}" for h in have))
    for m, (lab, hi, _) in METRICS.items():
        vals = {h: nominal[h][m] for h in have if nominal[h][m] is not None}
        order = sorted(vals, key=lambda h: vals[h], reverse=hi)
        ranks[m] = {h: order.index(h) + 1 for h in order}
        print(f"{lab:28s}" + "".join(f"{vals[h]:9.3g} ({ranks[m][h]})" if h in vals else f"{'-':>13s}"
                                     for h in have))

    # Robustness: goals/min under each perturbation relative to the hand's own nominal.
    retention = {}
    print("\n== robustness: goals/min relative to nominal (drops/min in brackets)")
    print(f"{'condition':12s}" + "".join(f"{h:>14s}" for h in have))
    for c in CONDS[1:]:
        row = {}
        for h in have:
            if (h, c) in R:
                row[h] = (R[(h, c)]["goals_per_min"] / max(R[(h, "nominal")]["goals_per_min"], 1e-9),
                          R[(h, c)]["drops_per_min"])
        retention[c] = row
        print(f"{c:12s}" + "".join(f"{row[h][0]:8.2f} ({row[h][1]:4.1f})" if h in row else f"{'-':>14s}"
                                   for h in have))

    # Pareto fronts on trade-offs that could define a niche.
    pairs = [("goals_per_min", "work_per_goal"), ("goals_per_min", "goals_per_drop"),
             ("goals_per_min", "obj_acc"), ("work_per_goal", "goals_per_drop"),
             ("path_efficiency", "goals_per_min")]
    fronts = {}
    print("\n== Pareto fronts (non-dominated hands)")
    for a, b in pairs:
        pts = {h: (nominal[h][a], nominal[h][b]) for h in have
               if nominal[h][a] is not None and nominal[h][b] is not None}
        fronts[f"{a}|{b}"] = pareto(pts, METRICS[a][1], METRICS[b][1])
        print(f"{METRICS[a][0]:>20s} vs {METRICS[b][0]:<22s}: {fronts[f'{a}|{b}']}")

    # Who is best at what: the hands that rank first on at least one axis.
    firsts = {}
    for m in METRICS:
        for h, r in ranks[m].items():
            if r == 1:
                firsts.setdefault(h, []).append(METRICS[m][0])
    for c, row in retention.items():
        if row:
            best = max(row, key=lambda h: row[h][0])
            firsts.setdefault(best, []).append(f"robust: {c}")
    print("\n== first place somewhere")
    for h in have:
        print(f"  {h:11s} {firsts.get(h, [])}")

    out = {"hands": have, "conditions": CONDS, "metrics": {m: {"label": l, "higher_better": hi}
                                                           for m, (l, hi, _) in METRICS.items()},
           "nominal": nominal, "ranks": ranks, "retention": retention, "fronts": fronts,
           "firsts": firsts,
           "raw": {f"{h}__{c}": R[(h, c)] for (h, c) in R}}
    (D / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {D/'summary.json'}")


if __name__ == "__main__":
    main()
