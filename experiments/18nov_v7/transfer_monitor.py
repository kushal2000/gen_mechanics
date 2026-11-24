"""Transfer monitor for a co-evolution campaign (15 Nov 2026): every new generation's policy, scored zero-shot on three
hand sets, so specialisation shows the generation it starts (v4's MAP-Elites policy fell to 0 on everything outside
its own archive, even its gen-0 ancestors):

    vendor     the commercial hands, unchanged vendor URDFs (no adapters)          -- the target
    replica    the same hands rebuilt in the campaign's grammar (commercial_snapped) -- in-grammar, never trained on
    seed       gen-0 seed designs from the vendor hands' cells (2 each)             -- the policy's own ancestors

experiments/14nov_analyze_v4/zero_shot_commercial.py does the scoring (one Kit process per hand, sampled actions,
first episode per env, the policy's own training config). Every --interval seconds the monitor collects the policies
not yet scored (each run's generations; --stride per run), scores them together (one scene boot per hand for the
whole batch), and rewrites summary.csv + transfer.png: mean goals / episode per hand set, per run, by generation.

    .venv_isaacsim/bin/python experiments/14nov_v5/transfer_monitor.py [--campaign v5] [--once]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ZS = REPO / "experiments/14nov_analyze_v4/zero_shot_commercial.py"
VENDORS = ("wuji2", "tesollo", "xhand", "leap", "dex3", "allegro", "shadow")
VENDOR_CELLS = {"wuji2": "f5_m18-20_{l}_kn1dof_mid", "xhand": "f5_m10-13_{l}_kn1dof_mid",
                "leap": "f4_m12-16_{l}_kn1dof_large", "dex3": "f3_m6-9_{l2}_kn1dof_mid"}
RUN_NAMES = {"k5_none": "MAP-Elites", "k5_global": "Greedy", "hod": "House of Dextra",
             "k5_anchor": "MAP-Elites + anchors"}


def hand_sets(campaign: str) -> dict[str, list[str]]:
    cd = REPO / "debug_outputs/coevolve" / campaign
    spread = (REPO / f"debug_outputs/coevolve/{campaign}/feasible_cells.json").read_text().count("spread") > 0
    l, l2 = ("spread0-120", "spread120-180") if spread else ("gap240-360", "gap180-240")
    reps = sorted(str(p.relative_to(REPO)) for p in (cd / "commercial_snapped").glob("*/*.spec.json"))
    seeds = []
    for c in VENDOR_CELLS.values():
        for i in ("00", "01"):
            p = cd / "seeds/hands" / f"{c.format(l=l, l2=l2)}_{i}" / f"{c.format(l=l, l2=l2)}_{i}.spec.json"
            if p.exists():
                seeds.append(str(p.relative_to(REPO)))
    return {"vendor": list(VENDORS), "replica": reps, "seed": seeds}


def set_of(hand: str) -> str:
    if hand in VENDORS:
        return "vendor"
    return "replica" if "commercial_snapped" in hand else "seed"


def pending(campaign: str, runs: dict[str, int], done: set) -> list[str]:
    out = []
    for run, stride in runs.items():
        rd = REPO / "debug_outputs/coevolve" / campaign / run
        gens = sorted(int(p.parent.name[4:]) for p in rd.glob("gen_*/policy.pth"))
        st = json.loads((rd / "state.json").read_text()) if (rd / "state.json").exists() else {}
        for g in gens:
            lab = f"{campaign}/{run}:{g}"
            shared0 = g == 0 and run != "hod" and (REPO / "debug_outputs/coevolve" / campaign / "hod").exists()
            if shared0:                                    # the searches branch from hod's gen 0: the same policy
                continue
            if lab not in done and (g % stride == 0 or g == gens[-1]) and str(g) in st.get("train_runs", {}):
                out.append(lab)
    return out


def summarise(results: Path, out_dir: Path, campaign: str) -> None:
    if not results.exists():
        return
    r = json.loads(results.read_text())
    rows = {}                                              # (run, gen) -> set -> [goals per hand]
    for hand, v in r.items():
        for lab, x in v["policies"].items():
            run, g = lab.split("/")[-1].split(":")
            rows.setdefault((run, int(g)), {}).setdefault(set_of(hand), []).append(x["goals_per_episode"])
    with open(out_dir / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "generation", "vendor_mean", "vendor_hands_ge1", "replica_mean", "seed_mean", "n_vendor"])
        for (run, g) in sorted(rows):
            s = rows[(run, g)]
            ven = s.get("vendor", [])
            w.writerow([run, g, f"{sum(ven) / len(ven):.3f}" if ven else "", sum(x >= 1 for x in ven),
                        f"{sum(s['replica']) / len(s['replica']):.3f}" if s.get("replica") else "",
                        f"{sum(s['seed']) / len(s['seed']):.3f}" if s.get("seed") else "", len(ven)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"hod": "#2a78d6", "k5_none": "#eb6834", "k5_global": "#1baf7a", "k5_anchor": "#6250d6"}
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), facecolor="white", sharey=True)
    for ax, hs in zip(axs, ("vendor", "replica", "seed")):
        for run in colors:
            pts = sorted((g, sum(s[hs]) / len(s[hs])) for (rn, g), s in rows.items() if rn == run and s.get(hs))
            if pts:
                ax.plot(*zip(*pts), color=colors[run], lw=2, marker="o", ms=3, label=RUN_NAMES[run])
        ax.axhline(1.0, color="#6b6a64", lw=1, ls=(0, (4, 3)))
        ax.set_title({"vendor": "commercial hands (vendor URDFs)", "replica": "replicas in the grammar",
                      "seed": "gen-0 seeds (vendor cells)"}[hs], fontsize=10, loc="left")
        ax.set_xlabel("generation")
        ax.grid(axis="y", color="#e4e3df")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("mean goals / episode (zero-shot)")
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{campaign}: zero-shot transfer by generation (dashed: 1 goal / episode)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out_dir / "transfer.png", dpi=130, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="v7")
    ap.add_argument("--runs", nargs="+", default=["k5_none:1", "k5_anchor:1", "hod:2", "k5_global:2"],
                    help="run:stride")
    ap.add_argument("--envs", type=int, default=50)
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--max-batch", type=int, default=4, help="policies scored per round")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    runs = {r.split(":")[0]: int(r.split(":")[1]) for r in a.runs}
    out_dir = REPO / "experiments" / {"v5": "14nov_v5", "v6": "16nov_v6", "v7": "18nov_v7"}.get(a.campaign, f"transfer_{a.campaign}") / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "transfer.json"
    sets = hand_sets(a.campaign)
    hands = sets["vendor"] + sets["replica"] + sets["seed"]
    print(f"[monitor] {a.campaign}: {len(sets['vendor'])} vendor, {len(sets['replica'])} replica, "
          f"{len(sets['seed'])} seed hands; runs {runs}", flush=True)
    while True:
        done = set()
        if results.exists():
            r = json.loads(results.read_text())
            per = {}
            for h, v in r.items():
                for lab in v["policies"]:
                    per.setdefault(lab, set()).add(h)
            done = {lab for lab, hs in per.items() if set(hands) <= hs}
        todo = pending(a.campaign, runs, done)
        todo.sort(key=lambda lab: (int(lab.split(":")[1]), lab))      # oldest generations first
        todo = todo[: a.max_batch]
        if todo:
            print(f"[monitor] {time.strftime('%H:%M')} scoring {todo}", flush=True)
            subprocess.run([sys.executable, str(ZS), "--hands", *hands, "--policies", *todo, "--envs", str(a.envs),
                            "--results", str(results)], cwd=REPO,
                           stdout=open(out_dir / "monitor_eval.log", "a"), stderr=subprocess.STDOUT)
            summarise(results, out_dir, a.campaign)
            print(f"[monitor] {time.strftime('%H:%M')} done; {out_dir / 'summary.csv'}", flush=True)
            continue
        summarise(results, out_dir, a.campaign)
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
