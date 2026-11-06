"""When does the design search saturate? One value-net search from the v1 campaign's gen 0 (same labels, net and seed
as the K runs) for 50 rounds, with selected rounds' archives scored by 100 envs of the gen-0 policy.

The search is coevolve.py's own (a K=50 fork of v1/hod gen 0 under debug_outputs/coevolve/studies/k50_none, run with
`coevolve.py next` and no training). It is deterministic, so rounds 1/5/10 equal the k1/k5/k10 runs' gen-1 archives
and their scores (gen_0001/eval100_policy_g0000) are reused; every other round scores only designs not scored before.

    python study_k50.py watch  --rounds 1 2 3 5 10 15 20 30 40 50   # CPU job: build + submit each round as it lands
    python study_k50.py report                                      # table + plots/search_k50.png
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics as st
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/05nov_noevolution/studies"))
CAMP = REPO / "debug_outputs/coevolve/v1"
RUN = REPO / "debug_outputs/coevolve/studies/k50_none"
EVAL_SUB = REPO / "experiments/05nov_noevolution/studies/study_beam_eval.sub"
REUSE = {"k1_none", "k5_none", "k10_none"}


def search_dir() -> Path:
    return RUN / "gen_0000/search"


def round_archive(r: int) -> dict:
    return json.loads((search_dir() / f"round_{r:02d}_archive.json").read_text())


def child_designs(rounds) -> dict[str, dict]:
    out = {}
    for r in rounds:
        with gzip.open(search_dir() / f"round_{r:02d}_children.jsonl.gz", "rt") as fh:
            for line in fh:
                c = json.loads(line)
                if c["admitted"]:
                    out[c["id"]] = c["design"]
    return out


def all_scores() -> dict[str, float]:
    """id -> 100-env goals/episode under the gen-0 policy: seeds, the reused K runs, and this study's chunks."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import plot_search_check as psc
    T = psc.scores(CAMP / "hod/gen_0000", "eval100")
    for n in REUSE:
        T.update(psc.scores(CAMP / n / "gen_0001", "eval100_policy_g0000"))
    ed = RUN / "eval100_rounds"
    for f in sorted(ed.glob("r*_c*_fitness.json")):
        man = json.loads(Path(str(f).replace("_fitness.json", "_manifest.json")).read_text())
        id_of = {h["spec"]: h["id"] for h in man["hands"]}
        for spec, v in json.loads(f.read_text())["hands"].items():
            T[id_of[spec]] = v["goals_per_episode"]
    return T


def pending_ids() -> set[str]:
    ed = RUN / "eval100_rounds"
    out = set()
    for m in ed.glob("r*_c*_manifest.json"):
        out |= {h["id"] for h in json.loads(m.read_text())["hands"]}
    return out


def eval_round(r: int) -> None:
    import study_beam as sb
    A = round_archive(r)["archive"]
    ids = [i for v in A.values() for i in v]
    have = set(all_scores()) | pending_ids()
    for n in REUSE:                                   # the K runs' gen-1 evals (queued or done) cover their ids
        have |= {h["id"] for h in json.loads((CAMP / n / "gen_0001/manifest.json").read_text())["hands"]}
    todo = [i for i in ids if i not in have]
    ed = RUN / "eval100_rounds"
    ed.mkdir(exist_ok=True)
    if not todo:
        print(f"[k50] round {r}: all {len(ids)} designs scored or queued", flush=True)
        return
    reg = json.loads((RUN / "designs.json").read_text())
    kids = child_designs(range(1, r + 1))
    design = {i: (reg[i]["design"] if i in reg else kids[i]) for i in todo}
    t = time.time()
    with Pool(8) as pool:
        built = dict((i, spec) for i, spec, ok in pool.map(
            sb._build, [(i, design[i], str(RUN / "hands")) for i in todo], chunksize=8))
    tr = json.loads((RUN / "state.json").read_text())["train_runs"]["0"]
    pol = (RUN / "gen_0000/policy.pth").resolve()
    lines = []
    for k in range(0, len(todo), 256):
        c = k // 256
        mp = ed / f"r{r:02d}_c{c:02d}_manifest.json"
        mp.write_text(json.dumps({"generation": 1, "grid": None,
                                  "hands": [{"id": i, "spec": built[i]} for i in todo[k:k + 256]]}))
        lines.append(f"{mp} 100 {ed / f'r{r:02d}_c{c:02d}_fitness.json'} {9100 + c} {pol} {tr} 0.02\n")
    (ed / f"r{r:02d}_jobs.txt").write_text("".join(lines))
    out = subprocess.run(["sbatch", f"--array=0-{len(lines) - 1}", f"--job-name=EV_k50_r{r:02d}", "--export",
                          f"ALL,JOBS={ed / f'r{r:02d}_jobs.txt'}", str(EVAL_SUB)], capture_output=True, text=True)
    print(f"[k50] round {r}: {len(todo)} of {len(ids)} designs new to score, built in {time.time() - t:.0f} s, "
          f"{len(lines)} chunks ({out.stdout.strip()} {out.stderr.strip()})", flush=True)


def watch(rounds: list[int]) -> None:
    left = sorted(rounds)
    while left:
        r = left[0]
        if (search_dir() / f"round_{r:02d}_archive.json").exists():
            if r in (1, 5, 10):                        # deterministic: check against the K runs before reusing
                A = round_archive(r)["archive"]
                man = json.loads((CAMP / f"k{r}_none/gen_0001/manifest.json").read_text())
                same = {i for v in A.values() for i in v} == {h["id"] for h in man["hands"]}
                print(f"[k50] round {r}: archive {'equals' if same else 'DIFFERS FROM'} k{r}_none gen 1", flush=True)
            eval_round(r)
            left.pop(0)
        else:
            time.sleep(30)


def report() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = all_scores()
    seeds_gd = CAMP / "hod/gen_0000"
    lab = {i: v[0] for i, v in json.loads((seeds_gd / "labels.json").read_text()).items()}
    T0 = {i: T[i] for i in lab if i in T}
    a, b = np.polyfit([lab[i] for i in T0], [T0[i] for i in T0], 1)
    rows = []
    r = 1
    while (search_dir() / f"round_{r:02d}_archive.json").exists():
        R = round_archive(r)
        A, P = R["archive"], R["pred"]
        ids = [i for v in A.values() for i in v]
        row = {"round": r, "pred": float(np.mean([a * P[i] + b for i in ids])), "scored": sum(i in T for i in ids)}
        if row["scored"] == len(ids):
            row.update(actual=float(np.mean([T[i] for i in ids])),
                       corr=float(np.corrcoef([T[i] for i in ids], [P[i] for i in ids])[0, 1]),
                       top10=float(np.mean([np.mean(sorted(T[i] for i in v)[-10:]) for v in A.values()])),
                       best=float(np.mean([max(T[i] for i in v) for v in A.values()])))
        rows.append(row)
        r += 1
    seed = {"round": 0, "actual": float(np.mean(list(T0.values()))), "pred": None,
            "top10": float(np.mean([np.mean(sorted(T0[i] for i in v)[-10:]) for v in _seed_archive().values()])),
            "best": float(np.mean([max(T0[i] for i in v) for v in _seed_archive().values()]))}
    print(f"{'round':>5s} {'actual':>7s} {'pred':>7s} {'gap':>7s} {'corr':>5s} {'top10':>6s} {'best':>6s}")
    print(f"{0:5d} {seed['actual']:7.3f} {'':>7s} {'':>7s} {'':>5s} {seed['top10']:6.3f} {seed['best']:6.3f}")
    for x in rows:
        if "actual" in x:
            print(f"{x['round']:5d} {x['actual']:7.3f} {x['pred']:7.3f} {x['pred'] - x['actual']:+7.3f} "
                  f"{x['corr']:5.2f} {x['top10']:6.3f} {x['best']:6.3f}")
        else:
            print(f"{x['round']:5d} {'':>7s} {x['pred']:7.3f} {'':>7s} {'':>5s} {'':>6s} {'':>6s}  "
                  f"({x['scored']}/4096 scored)")
    (RUN / "report.json").write_text(json.dumps({"calibration": [a, b], "seeds": seed, "rounds": rows}))

    import plot_vf_archive_grid as G
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=G.SURF)
    pr = [(0, seed["actual"])] + [(x["round"], x["pred"]) for x in rows]
    sc = [(0, seed["actual"])] + [(x["round"], x["actual"]) for x in rows if "actual" in x]
    best = [(0, seed["best"])] + [(x["round"], x["best"]) for x in rows if "best" in x]
    ax.plot(*zip(*pr), color="#9cc3ee", lw=2, label="predicted niche mean")
    ax.plot(*zip(*sc), color="#2a78d6", lw=2, marker="o", ms=5, label="actual niche mean (100 envs)")
    ax.plot(*zip(*best), color="#0d3d7a", lw=2, marker="o", ms=5, ls="--", label="actual best per niche")
    ax.set_xlabel("search rounds (K)", color=G.INK2)
    ax.set_ylabel("goals / episode", color=G.INK2)
    ax.set_title("Value-net search from gen 0, scored by the gen-0 policy", loc="left", fontsize=11,
                 fontweight="semibold", color=G.INK)
    ax.grid(color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=G.INK2, length=0)
    ax.legend(frameon=False, fontsize=9, labelcolor=G.INK2, loc="lower right")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "search_k50.png", dpi=170, facecolor=G.SURF)
    print(f"[k50] wrote {out / 'search_k50.png'}")


def _seed_archive() -> dict[str, list[str]]:
    A: dict[str, list[str]] = {}
    for h in json.loads((CAMP / "hod/gen_0000/manifest.json").read_text())["hands"]:
        A.setdefault(h["niche"], []).append(h["id"])
    return A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("watch", "eval-round", "report"))
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 5, 10, 15, 20, 30, 40, 50])
    ap.add_argument("--round", type=int)
    args = ap.parse_args()
    if args.phase == "watch":
        watch(args.rounds)
    elif args.phase == "eval-round":
        eval_round(args.round)
    else:
        report()


if __name__ == "__main__":
    main()
