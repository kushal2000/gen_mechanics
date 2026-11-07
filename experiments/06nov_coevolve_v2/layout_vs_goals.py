"""Kit-free: does the policy do worse on one-sided training hands? For every hand of a co-evolution generation, its
largest angular gap between neighbouring fingers about the palm normal (as audit/population_morph.py measures it),
mean finger length and finger count, against its training-log label (goals/episode under that generation's policy).

    python experiments/06nov_coevolve_v2/layout_vs_goals.py debug_outputs/coevolve/v1/k5_none 7
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.argv, _args = sys.argv[:1] + ["/dev/null"], sys.argv[1:]
src = Path(__file__).with_name("audit") / "population_morph.py"
code = src.read_text().split("m = json.load(open(sys.argv[1]))")[0]     # the stats() helper only
exec(compile(code, str(src), "exec"))

run, g = Path(_args[0]), int(_args[1])
m = json.loads((run / f"gen_{g:04d}/manifest.json").read_text())
lab = json.loads((run / f"gen_{g:04d}/labels.json").read_text())
rows = []
for h in m["hands"]:
    if h["id"] not in lab:
        continue
    try:
        s = stats(h["spec"])
    except Exception as e:                                  # noqa: BLE001
        continue
    rows.append((s["n"], s["max_gap"], s["len_mm"], lab[h["id"]][0], lab[h["id"]][1]))
a = np.array(rows)
n, gap, L, goals, eps = a.T
print(f"{run.name} gen {g}: {len(a)} hands with labels; overall mean goals/ep {goals.mean():.2f}")
bins = [0, 120, 180, 240, 300, 360]
for nf in sorted(set(n.astype(int))):
    sel = n == nf
    if sel.sum() < 20:
        continue
    print(f"  {nf} fingers ({sel.sum()} hands):")
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = sel & (gap >= lo) & (gap < hi)
        if b.sum():
            print(f"    largest gap {lo:3d}-{hi:3d} deg: {b.sum():4d} hands  goals/ep mean {goals[b].mean():6.2f}  "
                  f"median {np.median(goals[b]):6.2f}  best {goals[b].max():6.2f}")
for lo, hi in [(0, 140), (140, 180), (180, 230), (230, 400)]:
    b = (L >= lo) & (L < hi)
    if b.sum():
        print(f"  finger length {lo}-{hi} mm: {b.sum():4d} hands  goals/ep mean {goals[b].mean():6.2f}  best {goals[b].max():6.2f}")
