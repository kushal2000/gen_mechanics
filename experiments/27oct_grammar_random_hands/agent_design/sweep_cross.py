"""Exhaustive link-length sweep of the symmetric downward cross (4 fingers at 80 mm, tilt 90) and the 3-finger
~120 deg layout, for spread-first and twist-first chains, with every metric (gait and reset included).

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_design/sweep_cross.py
-> sweep_cross.json
"""
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pen_metrics as pm  # noqa: E402
from designs import CROSS80, RULES, hand, status  # noqa: E402
from hand_sampler.grammar import hand as gh  # noqa: E402

LAYOUTS = {"cross80": CROSS80, "tri80": [(0, 80), (80, -40), (-80, -40)]}
rows, seen = [], set()
for (lname, spots), ch in itertools.product(LAYOUTS.items(), ["sbb", "tbb", "sb", "tb"]):
    Ls = itertools.product([0, 30, 50, 70, 90], [30, 50, 70, 90], [30, 50, 70, 90]) if len(ch) == 3 else \
        itertools.product([30, 50, 70, 90], [30, 50, 70, 90])
    for L in Ls:
        if sum(L) > 225 or sum(L) < 90:
            continue
        h = hand(spots, 90, ch, L)
        lens = [j.length for j in h.fingers[0].joints]
        if lens != list(L):
            continue                      # not an allowed combination
        key = gh.hand_to_json(h)
        if key in seen:
            continue
        seen.add(key)
        v, c1, c2 = status(h)
        if v or not c1 or not c2[0]:
            continue
        S = [pm.finger_samples(h, i) for i in range(len(h.fingers))]
        m = pm.evaluate(h, samples=S, n_goals=64, n_axes=64)
        m["reset_clean"] = pm.reset_clean(S)
        m.update(pm.gait_hold(h, S, n_goals=64))
        m.update(score=pm.score(m), layout=lname, chain=ch, lengths=lens, design=gh.hand_to_dict(h))
        rows.append(m)
        print(lname, ch, lens, {k: round(m[k], 3) for k in ("pickup", "hold3", "pivot", "pivot_h", "reset_clean",
                                                           "gait_all", "dof", "score")}, flush=True)
(HERE / "sweep_cross.json").write_text(json.dumps(rows, indent=1))
