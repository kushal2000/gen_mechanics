"""Score a grid of hand-written symmetric designs (layout x tilt x chain x lengths) with pen_metrics.

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_design/grid.py
-> grid_metrics.json
"""
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pen_metrics as pm  # noqa: E402
from designs import CROSS40, CROSS80, DIAG40, hand, status  # noqa: E402
from hand_sampler.grammar import hand as gh  # noqa: E402

LAYOUTS = {
    "cross40": CROSS40, "cross80": CROSS80, "diag40": DIAG40,
    "tri": [(0, 80), (40, -40), (-40, -40)],          # roughly 120 deg apart
    "tri2": [(0, 40), (40, -40), (-40, -40)],
    "pair40": [(0, 40), (0, -40)],
    "pair80": [(0, 80), (0, -80)],
    "hex": [(0, 80), (80, 0), (0, -80), (-80, 0), (40, 40), (-40, -40)],
    "five": [(0, 80), (80, 0), (0, -80), (-80, 0), (40, 40)],
}
CHAINS = {90: ["sb", "tb", "sbb", "tbb", "sbbb", "stb", "tsb", "sbt"], 60: ["sbb", "tbb"], 0: ["bbb", "bbbb", "bsb"]}
LENGTHS = {2: [(30, 60), (50, 70), (50, 90)], 3: [(30, 40, 40), (30, 60, 50), (50, 50, 50), (50, 70, 50), (30, 80, 70)],
           4: [(30, 40, 40, 40), (30, 50, 50, 50)]}
rows = []
seen = set()
for (lname, spots), (tilt, chains) in itertools.product(LAYOUTS.items(), CHAINS.items()):
    for ch in chains:
        for L in LENGTHS[len(ch)]:
            h = hand(spots, tilt, ch, L)
            key = gh.hand_to_json(h)
            if key in seen:
                continue
            seen.add(key)
            v, c1, c2 = status(h)
            if v or not c1:
                continue
            S = [pm.finger_samples(h, i) for i in range(len(h.fingers))]
            m = pm.evaluate(h, samples=S, n_goals=64, n_axes=64)
            m['reset_clean'] = pm.reset_clean(S)
            m.update(score=pm.score(m), layout=lname, tilt=tilt, chain=ch,
                     lengths=[j.length for j in h.fingers[0].joints], c2=c2[0], design=gh.hand_to_dict(h))
            rows.append(m)
            print(lname, tilt, ch, m["lengths"], {k: round(m[k], 3) for k in
                  ("pickup", "hold3", "pivot", "pivot_h", "pivot_reach", "reset_clean", "dof", "score")}, flush=True)
(HERE / "grid_metrics.json").write_text(json.dumps(rows, indent=1))
