"""Score the 100 random coarse hands with pen_metrics (for comparison with the designed hand).

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_design/eval_random.py
-> experiments/27oct_grammar_random_hands/agent_design/random_metrics.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pen_metrics as pm  # noqa: E402
from hand_sampler.grammar import hand as gh  # noqa: E402

ROOT = pm.REPO / "assets/urdf/27oct_grammar_random_hands"
rows = []
man = json.loads((ROOT / "manifest.json").read_text())
for r in man["hands"]:
    name = r["name"]
    h = gh.hand_from_dict(json.loads((ROOT / name / f"{name}.design.json").read_text()))
    S = [pm.finger_samples(h, i) for i in range(len(h.fingers))]
    m = pm.evaluate(h, samples=S)
    m['reset_clean'] = pm.reset_clean(S)
    m["score"] = pm.score(m)
    m["name"] = name
    m["palm_cut"] = bool(h.palm_joints)
    m["c2_ok"] = r["c2_ok"]
    rows.append(m)
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
(HERE / "random_metrics.json").write_text(json.dumps(rows, indent=1))
