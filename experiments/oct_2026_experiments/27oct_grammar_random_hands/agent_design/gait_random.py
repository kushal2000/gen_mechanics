"""The gait metric (pen_metrics.gait_hold) for the 100 random hands -> random_gait.json."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pen_metrics as pm  # noqa: E402
from hand_sampler.grammar import hand as gh  # noqa: E402

ROOT = pm.REPO / "assets/urdf/27oct_grammar_random_hands"
out = {}
for r in json.loads((ROOT / "manifest.json").read_text())["hands"]:
    name = r["name"]
    h = gh.hand_from_dict(json.loads((ROOT / name / f"{name}.design.json").read_text()))
    if len(h.fingers) < 4:
        out[name] = {"gait_any": 0.0, "gait_all": 0.0}
        continue
    S = [pm.finger_samples(h, i) for i in range(len(h.fingers))]
    out[name] = pm.gait_hold(h, S, n_goals=96)
    print(name, out[name], flush=True)
(HERE / "random_gait.json").write_text(json.dumps(out, indent=1))
