"""Export the chosen design with the random hands' exporter (make_random.build: shared uniform dynamics).

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_design/export_design.py
-> assets/urdf/27oct_grammar_claude_hands/claude_design_0/claude_design_0.{urdf,spec.json,design.json} + meshes
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from hand_sampler.grammar import hand as gh  # noqa: E402
from hand_sampler.grammar import operators as gops  # noqa: E402
from hand_sampler.grammar import viability as gvb  # noqa: E402

spec = importlib.util.spec_from_file_location("make_random", HERE.parent / "make_random.py")
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)

RULES = gh.FEETECH_COARSE_RULES
B, S = (0, 0), (90, 0)
# claude_design_0: four fingers in a cross, 80 mm out, pointing straight down (tilt 90); each a spread and a bend
# crossing at the base (0 mm link), a 70 mm link, a second bend, a 70 mm fingertip link.
fingers = tuple(gh.Finger(y=y, z=z, facing=fa, tilt=90,
                          joints=(gh.Joint(axis=S, length=0), gh.Joint(axis=B, length=70), gh.Joint(axis=B, length=70)))
                for y, z, fa in ((0, 80, 0), (80, 0, 90), (0, -80, 180), (-80, 0, 270)))
hand = gops.canonical(gh.Hand(fingers=fingers), RULES)
assert gh.check(hand, RULES) == [], gh.check(hand, RULES)
assert gvb.clear(hand)
c2 = gvb.c2_workspace_overlap(hand)
assert c2.ok, c2.best_mm
mr.check_uniform()
out = REPO / "assets/urdf/27oct_grammar_claude_hands/claude_design_0"
row = mr.build(hand, "claude_design_0", out)
print(json.dumps(row, indent=1))
print("C2 best", round(c2.best_mm, 2), "mm, pair", c2.pair)
