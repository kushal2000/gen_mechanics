"""training_final.json for a unified-policy niche set: each hand's training objective (goals/episode, mean
of its last 100 logged values) read from the unified run's per_hand/<hand>_successes curve.

    .venv_isaacsim/bin/python experiments/01oct_embodiment_niches/analysis/unified_training_final.py \
        --set=unified_final --run-glob '<run dir glob>'
"""
import glob, json, os, pathlib, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent.parent
arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{k}=")), d)
SET, RUN_GLOB = arg("set", "unified_final"), arg("run-glob")
ev = max(glob.glob(f"{RUN_GLOB}/rank_0/*/summaries/events*"), key=os.path.getmtime)
ea = EventAccumulator(ev, size_guidance={"scalars": 0})
ea.Reload()
out = {}
for t in sorted(x for x in ea.Tags()["scalars"] if x.startswith("per_hand/") and x.endswith("_successes")):
    hand = t.split("/")[1][: -len("_successes")]
    s = ea.Scalars(t)
    last = [x.value for x in s[-100:]]
    out[hand] = {"study": "left_multi_uniform_canon (unified)", "goals_per_episode": sum(last) / len(last),
                 "last_epoch": s[-1].step / 196608}
    print(f"{hand:9s} {out[hand]['goals_per_episode']:6.2f} goals/episode @ {out[hand]['last_epoch']:.0f}")
p = HERE / "results" / SET / "training_final.json"
p.write_text(json.dumps(out, indent=1))
print("->", p)
