"""Each policy's training objective at the end of training: goals/episode (capped at 50), the
mean of its last 100 logged values, from the run the evaluation used.

    .venv_isaacsim/bin/python experiments/01oct_embodiment_niches/analysis/training_final.py
"""
import glob, json, os, pathlib, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "experiments/old_experiments/30sep_ten_hands"))
from viser_zero_shot import HANDS, LOGS  # noqa: E402

out = {}
for hand, (_, study, _) in HANDS.items():
    fs = glob.glob(f"{LOGS}/0_scale_train_{study}_c01_*/rank_0/**/events.out.tfevents*", recursive=True)
    ea = EA(max(fs, key=os.path.getmtime), size_guidance={"scalars": 0})
    ea.Reload()
    tag = next(t for t in ea.Tags()["scalars"] if t.endswith("episode_final/successes"))
    s = ea.Scalars(tag)
    last = [x.value for x in s[-100:]]
    out[hand] = {"study": study, "goals_per_episode": sum(last) / len(last), "last_epoch": s[-1].step / 196608}
    print(f"{hand:11s} {out[hand]['goals_per_episode']:6.2f} goals/episode (mean of last 100) @ {out[hand]['last_epoch']:.0f}")
p = REPO / "experiments/01oct_embodiment_niches/results/vendor/training_final.json"
p.write_text(json.dumps(out, indent=1))
print("->", p)
