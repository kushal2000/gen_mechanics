"""Ten-hands board: every left-hand run ranked by epochs to 1 goal/episode, then goals at matched epochs.

    .venv_isaacsim/bin/python experiments/old_experiments/30sep_ten_hands/status.py
"""
import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA

R = "debug_outputs/train_logs/30sep_ten_hands"
EP = [250, 500, 1000, 2000, 3000, 5000]
F = 196608
rows = []
for d in sorted(glob.glob(f"{R}/0_scale_train_left_*")):
    name = os.path.basename(d).split("_c01_")[0].replace("0_scale_train_left_", "")
    fs = glob.glob(f"{d}/rank_0/**/events.out.tfevents*", recursive=True)
    if not fs:
        rows.append(((float("inf"), 0), f"{name:11s} (no data yet)")); continue
    ea = EA(max(fs, key=os.path.getmtime), size_guidance={"scalars": 0}); ea.Reload(); t = ea.Tags()["scalars"]
    g = lambda k: (lambda m: ea.Scalars(m[0]) if m else [])([x for x in t if x.endswith(k)])
    s, L, fall = g("episode_final/successes"), g("episode_lengths/step"), g("episode_final/done_fall")
    if not s:
        rows.append(((float("inf"), 0), f"{name:11s} (no episodes yet)")); continue
    at = lambda e: next((x.value for x in s if x.step / F >= e), None)
    fmt = lambda v: "     -" if v is None else (f"{v:6.2f}" if v >= 1 else f"{v:6.3f}")
    first1 = next((x.step / F for x in s if x.value >= 1), None)
    e = s[-1].step / F
    rows.append(((first1 or float("inf"), -s[-1].value),
                 f"{name:11s} {('%5.0f' % first1) if first1 else '    -'} | " + " ".join(fmt(at(x)) for x in EP)
                 + f" | {s[-1].value:6.2f} @{e:5.0f} | len {L[-1].value if L else 0:5.0f} | drop {fall[-1].value if fall else 0:.2f}"))
print(f"{'hand':11s} ep@1  | " + " ".join(f"@{x:>5d}" for x in EP) + " | latest       | ...")
for _, r in sorted(rows):
    print(r)
