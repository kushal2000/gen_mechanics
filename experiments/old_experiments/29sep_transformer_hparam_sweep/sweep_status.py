"""Rank every sweep run (plus the reference runs) by EPOCHS TO 1 GOAL/EPISODE (the user's preferred
metric), then by goals/episode at matched epochs. Runs that have not reached 1 are listed after, by value.

    .venv_isaacsim/bin/python experiments/old_experiments/29sep_transformer_hparam_sweep/sweep_status.py [hand]

Columns: goals/episode at epochs 200..2000, the first epoch reaching 1 goal/episode, the latest
value, episode length, current lr, and epochs per hour (bigger models are slower per epoch, so a
variant can win per epoch and lose per hour).
"""
import glob, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as EA

LOGS = "debug_outputs/train_logs"
REFS = {  # the runs the sweep's BASE was copied from (same config, seed 100, 15000-epoch chain)
    "REF sharpa base (664621)": f"{LOGS}/29sep_mlp_learner_trimmed_obs/0_scale_train_tf_5deg_sharpa_trimobs_kpfix_nopen_c*",
    "REF allegro base (664619)": f"{LOGS}/29sep_mlp_learner_trimmed_obs/0_scale_train_tf_5deg_allegro_trimobs_kpfix_nopen_c*",
    "REF gen_sharpa base (664620)": f"{LOGS}/29sep_mlp_learner_trimmed_obs/0_scale_train_tf_5deg_gen_sharpa_trimobs_kpfix_nopen_c*",
    "REF sharpa MLP (663821)": f"{LOGS}/29sep_mlp_learner_trimmed_obs/0_scale_train_mlp_5deg_sharpa_trimobs_kpfix_nopen*",
    "REF gen_sharpa MLP (663820)": f"{LOGS}/29sep_mlp_learner_trimmed_obs/0_scale_train_mlp_5deg_gen_sharpa_trimobs_kpfix_nopen*",
}
EP = [200, 400, 600, 800, 1000, 1500, 2000]
# Runs whose numbers are not results. h4 (667135): PhysX GPU crash ~13 min in, training continued on a
# frozen sim and logged 50 "goals" per 50-step episode (see README).
INVALID = {"sw1_gen_sharpa_h4"}
EPOCH_FRAMES = 196608

def series(pattern):
    d = sorted(glob.glob(pattern))
    fs = glob.glob(f"{d[0]}/rank_0/**/events.out.tfevents*", recursive=True) if d else []
    if not fs: return None
    ea = EA(max(fs, key=os.path.getmtime), size_guidance={"scalars": 0}); ea.Reload()
    t = ea.Tags()["scalars"]
    g = lambda k: (lambda m: ea.Scalars(m[0]) if m else [])([x for x in t if x.endswith(k)])
    return g("episode_final/successes"), g("episode_lengths/step"), g("info/last_lr")

def row(label, pattern):
    """(sort key, text)."""
    r = series(pattern)
    if not r or not r[0]: return (float("inf"), 0.0), f"{label:34s} (no data yet)"
    s, L, lr = r
    at = lambda e: next((x.value for x in s if x.step / EPOCH_FRAMES >= e), None)
    f = lambda v: "     -" if v is None else f"{v:6.2f}" if v >= 1 else f"{v:6.3f}"
    e = s[-1].step / EPOCH_FRAMES
    first1 = next((x.step / EPOCH_FRAMES for x in s if x.value >= 1), None)
    eph = (e / ((s[-1].wall_time - s[0].wall_time) / 3600)) if len(s) > 1 and s[-1].wall_time > s[0].wall_time else 0
    text = (f"{label:34s} {('%5.0f' % first1) if first1 else '    -'} | " + " ".join(f(at(x)) for x in EP) +
            f" | {s[-1].value:6.2f} @{e:5.0f} | len {L[-1].value if L else 0:5.0f} | lr {lr[-1].value if lr else 0:.0e} | {eph:4.0f} ep/h")
    return ((first1 if first1 else float("inf")), -s[-1].value), text

if __name__ == "__main__":
    hand = sys.argv[1] if len(sys.argv) > 1 else None
    rows = []
    for lab, pat in REFS.items():
        if hand is None or f" {hand} " in f" {lab.split()[1]} ":
            rows.append(row(lab, pat))
    for d in sorted(glob.glob(f"{LOGS}/29sep_transformer_hparam_sweep/0_scale_train_sw*")):
        name = os.path.basename(d).split("_c01_")[0].replace("0_scale_train_", "")
        if name in INVALID:
            continue
        if hand is None or f"_{hand}_" in f"{name}_":
            rows.append(row(name, d))
    print(f"{'run':34s} ep@1  | " + " ".join(f"@{x:>5d}" for x in EP) + " | latest       | ...")
    for _, text in sorted(rows, key=lambda r: r[0]):
        print(text)
