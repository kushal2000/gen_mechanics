"""results/uniform_best: per hand, the better of its two uniform-dynamics training runs.

Every hand was trained twice with the same config (results/uniform_run1, results/uniform_run2); identical
runs ended far apart (Shadow 42.5 vs 0.01 goals/episode), so the niche analysis takes each hand's better
run -- picked by its TRAINING score (goals/episode, mean of the last 100 logged values, as
training_final.py defines it), never by the niche metrics themselves. A hand evaluated in only one run is
taken from that run. Writes selection.json and the niche map's training_final.json alongside.

    .venv_isaacsim/bin/python experiments/01oct_embodiment_niches/analysis/select_best.py
    ... then aggregate.py / plot_bars.py / plot_main.py with --set=uniform_best
"""
import glob
import json
import pathlib
import re
import shutil

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = pathlib.Path(__file__).resolve().parent.parent
RES = HERE / "results"
RUNS = ("uniform_run1", "uniform_run2")
OUT = RES / "uniform_best"


def training_score(ckpt: str) -> tuple[float, int]:
    """(goals/episode over the last 100 logged values, final epoch) of the run that wrote ``ckpt``."""
    run_dir = pathlib.Path(ckpt).parents[1]
    ev = max(glob.glob(f"{run_dir}/summaries/events*"), key=lambda p: pathlib.Path(p).stat().st_mtime)
    ea = EventAccumulator(ev, size_guidance={"scalars": 0})
    ea.Reload()
    s = ea.Scalars("episode_final/successes")
    last = [x.value for x in s[-100:]]
    return sum(last) / len(last), int(s[-1].step / 196608)


def main():
    cands = {}
    for run in RUNS:
        for f in sorted((RES / run).glob("*__nominal.json")):
            hand = f.name.split("__")[0]
            files = sorted((RES / run).glob(f"{hand}__*.json"))
            if len(files) < 10:
                print(f"  {run}/{hand}: {len(files)}/10 conditions, not complete -- skipped")
                continue
            ckpt = json.loads(f.read_text())["checkpoint"]
            score, ep = training_score(ckpt)
            job = re.search(r"_s100_(\d+)_", ckpt).group(1)
            cands.setdefault(hand, []).append({"run": run, "job": job, "goals": score, "epoch": ep,
                                               "checkpoint": ckpt, "files": [str(p) for p in files]})
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    selection, train = {}, {}
    for hand, cs in sorted(cands.items()):
        best = max(cs, key=lambda c: c["goals"])
        for p in best["files"]:
            shutil.copy(p, OUT / pathlib.Path(p).name)
        selection[hand] = {"picked": best["run"], "job": best["job"],
                           "candidates": {c["run"]: {"job": c["job"], "train_goals": round(c["goals"], 2),
                                                     "epoch": c["epoch"]} for c in cs}}
        train[hand] = {"study": f"left_{hand}_uniform", "job": best["job"],
                       "goals_per_episode": best["goals"], "last_epoch": best["epoch"]}
        others = ", ".join(f"{c['run']} {c['goals']:.1f}" for c in cs if c is not best)
        print(f"{hand:11s} <- {best['run']} (job {best['job']}, {best['goals']:.1f} goals/ep)"
              + (f"   vs {others}" if others else "   (only run evaluated)"))
    (OUT / "selection.json").write_text(json.dumps(selection, indent=1))
    (OUT / "training_final.json").write_text(json.dumps(train, indent=1))


if __name__ == "__main__":
    main()
