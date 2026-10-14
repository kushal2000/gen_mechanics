"""The embodiment-curriculum result: the 5-finger Wuji hand on pen lift, learned via a 2-finger (thumb + index)
policy versus from scratch. Goals per episode of the 5-finger hand (training-time, leader block, 25-epoch mean)
against total training epochs (the curriculum's 900 thumb + index epochs, with no 5-finger experience, count), for
the curriculum run (thumb + index joint transformer s102 at epoch 900 -> the 2 + 5 mix, default pushes) and the same
network trained on the full hand from scratch (seeds 101-103, stagger). The numbers also cover the full-hand MLP runs. In the lab's paper style
(08oct_repro_wuji_full/analysis/paper_style.py), PNG only. Re-run to update while runs train.

    .venv_isaacsim/bin/python experiments/12oct_inhand_olympics/analysis/plot_embodiment_curriculum.py
    # -> experiments/12oct_inhand_olympics/plots/embodiment_curriculum.png (the headline figure, tracked);
    #    its numbers -> debug_outputs/inhand_olympics/embodiment_curriculum/embodiment_curriculum.json
"""
from __future__ import annotations

import glob
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
LOGS = REPO / "debug_outputs/train_logs/12oct_inhand_olympics"
PLOTS = pathlib.Path(__file__).resolve().parent.parent / "plots"
NUMBERS = REPO / "debug_outputs/inhand_olympics/embodiment_curriculum"
PRETRAIN_EPOCHS = 900                      # the thumb + index checkpoint the mix starts from (257159, epoch 900)
CURRICULUM = ("264937", "per_hand_successes/L5_full")
SCRATCH = {"MLP": ("256376", "256377", "256378"), "joint transformer": ("257623", "257624", "257625")}
LEADER = "episode_final/successes_per_block/block_5"


SMOOTH = 25                                  # moving average, as 08oct_repro_wuji_full/analysis/plot_training.py


def series(job: str, tag: str) -> np.ndarray:
    """Goals per episode by epoch (one point per epoch), a SMOOTH-epoch trailing mean."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    f = glob.glob(str(LOGS / f"0_*_{job}/0_*/summaries/events*"))[0]
    ea = EventAccumulator(f, size_guidance={"scalars": 0, "histograms": 1, "images": 1, "tensors": 1})
    ea.Reload()
    v = np.array([e.value for e in sorted(ea.Scalars(tag), key=lambda e: e.step)])
    return np.array([v[max(0, i - SMOOTH + 1):i + 1].mean() for i in range(len(v))])


def main() -> None:
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # The lab's paper style (Times, no grid, bottom legend, hero blue = ours), shared with 08oct's figures.
    sys.path.insert(0, str(REPO / "experiments/08oct_repro_wuji_full/analysis"))
    import paper_style as ps

    cur = series(*CURRICULUM)
    scratch = {(arch, j): series(j, LEADER) for arch, jobs in SCRATCH.items() for j in jobs}
    seed = {j: f"s{101 + i}" for jobs in SCRATCH.values() for i, j in enumerate(jobs)}
    # Ours in hero blue; the matched-architecture baseline (the joint transformer from scratch, every seed its own
    # thin line) in the orange foil. The MLP runs stay in the numbers (json), not the figure.

    ps.configure()
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH * 0.6, 2.6))
    ax.axvspan(0, PRETRAIN_EPOCHS, color=ps.COLORS["hero_light"], alpha=0.25, lw=0)
    ax.text(PRETRAIN_EPOCHS / 2, 46, "Thumb + index\npretraining", ha="center", va="top", fontsize=7,
            color=ps.COLORS["muted_ink"])
    jt = [g for (arch, _), g in scratch.items() if arch == "joint transformer"]
    for g in jt:
        ax.plot(np.arange(1, len(g) + 1), g, color=ps.COLORS["foil"], linewidth=1.2, alpha=0.9)
    ax.plot(PRETRAIN_EPOCHS + np.arange(1, len(cur) + 1), cur, color=ps.COLORS["hero"], linewidth=2.0)
    stuck = sum(float(g.max()) < 1 for g in jt)
    if stuck:      # their lines lie on the axis: say so
        ax.text(1950, 9, f"{stuck} of {len(jt)} from-scratch seeds never\nreach 1 goal / episode (lines at 0)",
                ha="center", va="center", fontsize=6.5, color=ps.COLORS["muted_ink"])
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 48)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Goals / episode (5-finger hand)")
    ax.set_title("Training Progress", fontweight="bold")
    ps.style_axis(ax)
    fig.tight_layout()
    handles = [Line2D([], [], color=ps.COLORS["hero"], linewidth=2.2),
               Line2D([], [], color=ps.COLORS["foil"], linewidth=1.4)]
    labels = ["Thumb + index → 2 + 5 mix (Ours)", "Full hand from scratch (3 seeds)"]
    ps.bottom_legend(fig, handles, labels, y=-0.01, loc="upper center", ncol=1, handlelength=2.2)
    ps.save_figure(fig, PLOTS / "embodiment_curriculum", formats=("png",))
    plt.close(fig)

    def at(g, e):
        e = int(e)
        return round(float(g[e - 1]), 1) if 0 < e <= len(g) else None
    summary = {
        "smoothing_epochs": SMOOTH,
        "curriculum": {"mix_epochs": len(cur), "total_epochs": PRETRAIN_EPOCHS + len(cur),
                       "five_finger_epochs_equiv": len(cur) / 2, "goals_now": round(float(cur[-1]), 1)},
        "scratch_at_total_epochs": {f"{a} {seed[j]}": at(g, PRETRAIN_EPOCHS + len(cur)) for (a, j), g in scratch.items()},
        "scratch_at_1500": {f"{a} {seed[j]}": at(g, 1500) for (a, j), g in scratch.items()},
        "scratch_final": {f"{a} {seed[j]}": {"epochs": len(g), "goals": round(float(g[-1]), 1),
                                             "first_epoch_ge_1": next((i + 1 for i, x in enumerate(g) if x >= 1), None)}
                          for (a, j), g in scratch.items()},
    }
    NUMBERS.mkdir(parents=True, exist_ok=True)
    (NUMBERS / "embodiment_curriculum.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
