"""The embodiment-curriculum result: the 5-finger Wuji hand on pen lift, learned via a 2-finger (thumb + index)
policy versus from scratch. Goals per episode of the 5-finger hand (training-time, leader block, 20-epoch mean)
against

    left   total training epochs: the curriculum's 900 thumb + index epochs (no 5-finger experience) count
    right  5-finger experience, in full-hand-only epochs: a 2 + 5 mix epoch is half a 5-finger epoch

for the curriculum run (thumb + index joint transformer s102 at epoch 900 -> the 2 + 5 mix, default pushes) and every
full-hand run from scratch (MLP and joint transformer, seeds 101-103, stagger). Re-run to update while runs train.

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


def series(job: str, tag: str) -> np.ndarray:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    f = glob.glob(str(LOGS / f"0_*_{job}/0_*/summaries/events*"))[0]
    ea = EventAccumulator(f, size_guidance={"scalars": 0, "histograms": 1, "images": 1, "tensors": 1})
    ea.Reload()
    v = np.array([e.value for e in sorted(ea.Scalars(tag), key=lambda e: e.step)])
    k = 20
    return np.array([v[max(0, i - k + 1):i + 1].mean() for i in range(len(v))])


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cur = series(*CURRICULUM)
    scratch = {(arch, j): series(j, LEADER) for arch, jobs in SCRATCH.items() for j in jobs}
    seed = {j: f"s{101 + i}" for jobs in SCRATCH.values() for i, j in enumerate(jobs)}

    ink, muted, grid = "#1f2328", "#59636e", "#e6e8eb"
    c_cur = "#c2410c"                                        # the curriculum: the one strong colour
    c_arch = {"MLP": "#7c8794", "joint transformer": "#2563a8"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, mode in zip(axes, ("total", "five")):
        if mode == "total":
            ax.axvspan(0, PRETRAIN_EPOCHS, color="#f4ede6", lw=0)
            ax.text(PRETRAIN_EPOCHS / 2, 47, "thumb + index only\n(no 5-finger experience)", ha="center", va="top",
                    fontsize=8.5, color=muted)
            x_cur = PRETRAIN_EPOCHS + np.arange(1, len(cur) + 1)
            ax.set_xlabel("total training epochs (curriculum: 900 thumb + index epochs, then the 2 + 5 mix)")
        else:
            x_cur = 0.5 * np.arange(1, len(cur) + 1)
            ax.set_xlabel("5-finger experience, in full-hand-only epochs (a 2 + 5 mix epoch counts as half)")
        ends = []
        for (arch, j), g in scratch.items():
            ax.plot(np.arange(1, len(g) + 1), g, color=c_arch[arch], lw=1.6, alpha=0.85,
                    ls="-" if arch == "MLP" else (0, (5, 2)))
            ends.append([float(g[-1]), len(g), f"{'MLP' if arch == 'MLP' else 'JT'} {seed[j]}"])
        # Direct labels at each line's end, nudged apart where several end together (the runs stuck at 0).
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 2.4)
        for y, x, lab in ends:
            ax.annotate(lab, (5000, y), xytext=(4, 0), textcoords="offset points", fontsize=8, color=muted,
                        va="center", annotation_clip=False)
        ax.plot(x_cur, cur, color=c_cur, lw=2.6)
        ax.annotate("curriculum", (x_cur[-1], cur[-1]), xytext=(4, 0), textcoords="offset points", fontsize=9,
                    color=ink, va="center", fontweight="bold")
        ax.set_xlim(0, 5000)
        ax.set_ylim(-1, 50)
        ax.grid(color=grid, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#9aa3ad")
        ax.tick_params(colors=muted, labelsize=9)
        ax.xaxis.label.set_color(ink)
        ax.xaxis.label.set_size(9.5)
    axes[0].set_ylabel("5-finger hand: goals per episode\n(training, leader block, 20-epoch mean)", color=ink,
                       fontsize=9.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c_cur, lw=2.6, label="curriculum: thumb + index (JT, 900 epochs) -> 2 + 5 mix"),
               Line2D([], [], color=c_arch["MLP"], lw=1.6, label="full hand from scratch, MLP (3 seeds)"),
               Line2D([], [], color=c_arch["joint transformer"], lw=1.6, ls=(0, (5, 2)),
                      label="full hand from scratch, joint transformer (3 seeds)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Pen lift, 5-finger Wuji hand: learned via a 2-finger policy vs from scratch", y=1.09, fontsize=12,
                 color=ink)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    png = PLOTS / "embodiment_curriculum.png"
    fig.savefig(png, dpi=160, bbox_inches="tight", facecolor="white")

    def at(g, e):
        e = int(e)
        return round(float(g[e - 1]), 1) if 0 < e <= len(g) else None
    summary = {
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
    print(f"wrote {png}\n{json.dumps(summary, indent=1)}")


if __name__ == "__main__":
    main()
