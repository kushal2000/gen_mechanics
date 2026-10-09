"""Training progress, goals per episode (episode_final/successes) against epoch, of the OOD-axis DR runs (D1 full hand,
D2 index + middle) with their no-DR counterparts (M1, M2: the same config without the DR, seed 100) dashed.

Reads each run's tensorboard summaries; a 25-epoch moving average. Runs still training are drawn up to where they
are. Writes plots/h5_dr_on_ood_axes/training_progress.png.

    .venv_isaacsim/bin/python experiments/08oct_repro_wuji_full/analysis/plot_training.py
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

import paper_style as ps  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
LOGS = HERE.parents[1] / "debug_outputs/train_logs/08oct_repro_wuji_full"
RUNS = [   # (slurm job, legend, colour, line style, alpha): one hue per hand, DR solid, no DR dashed and lighter
    (135615, "Full Wuji v2, OOD-axis DR", ps.COLORS["hero"], "-", 1.0),
    (118744, "Full Wuji v2, no DR", ps.COLORS["hero"], (0, (3, 2)), 0.55),
    (135616, "Index + middle, OOD-axis DR", ps.COLORS["foil"], "-", 1.0),
    (118745, "Index + middle, no DR", ps.COLORS["foil"], (0, (3, 2)), 0.55),
]
SMOOTH = 25


def curve(job: int):
    """(epochs, goals per episode) from the run's summaries; info/epochs maps tensorboard steps to epochs."""
    summaries = next(LOGS.glob(f"0_scale_train_*_{job}_0_*/rank_0/0_scale_train_*/summaries"))
    ea = EventAccumulator(str(summaries), size_guidance={"scalars": 0})
    ea.Reload()
    epoch_at = {e.step: e.value for e in ea.Scalars("info/epochs")}
    pts = sorted((epoch_at[s.step], s.value) for s in ea.Scalars("episode_final/successes") if s.step in epoch_at)
    x, y = (np.array(v) for v in zip(*pts))
    if len(y) >= SMOOTH:
        y = np.convolve(y, np.ones(SMOOTH) / SMOOTH, mode="valid")
        x = x[SMOOTH - 1:]
    return x, y


def main():
    ps.configure()
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH * 0.6, 2.6))
    for job, label, color, ls, alpha in RUNS:
        x, y = curve(job)
        ax.plot(x, y, color=color, linestyle=ls, linewidth=1.4, alpha=alpha, label=label)
        print(f"{label:30s} epoch {int(x[-1])}: {y[-1]:.2f} goals/episode")
    ax.set_xlim(0, 5000)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Goals / episode")
    ax.set_title("Training Progress", fontweight="bold")
    ps.style_axis(ax)
    fig.tight_layout()
    # Two keys instead of four combined entries: colour is the hand, line style how it was trained.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ps.COLORS["hero"], linewidth=2.2),
               Line2D([], [], color=ps.COLORS["foil"], linewidth=2.2),
               Line2D([], [], color=ps.COLORS["muted_ink"], linewidth=1.4),
               Line2D([], [], color=ps.COLORS["muted_ink"], linewidth=1.4, linestyle=(0, (3, 2)), alpha=0.55)]
    labels = ["Full hand", "Index + middle", "Trained with OOD-axis DR", "Trained without DR"]
    ps.bottom_legend(fig, handles, labels, y=-0.01, loc="upper center", ncol=2, bold=(),
                     handlelength=2.2, columnspacing=1.6)
    out = HERE / "plots/h5_dr_on_ood_axes/training_progress"
    ps.save_figure(fig, out, formats=("png",))
    plt.close(fig)


if __name__ == "__main__":
    main()
