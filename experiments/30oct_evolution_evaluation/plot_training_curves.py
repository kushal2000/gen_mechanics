"""Training goals per episode against training epochs: RL only vs greedy evolution (baseline) vs MAP-Elites.

    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/plot_training_curves.py

One run of each method from experiments/29oct_evolution_from_scratch (one SLURM job per 636-epoch iteration): the
iterations' TensorBoard logs (debug_outputs/train_logs/evolution/<run>/gen_NNNN/) stitched end to end, each
iteration's epoch = 636 x iteration + its own frames / (24,576 envs x 16 steps). The metric is the trainer's
``episode_final/successes``: mean goals over every episode that ended in that epoch, over the whole population (every
hand the run is training, every SAPG block) -- what wandb shows, not the 10-env per-hand fitness. Lines are a 50-epoch
rolling mean; the marker is the first epoch the rolling mean reaches 1 goal per episode (the headline metric).
Writes experiments/30oct_evolution_evaluation/plots/train_goals_per_episode.png.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

LOGS = Path("debug_outputs/train_logs/evolution")
OUT = Path(__file__).resolve().parent / "plots" / "train_goals_per_episode.png"
FRAMES_PER_EPOCH = 24576 * 16
EPOCHS_PER_ITER = 636
SMOOTH = 50
# Each iteration restarts every env, so the first episodes to end are the quick drops and the metric dips until the
# long, high-goal episodes finish (median ~110 epochs on the baseline, up to ~450). After a restart, points are dropped
# until the iteration first gets back to RECOVER x where the previous one ended (at most MAX_TRIM epochs).
RECOVER, MAX_TRIM = 0.8, 200
YMIN = 0.01                                   # log axis floor: 0 goals (the first epochs from scratch) is not drawn
# (run, label, light-mode categorical slot in fixed order: references/palette.md of the dataviz skill)
METHODS = (("rl_only", "No evolution (RL only)", "#2a78d6"),
           ("baseline", "Greedy evolution (baseline)", "#eb6834"),
           ("map_elites", "MAP-Elites evolution", "#1baf7a"))
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


CACHE = Path("debug_outputs/evolution/curve_cache")
TAGS = ("episode_final/successes", "per_hand_successes/")      # exact tag, or a prefix ending in "/"


def _read(summ: str) -> dict:
    """{tag: (steps, values)} for TAGS from one summaries dir, parsed straight from the event files (much faster
    than EventAccumulator) and cached by the files' sizes and mtimes."""
    import hashlib
    import pickle

    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util import tensor_util
    files = sorted(glob.glob(str(Path(summ) / "events.out.tfevents.*")))
    sig = hashlib.md5(repr([(f, Path(f).stat().st_size, Path(f).stat().st_mtime) for f in files]).encode()).hexdigest()
    cache = CACHE / f"{sig}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    out: dict = {}
    for f in files:
        for ev in EventFileLoader(f).Load():
            for v in ev.summary.value:
                if not any(v.tag == t or (t.endswith("/") and v.tag.startswith(t)) for t in TAGS):
                    continue
                x = v.simple_value if v.HasField("simple_value") else float(tensor_util.make_ndarray(v.tensor))
                out.setdefault(v.tag, ([], []))
                out[v.tag][0].append(ev.step)
                out[v.tag][1].append(x)
    out = {t: (np.asarray(a), np.asarray(b)) for t, (a, b) in out.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def read_many(summs: list[str]) -> list[dict]:
    from multiprocessing import Pool
    with Pool(10) as pool:
        return pool.map(_read, summs)


def iterations(run: str) -> list[tuple[int, dict]]:
    """[(iteration, its series)] for every iteration of a loop run with a training log."""
    dirs = [(int(d.name[4:]), glob.glob(str(d / "*" / "*" / "summaries"))) for d in sorted((LOGS / run).glob("gen_*"))]
    dirs = [(g, s[0]) for g, s in dirs if s]
    return list(zip([g for g, _ in dirs], read_many([s for _, s in dirs])))


def stitch(segments: list[tuple[int, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """[(iteration, epoch within it, value)] -> one series: each restart's dip trimmed, then a SMOOTH-epoch rolling
    mean over the whole (trimmed) series."""
    xs, ys, prev = [], [], None
    for g, e, v in sorted(segments, key=lambda s: s[0]):
        start = 0
        if prev is not None and g > 0 and len(v):
            rm = np.convolve(v, np.ones(10) / 10, mode="full")[:len(v)]          # trailing 10-point mean
            ok = np.nonzero((rm >= RECOVER * prev) | (e >= MAX_TRIM))[0]
            start = ok[0] if len(ok) else len(v)
        xs.extend(g * EPOCHS_PER_ITER + e[start:])
        ys.extend(v[start:])
        if len(v[start:]):
            prev = float(np.mean(v[start:][-50:]))
    return np.asarray(xs), rolling(np.asarray(ys), SMOOTH) if ys else np.asarray(ys)


def curve(run: str, tag: str = "episode_final/successes") -> tuple[np.ndarray, np.ndarray, int]:
    """The run's iterations stitched end to end (stitch: restart dips trimmed, smoothed)."""
    segs, last = [], -1
    for g, ser in iterations(run):
        if tag in ser:
            st, v = ser[tag]
            segs.append((g, st / FRAMES_PER_EPOCH, v))
            last = g
    x, y = stitch(segs)
    return x, y, last


def first_at(x: np.ndarray, y: np.ndarray, level: float = 1.0):
    hit = np.nonzero(y >= level)[0]
    return (x[hit[0]], y[hit[0]]) if len(hit) else None


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)


def rolling(y: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.6), facecolor=SURFACE)
    style(ax)
    data = []
    for run, label, color in METHODS:
        x, y, last = curve(run)
        if len(x):
            data.append((x, y, last, label, color))
    xmax = max(d[0][-1] for d in data)
    for x, y, last, label, color in data:
        hit = first_at(x, y)
        lab = f"{label}: {y[-1]:.1f} goals/ep at iteration {last}; " + (
            f"1 goal/ep at epoch {hit[0]:,.0f}" if hit else "never 1 goal/ep")
        ax.plot(x, np.where(y > YMIN, y, np.nan), color=color, lw=2.0, label=lab, zorder=3)
        if hit:
            ax.plot([hit[0]], [hit[1]], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(YMIN, max(d[1].max() for d in data) * 2.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="y", which="major", color=GRID, lw=0.8)
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, xmax * 1.02)
    ax.set_ylabel("training goals per episode (log scale)", color=INK2, fontsize=11)
    ax.set_xlabel("training epochs (24,576 envs x 16 steps each; 636 per evolution iteration)", color=INK2,
                  fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK)
    sec = ax.secondary_xaxis("top", functions=(lambda e: e / EPOCHS_PER_ITER, lambda i: i * EPOCHS_PER_ITER))
    sec.set_xlabel("evolution iteration", color=INK2, fontsize=11)
    sec.tick_params(colors=INK2)
    ax.set_title("Training goals per episode: RL only vs greedy evolution vs MAP-Elites", color=INK, fontsize=14,
                 loc="left", pad=40)
    fig.text(0.01, 0.012, "One run per method (single seed). The trainer's episode_final/successes over every hand "
             f"being trained.\nEach restart's dip trimmed (until back to {RECOVER:.0%} of the previous iteration's "
             f"level, at most {MAX_TRIM} epochs), then a {SMOOTH}-epoch rolling mean. Markers: first epoch at "
             "1 goal/episode (dotted line).", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.15)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
