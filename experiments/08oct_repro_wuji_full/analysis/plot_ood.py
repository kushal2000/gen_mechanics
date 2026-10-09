"""OOD robustness of the single-hand MLP policies, one line per embodiment: object mass, friction (object + hand
together), push force. Goals in each env's first episode (64 envs), from run_ood.sub / eval_ood.py.

Figures, paper style, bottom legend, one per metric x checkpoint group -- "mid" (checkpoints before epoch 5000: 2000,
2500, 3800), "final" (epoch 5000) and "best" (each run's best-training-reward checkpoint, chosen without OOD data):
  plots/ood_sweeps_<group>[_<a>_vs_<b>]_goals_per_episode[_relative].{pdf,png}   goals per (first) episode
  plots/ood_sweeps_<group>[_<a>_vs_<b>]_goals_per_drop[_relative].{pdf,png}      total goals / max(total drops, 1)
_relative: as a fraction of each policy's own trained-condition value. The trained value of each axis is a dotted line
labelled "Train value"; mass on a log axis. Seeds are averaged (line) with their min-max range (band). Embodiments with
no results yet are skipped. --only a,b restricts to those embodiments.

    .venv_isaacsim/bin/python experiments/08oct_repro_wuji_full/analysis/plot_ood.py [--results DIR]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import paper_style as ps  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
# results label prefix -> (legend name, colour, marker)
EMBODIMENTS = {
    "mlp_full": ("Full Wuji v2 (MLP)", ps.COLORS["reference"], "o"),
    "mlp_len75": ("Links 75% length (MLP)", ps.BLUE_RAMP[2], "s"),
    "mlp_len50": ("Links 50% length (MLP)", ps.BLUE_RAMP[0], "^"),
    "mlp_wid120": ("Links 120% width (MLP)", ps.COLORS["foil"], "D"),
    "mlp_index_middle": ("Index + middle only (MLP)", ps.COLORS["third"], "v"),
    "tf_full": ("Full Wuji v2 (transformer, no delay)", ps.GREEN_RAMP[3], "P"),
    "tf_full_dact": ("Full Wuji v2 (transformer)", ps.GREEN_RAMP[1], "X"),
}
AXES = {   # axis -> (title, x label, trained value, log x)
    "mass": ("Object Mass", "Object mass (g)", 45.5625, True),
    "friction": ("Friction", "Friction, object and hand", 0.5, False),
    "push": ("Push Force", "Push force (N), every 0.3 s", 0.0, False),
}


def goals_per_drop(r: dict) -> float:
    """Total goals / max(total drops, 1) over the eval window; older results carry only per-minute rates."""
    if "goals_per_drop" in r:
        return r["goals_per_drop"]
    scale = r["envs"] * r["sim_seconds"] / 60.0
    return r["goals_per_min"] * scale / max(r["drops_per_min"] * scale, 1.0)


PUSH_MAX = 2.5          # N: the push axis is plotted up to here

METRICS = {"goals_per_episode_first": ("Goals / episode", False),
           "goals_per_drop": ("Goals / drop", False)}      # metric -> (label, log y): linear axes throughout


# <emb>[_s<seed>][_best]_ep<epoch>; no seed = 100. "_best": the run's best-training-reward checkpoint (rl_games'
# nn/<run>.pth), whose epoch differs per seed -- grouped under BEST rather than by epoch.
LABEL = re.compile(r"^(?P<emb>.+?)(?:_s(?P<seed>\d+))?(?P<best>_best)?_ep(?P<ep>\d+)$")
BEST = -1


def load(results: pathlib.Path) -> dict:
    """{(embodiment, epoch): {axis: {metric: [(value, mean, lo, hi, n_seeds), ...]}}}: one line per checkpoint (an
    embodiment can have several, e.g. index + middle at epoch 2500 and 3800), averaged over training seeds with the
    min-max range as a band."""
    raw = {}
    for d in sorted(results.iterdir()):
        m = LABEL.match(d.name)
        if not m or m["emb"] not in EMBODIMENTS:
            continue
        key = (m["emb"], BEST if m["best"] else int(m["ep"]))
        for axis in AXES:
            f = d / f"{axis}.json"
            if not f.exists():
                continue
            for r in json.loads(f.read_text())["results"].values():
                if axis == "push" and r["value"] > PUSH_MAX:       # earlier grids went to 10 / 20 N
                    continue
                for metric, y in (("goals_per_episode_first", r["goals_per_episode_first"]),
                                  ("goals_per_drop", goals_per_drop(r))):
                    raw.setdefault(key, {}).setdefault(axis, {}).setdefault(metric, {}).setdefault(
                        r["value"], []).append(y)
    data = {key: {axis: {metric: [(v, sum(ys) / len(ys), min(ys), max(ys), len(ys)) for v, ys in sorted(vals.items())]
                         for metric, vals in metrics.items()} for axis, metrics in axes.items()}
            for key, axes in raw.items()}
    return data, None


def n_seeds(entry) -> int:
    return max((p[4] for axes in entry.values() for pts in axes.values() for p in pts), default=1)


def plot(data, epochs, relative: bool, out: pathlib.Path, metrics=None):
    metrics = metrics or METRICS
    rows = len(metrics)
    fig, grid = plt.subplots(rows, 3, figsize=(ps.TEXT_WIDTH, 2.3 * rows), squeeze=False)
    for row, (metric, (ylabel, logy)) in zip(grid, metrics.items()):
        _plot_row(row, data, metric, relative, logy and not relative)
        row[0].set_ylabel(("Fraction of trained " + ylabel.lower()) if relative else ylabel)
    for r in range(rows - 1):              # x labels on the bottom row only, titles on the top row only
        for ax in grid[r]:
            ax.set_xlabel("")
    for r in range(1, rows):
        for ax in grid[r]:
            ax.set_title("")
    present = ordered(data)
    handles = [Line2D([], [], color=EMBODIMENTS[e][1], marker=EMBODIMENTS[e][2], markersize=3.5, linewidth=1.3,
                      linestyle=style(data, (e, ep))) for e, ep in present]
    labels = [f"{EMBODIMENTS[e][0]} (" + ("best ckpt" if ep == BEST else f"ep {ep}") +
              (f", {n_seeds(data[(e, ep)])} seeds)" if n_seeds(data[(e, ep)]) > 1 else ")") for e, ep in present]
    fig.tight_layout()
    ps.bottom_legend(fig, handles, labels, y=-0.04 if rows > 1 else -0.12, ncol=min(len(labels), 3), bold=())
    ps.save_figure(fig, out)
    plt.close(fig)


def ordered(data):
    """Checkpoints in EMBODIMENTS order, earliest epoch first."""
    return sorted(data, key=lambda k: (list(EMBODIMENTS).index(k[0]), k[1]))


def style(data, key):
    """Solid for an embodiment's first checkpoint, then dashed, dotted, ..."""
    eps = sorted(ep for e, ep in data if e == key[0])
    return ps.LINE_STYLES[eps.index(key[1]) % len(ps.LINE_STYLES)]


def _plot_row(axs, data, metric, relative, logy):
    for ax, (axis, (title, xlabel, trained, logx)) in zip(axs, AXES.items()):
        for key in ordered(data):
            name, color, marker = EMBODIMENTS[key[0]]
            pts = data[key].get(axis, {}).get(metric)
            if not pts:
                continue
            xs, ys, los, his, ns = zip(*pts)
            if relative:
                base = min(pts, key=lambda p: abs(p[0] - trained))[1]
                if not base or base != base:
                    continue
                ys, los, his = ([y / base for y in seq] for seq in (ys, los, his))
            ax.plot(xs, ys, color=color, marker=marker, markersize=3.5, linewidth=1.3, linestyle=style(data, key))
            if max(ns) > 1:
                ax.fill_between(xs, los, his, color=color, alpha=ps.BAND_ALPHA, linewidth=0)
        ax.axvline(trained, color=ps.COLORS["neutral"], linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
        ax.text(trained, 0.99, " Train value", transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=7, color=ps.COLORS["muted_ink"])
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ps.style_axis(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results/ood"))
    ap.add_argument("--out", default=str(HERE / "plots"))
    ap.add_argument("--only", default="", help="comma-separated embodiments, e.g. mlp_full,mlp_index_middle; "
                                               "writes ood_sweeps_<group>_<names>.{pdf,png}")
    args = ap.parse_args()
    ps.configure()
    data, epochs = load(pathlib.Path(args.results))
    only = [s for s in args.only.split(",") if s]
    if only:
        data = {k: v for k, v in data.items() if k[0] in only}
    suffix = ("_" + "_vs_".join(s.removeprefix("mlp_") for s in only)) if only else ""
    if not data:
        raise SystemExit(f"no OOD results under {args.results}")
    groups = {"mid": {k: v for k, v in data.items() if 0 <= k[1] < 5000},
              "final": {k: v for k, v in data.items() if k[1] >= 5000},
              "best": {k: v for k, v in data.items() if k[1] == BEST}}
    for name, sub in groups.items():
        if sub:
            stem = pathlib.Path(args.out) / f"ood_sweeps_{name}{suffix}"
            for metric in METRICS:               # one image per metric (no combined figure)
                tag = {"goals_per_episode_first": "goals_per_episode", "goals_per_drop": "goals_per_drop"}[metric]
                one = {metric: METRICS[metric]}
                plot(sub, epochs, False, stem.with_name(f"{stem.name}_{tag}"), one)
                plot(sub, epochs, True, stem.with_name(f"{stem.name}_{tag}_relative"), one)
    for key in ordered(data):
        for axis, metrics in data[key].items():
            for metric, pts in metrics.items():
                print(f"{key[0]:18s} ep{key[1]:<5d} {axis:9s} {metric:24s} " +
                      "  ".join(f"{p[0]:g}: {p[1]:.2f}" + (f" (n={p[4]})" if p[4] > 1 else "") for p in pts))


if __name__ == "__main__":
    main()
