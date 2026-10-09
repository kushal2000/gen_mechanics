"""OOD robustness, one line per policy (embodiment x checkpoint): object mass, friction (object + hand together), push
force, palm pitch, object restitution. From run_ood.sub / eval_ood.py (64 envs, each env's first episode, greedy).

Figures, paper style, bottom legend, one image per metric (goals / episode: goals in each env's first episode;
goals / drop: total goals / max(total drops, 1)) and checkpoint group -- "mid" (checkpoints before epoch 5000), "final"
(epoch 5000), "best" (each run's best-training-reward checkpoint, chosen without OOD data). The trained value of each
axis is a dotted line labelled "Train value"; mass on a log axis. Seeds are averaged (line) with their min-max range
(band). Embodiments with no results are skipped. make_plots.sh draws the whole plots/ tree with this script.

    .venv_isaacsim/bin/python experiments/08oct_repro_wuji_full/analysis/plot_ood.py \\
        --only mlp_full,mlp_index_middle --groups best --all-axes --out plots --name full_vs_index_middle
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
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

import paper_style as ps  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
# results label prefix -> (legend name, colour, marker)
EMBODIMENTS = {
    "mlp_full": ("Full Wuji v2 (MLP)", ps.COLORS["reference"], "o"),
    "mlp_len75": ("Links 75% length (MLP)", ps.BLUE_RAMP[2], "s"),
    "mlp_len50": ("Links 50% length (MLP)", ps.BLUE_RAMP[0], "^"),
    "mlp_wid120": ("Links 120% width (MLP)", ps.COLORS["foil"], "D"),
    "mlp_index_middle": ("Index + middle only (MLP)", ps.COLORS["third"], "v"),
    "mlp_thumb_index_middle": ("Thumb + index + middle (MLP)", ps.GREEN_RAMP[2], "<"),
    "mlp_no_ring": ("No ring finger (MLP)", ps.BLUE_RAMP[1], ">"),
    "tf_full": ("Full Wuji v2 (transformer, no delay)", ps.GREEN_RAMP[3], "P"),
    "tf_full_dact": ("Full Wuji v2 (transformer)", ps.GREEN_RAMP[1], "X"),
    # Multi-embodiment hypothesis: the 6 Oct 52-hand transformer (31826) evaluated on one of its hands vs the 6 Oct
    # full-hand specialist with the identical config (12491): EMA 0.1, no action delay, 12288 envs, seed 100.
    "tf6oct_full": ("Full-hand specialist (transformer)", ps.GREEN_RAMP[3], "P"),
    "tf6oct_multi52_on_full": ("52-hand policy, on the full hand", ps.COLORS["foil"], "*"),
    "tf6oct_multi52_on_index_middle": ("52-hand policy, on index + middle", ps.COLORS["foil"], "*"),
    # The same test on index + middle with action delay on: E4 (40 hands with 2 or 3 fingers, 127128) vs the
    # index + middle specialist T1 (make_runs.py), both A2's config.
    "tf_index_middle_dact": ("Index + middle specialist (transformer)", ps.GREEN_RAMP[3], "P"),
    "tf_multi2to3_on_index_middle": ("40-hand policy (2-3 fingers), on index + middle", ps.COLORS["foil"], "*"),
}
AXES = {   # axis -> (title, x label, trained value, log x)
    "mass": ("Object Mass", "Object mass (g)", 45.5625, True),
    "friction": ("Friction", "Friction, object and hand", 0.5, False),
    "push": ("Push Force", "Push force (N), every 0.3 s", 0.0, False),
}
# Axes added later, each a figure of its own (ood_<axis>_<group>...), for the policies that have them.
EXTRA_AXES = {"pitch": ("Hand Tilt Angle", "Hand tilt angle (deg)", 5.0, False),
              "restitution": ("Object Restitution", "Object restitution (hand 0)", 0.0, False)}


def goals_per_drop(r: dict) -> float:
    """Total goals / max(total drops, 1) over the eval window; older results carry only per-minute rates."""
    if "goals_per_drop" in r:
        return r["goals_per_drop"]
    scale = r["envs"] * r["sim_seconds"] / 60.0
    return r["goals_per_min"] * scale / max(r["drops_per_min"] * scale, 1.0)


PUSH_MAX = 2.5          # N: the push axis is plotted up to here
PITCH_RANGE = (-10, 20)  # deg: the scoping runs (seed 100) went -30..50, all ~0 outside this

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
        every = {**AXES, **EXTRA_AXES}
        records = {axis: list(json.loads((d / f"{axis}.json").read_text())["results"].values())
                   for axis in every if (d / f"{axis}.json").exists()}
        # The training condition is a point of every sweep. Plot one number for it on every panel: the mean over this
        # policy's distinct evaluations of it (sweeps that start a fresh process at it -- mass, push, pitch,
        # restitution -- replay the same 64 episodes exactly and count once; friction's comes third in its process,
        # so it is a different draw).
        nominal = {}
        for axis, rs in records.items():
            for r in rs:
                if abs(r["value"] - every[axis][2]) < 1e-6:
                    nominal[(r["goals_total"], r["drops_total"], r["sim_seconds"]) if "goals_total" in r
                            else (axis,)] = r
        pooled = {"goals_per_episode_first": sum(r["goals_per_episode_first"] for r in nominal.values()) / len(nominal),
                  "goals_per_drop": sum(goals_per_drop(r) for r in nominal.values()) / len(nominal)} if nominal else {}
        for axis, rs in records.items():
            for r in rs:
                if axis == "push" and r["value"] > PUSH_MAX:       # earlier grids went to 10 / 20 N
                    continue
                if axis == "pitch" and not PITCH_RANGE[0] <= r["value"] <= PITCH_RANGE[1]:
                    continue
                at_train = abs(r["value"] - every[axis][2]) < 1e-6
                for metric, y in (("goals_per_episode_first", r["goals_per_episode_first"]),
                                  ("goals_per_drop", goals_per_drop(r))):
                    y = pooled[metric] if at_train else y
                    raw.setdefault(key, {}).setdefault(axis, {}).setdefault(metric, {}).setdefault(
                        r["value"], []).append(y)
    data = {key: {axis: {metric: [(v, sum(ys) / len(ys), min(ys), max(ys), len(ys)) for v, ys in sorted(vals.items())]
                         for metric, vals in metrics.items()} for axis, metrics in axes.items()}
            for key, axes in raw.items()}
    return data, None


def n_seeds(entry) -> int:
    return max((p[4] for axes in entry.values() for pts in axes.values() for p in pts), default=1)


def plot(data, relative: bool, out: pathlib.Path, metric: str, axes: dict, ncols: int | None = None):
    """One metric, one panel per axis: a row of up to 3 (or ncols per row)."""
    ylabel, logy = METRICS[metric]
    ncols = ncols or min(3, len(axes))
    nrows = -(-len(axes) // ncols)
    width = ps.TEXT_WIDTH * {1: 0.45, 2: 0.7}.get(ncols, 1.0)
    fig, grid = plt.subplots(nrows, ncols, figsize=(width, 2.3 * nrows), squeeze=False)
    panels = grid.flatten()
    for ax in panels[len(axes):]:
        ax.set_axis_off()
    _plot_row(panels[:len(axes)], data, metric, relative, logy and not relative, axes)
    for r in range(nrows):
        grid[r][0].set_ylabel(("Fraction of trained " + ylabel.lower()) if relative else ylabel)
    present = ordered(data)
    handles = [Line2D([], [], color=EMBODIMENTS[e][1], marker=EMBODIMENTS[e][2], markersize=3.5, linewidth=1.3,
                      linestyle=style(data, (e, ep))) for e, ep in present]
    labels = [f"{EMBODIMENTS[e][0]} (" + ("best ckpt" if ep == BEST else f"ep {ep}") +
              (f", {n_seeds(data[(e, ep)])} seeds)" if n_seeds(data[(e, ep)]) > 1 else ")") for e, ep in present]
    fig.tight_layout()
    # Anchored by its top just under the panels, so a long legend grows downward instead of into the x labels.
    ps.bottom_legend(fig, handles, labels, y=-0.01, loc="upper center",
                     ncol=min(len(labels), {1: 1, 2: 1 if len(labels) <= 2 else 2}.get(ncols, 3)), bold=())
    out.parent.mkdir(parents=True, exist_ok=True)
    ps.save_figure(fig, out, formats=("png",))     # PNG only: no PDFs stored
    plt.close(fig)


def ordered(data):
    """Checkpoints in EMBODIMENTS order, earliest epoch first."""
    return sorted(data, key=lambda k: (list(EMBODIMENTS).index(k[0]), k[1]))


def style(data, key):
    """Solid for an embodiment's first checkpoint, then dashed, dotted, ..."""
    eps = sorted(ep for e, ep in data if e == key[0])
    return ps.LINE_STYLES[eps.index(key[1]) % len(ps.LINE_STYLES)]


def _plot_row(axs, data, metric, relative, logy, axes):
    for ax, (axis, (title, xlabel, trained, logx)) in zip(axs, axes.items()):
        evaluated = set()
        for key in ordered(data):
            name, color, marker = EMBODIMENTS[key[0]]
            pts = data[key].get(axis, {}).get(metric)
            if not pts:
                continue
            xs, ys, los, his, ns = zip(*pts)
            evaluated.update(xs)
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
        if evaluated:          # one x tick per evaluated value (mass: 45.6 g is the trained cube)
            ticks = sorted(evaluated)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([f"{t:.3g}".replace("-", "\u2212") for t in ticks]))
            ax.xaxis.set_minor_locator(NullLocator())


GRID_2X2 = ("push", "friction", "mass", "pitch")     # --all-axes: reading order, left to right then down
TAGS = {"goals_per_episode_first": "goals_per_episode", "goals_per_drop": "goals_per_drop"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results/ood"))
    ap.add_argument("--out", default=str(HERE / "plots"), help="directory (created)")
    ap.add_argument("--only", default="", help="comma-separated embodiments, e.g. mlp_full,mlp_index_middle")
    ap.add_argument("--groups", default="mid,final,best", help="checkpoint groups to draw")
    ap.add_argument("--name", default="{group}", help="file stem, {group} filled in; + _<metric>[_relative]")
    ap.add_argument("--all-axes", action="store_true",
                    help="one 2x2 figure per metric -- push force, friction (top row), mass, pitch (bottom row) -- instead "
                         "of the 3-axis figure plus one per extra axis (restitution is left out)")
    ap.add_argument("--axes", default="", help="one figure with just these axes, e.g. restitution")
    ap.add_argument("--relative", action="store_true", help="also the _relative (fraction of trained) figures")
    ap.add_argument("--metric", choices=sorted(TAGS.values()), default=None,
                    help="draw only this metric, and name the file <name> with no metric suffix")
    args = ap.parse_args()
    ps.configure()
    data, _ = load(pathlib.Path(args.results))
    only = [s for s in args.only.split(",") if s]
    if only:
        data = {k: v for k, v in data.items() if k[0] in only}
    if not data:
        raise SystemExit(f"no OOD results under {args.results}")
    groups = {"mid": {k: v for k, v in data.items() if 0 <= k[1] < 5000},
              "final": {k: v for k, v in data.items() if k[1] >= 5000},
              "best": {k: v for k, v in data.items() if k[1] == BEST}}
    out = pathlib.Path(args.out)
    for group in args.groups.split(","):
        sub = groups[group]
        if not sub:
            continue
        stem = args.name.format(group=group)
        for metric, tag in TAGS.items():
            if args.metric and tag != args.metric:
                continue
            tag_sfx = "" if args.metric else f"_{tag}"
            if args.all_axes or args.axes:
                every = {**AXES, **EXTRA_AXES}
                names = GRID_2X2 if args.all_axes else args.axes.split(",")
                have = {k: v for k, v in sub.items() if any(a in v for a in names)}
                plot(have, False, out / f"{stem}{tag_sfx}", metric, {a: every[a] for a in names},
                     ncols=2 if len(names) == 4 else None)
                continue
            plot(sub, False, out / f"{stem}{tag_sfx}", metric, AXES)
            if args.relative:
                plot(sub, True, out / f"{stem}{tag_sfx}_relative", metric, AXES)
            for extra, spec in EXTRA_AXES.items():
                have = {k: v for k, v in sub.items() if extra in v}
                if have:
                    plot(have, False, out / f"{stem}_{extra}{tag_sfx}", metric, {extra: spec})
    for key in ordered(data):
        for axis, metrics in data[key].items():
            for metric, pts in metrics.items():
                print(f"{key[0]:18s} ep{key[1]:<5d} {axis:9s} {metric:24s} " +
                      "  ".join(f"{p[0]:g}: {p[1]:.2f}" + (f" (n={p[4]})" if p[4] > 1 else "") for p in pts))


if __name__ == "__main__":
    main()
