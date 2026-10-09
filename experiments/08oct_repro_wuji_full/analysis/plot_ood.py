"""OOD robustness of the single-hand MLP policies, one line per embodiment: object mass, friction (object + hand
together), push force. Goals in each env's first episode (64 envs), from run_ood.sub / eval_ood.py.

Figures, paper style, bottom legend, one pair per checkpoint group -- "mid" (checkpoints before epoch 5000: 2000,
2500, 3800) and "final" (epoch 5000):
  plots/ood_sweeps_<group>.{pdf,png}           top: goals per (first) episode; bottom: goals per drop over the whole eval
                                       window (total goals / max(total drops, 1)); linear axes
  plots/ood_sweeps_<group>_relative.{pdf,png}  both as a fraction of the embodiment's own trained-condition value
The trained value of each axis is marked (dotted line). Embodiments with no results yet are skipped.

    .venv_isaacsim/bin/python experiments/08oct_repro_wuji_full/analysis/plot_ood.py [--results DIR]
"""
from __future__ import annotations

import argparse
import json
import pathlib

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
    "mass": ("Object Mass", "Object mass (g)", 45.5625, False),
    "friction": ("Friction", "Friction, object and hand", 0.5, False),
    "push": ("Push Force", "Push force (N), every 0.3 s", 0.0, False),
}


def goals_per_drop(r: dict) -> float:
    """Total goals / max(total drops, 1) over the eval window; older results carry only per-minute rates."""
    if "goals_per_drop" in r:
        return r["goals_per_drop"]
    scale = r["envs"] * r["sim_seconds"] / 60.0
    return r["goals_per_min"] * scale / max(r["drops_per_min"] * scale, 1.0)


METRICS = {"goals_per_episode_first": ("Goals / episode", False),
           "goals_per_drop": ("Goals / drop", False)}      # metric -> (label, log y): linear axes throughout


def load(results: pathlib.Path) -> dict:
    """{(embodiment, epoch): {axis: {metric: [(value, y), ...]}}}: one line per checkpoint (an embodiment can have
    several, e.g. index + middle at epoch 2500 and 3800)."""
    data = {}
    for d in sorted(results.iterdir()):
        emb = next((k for k in EMBODIMENTS if d.name.startswith(k + "_ep")), None)
        if emb is None:
            continue
        key = (emb, int(d.name.split("_ep")[-1]))
        for axis in AXES:
            f = d / f"{axis}.json"
            if f.exists():
                rows = list(json.loads(f.read_text())["results"].values())
                data.setdefault(key, {})[axis] = {
                    "goals_per_episode_first": sorted((r["value"], r["goals_per_episode_first"]) for r in rows),
                    "goals_per_drop": sorted((r["value"], goals_per_drop(r)) for r in rows)}
    return data, None


def plot(data, epochs, relative: bool, out: pathlib.Path):
    fig, grid = plt.subplots(2, 3, figsize=(ps.TEXT_WIDTH, 4.4))
    for row, (metric, (ylabel, logy)) in zip(grid, METRICS.items()):
        _plot_row(row, data, metric, relative, logy and not relative)
        row[0].set_ylabel(("Fraction of trained " + ylabel.lower()) if relative else ylabel)
    for ax in grid[0]:
        ax.set_xlabel("")
    for ax in grid[1]:
        ax.set_title("")
    present = ordered(data)
    handles = [Line2D([], [], color=EMBODIMENTS[e][1], marker=EMBODIMENTS[e][2], markersize=3.5, linewidth=1.3,
                      linestyle=style(data, (e, ep))) for e, ep in present]
    labels = [f"{EMBODIMENTS[e][0]} (ep {ep})" for e, ep in present]
    fig.tight_layout()
    ps.bottom_legend(fig, handles, labels, y=-0.04, ncol=min(len(labels), 3), bold=())
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
            xs, ys = zip(*pts)
            if relative:
                base = min(pts, key=lambda p: abs(p[0] - trained))[1]
                if not base or base != base:
                    continue
                ys = [y / base for y in ys]
            ax.plot(xs, ys, color=color, marker=marker, markersize=3.5, linewidth=1.3, linestyle=style(data, key))
        ax.axvline(trained, color=ps.COLORS["neutral"], linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
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
    args = ap.parse_args()
    ps.configure()
    data, epochs = load(pathlib.Path(args.results))
    if not data:
        raise SystemExit(f"no OOD results under {args.results}")
    groups = {"mid": {k: v for k, v in data.items() if k[1] < 5000},
              "final": {k: v for k, v in data.items() if k[1] >= 5000}}
    for name, sub in groups.items():
        if sub:
            plot(sub, epochs, False, pathlib.Path(args.out) / f"ood_sweeps_{name}")
            plot(sub, epochs, True, pathlib.Path(args.out) / f"ood_sweeps_{name}_relative")
    for key in ordered(data):
        for axis, metrics in data[key].items():
            for metric, pts in metrics.items():
                print(f"{key[0]:18s} ep{key[1]:<5d} {axis:9s} {metric:24s} " +
                      "  ".join(f"{v:g}: {y:.2f}" for v, y in pts))


if __name__ == "__main__":
    main()
