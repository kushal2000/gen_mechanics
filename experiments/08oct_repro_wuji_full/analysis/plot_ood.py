"""OOD robustness of the single-hand MLP policies, one line per embodiment: object mass, friction (object + hand
together), push force. Goals in each env's first episode (64 envs), from run_ood.sub / eval_ood.py.

Two figures, paper style, bottom legend:
  plots/ood_sweeps.{pdf,png}           top: goals per (first) episode; bottom: goals per drop over the whole eval
                                       window (total goals / max(total drops, 1)); linear axes
  plots/ood_sweeps_relative.{pdf,png}  both as a fraction of the embodiment's own trained-condition value
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
    "mlp_full": ("Full Wuji v2", ps.COLORS["reference"], "o"),
    "mlp_len75": ("Links 75% length", ps.BLUE_RAMP[2], "s"),
    "mlp_len50": ("Links 50% length", ps.BLUE_RAMP[0], "^"),
    "mlp_wid120": ("Links 120% width", ps.COLORS["foil"], "D"),
    "mlp_index_middle": ("Index + middle only", ps.COLORS["third"], "v"),
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
    """{embodiment: {axis: {metric: [(value, y), ...]}}} and the checkpoint epoch per embodiment."""
    data, epochs = {}, {}
    for d in sorted(results.iterdir()):
        key = next((k for k in EMBODIMENTS if d.name.startswith(k + "_ep")), None)
        if key is None:
            continue
        epochs[key] = d.name.split("_ep")[-1]
        for axis in AXES:
            f = d / f"{axis}.json"
            if f.exists():
                rows = list(json.loads(f.read_text())["results"].values())
                data.setdefault(key, {})[axis] = {
                    "goals_per_episode_first": sorted((r["value"], r["goals_per_episode_first"]) for r in rows),
                    "goals_per_drop": sorted((r["value"], goals_per_drop(r)) for r in rows)}
    return data, epochs


def plot(data, epochs, relative: bool, out: pathlib.Path):
    fig, grid = plt.subplots(2, 3, figsize=(ps.TEXT_WIDTH, 4.4))
    for row, (metric, (ylabel, logy)) in zip(grid, METRICS.items()):
        _plot_row(row, data, metric, relative, logy and not relative)
        row[0].set_ylabel(("Fraction of trained " + ylabel.lower()) if relative else ylabel)
    for ax in grid[0]:
        ax.set_xlabel("")
    for ax in grid[1]:
        ax.set_title("")
    present = [k for k in EMBODIMENTS if k in data]
    handles = [Line2D([], [], color=EMBODIMENTS[k][1], marker=EMBODIMENTS[k][2], markersize=3.5, linewidth=1.3)
               for k in present]
    labels = [f"{EMBODIMENTS[k][0]} (ep {epochs[k]})" for k in present]
    fig.tight_layout()
    ps.bottom_legend(fig, handles, labels, y=-0.04, ncol=min(len(labels), 3), bold=())
    ps.save_figure(fig, out)
    plt.close(fig)


def _plot_row(axs, data, metric, relative, logy):
    for ax, (axis, (title, xlabel, trained, logx)) in zip(axs, AXES.items()):
        for key, (name, color, marker) in EMBODIMENTS.items():
            pts = data.get(key, {}).get(axis, {}).get(metric)
            if not pts:
                continue
            xs, ys = zip(*pts)
            if relative:
                base = min(pts, key=lambda p: abs(p[0] - trained))[1]
                if not base or base != base:
                    continue
                ys = [y / base for y in ys]
            ax.plot(xs, ys, color=color, marker=marker, markersize=3.5, linewidth=1.3)
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
    plot(data, epochs, False, pathlib.Path(args.out) / "ood_sweeps")
    plot(data, epochs, True, pathlib.Path(args.out) / "ood_sweeps_relative")
    for key in data:
        for axis, metrics in data[key].items():
            for metric, pts in metrics.items():
                print(f"{key:18s} {axis:9s} {metric:24s} " + "  ".join(f"{v:g}: {y:.2f}" for v, y in pts))


if __name__ == "__main__":
    main()
