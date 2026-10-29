"""Plots from bench.py's results.json: params vs modelled epoch time, and the per-configuration cost breakdown.

    .venv_isaacsim/bin/python experiments/29oct_transformer_speed/analysis/plot.py [results.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # dataviz reference palette, slots 1-4


def main() -> None:
    f = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.json"
    d = json.loads(f.read_text())
    rows = [r for r in d["rows"] if not r.get("oom")]
    out = HERE / "plots"
    out.mkdir(exist_ok=True)

    # 1. params vs epoch time: what "fast with many params" has to beat
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r in rows:
        mlp = r["name"].startswith("mlp")
        ax.scatter(r["params"] / 1e6, r["epoch_s"], s=70 if mlp else 40, color=ORANGE if mlp else BLUE, zorder=3)
        ax.annotate(r["name"].replace("jt ", ""), (r["params"] / 1e6, r["epoch_s"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("parameters (M)")
    ax.set_ylabel(f"modelled epoch (s): sim {d['sim_s']} s + 16 rollout inferences + the update")
    ax.set_title("hand 000: parameters against time per epoch", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "params_vs_epoch.png", dpi=140)

    # 2. where the epoch goes, per configuration
    rows_sorted = sorted(rows, key=lambda r: r["epoch_s"])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows_sorted) + 1.5))
    names = [r["name"] for r in rows_sorted]
    y = range(len(names))
    sim = [d["sim_s"]] * len(names)
    roll = [r["rollout_s"] for r in rows_sorted]
    upd = [r["update_s"] for r in rows_sorted]
    ax.barh(y, sim, color=YELLOW, label="simulation (fixed)", height=0.7)
    ax.barh(y, roll, left=sim, color=AQUA, label="rollout: 16 policy inferences at 24,576 envs", height=0.7)
    ax.barh(y, upd, left=[s + r for s, r in zip(sim, roll)], color=BLUE, label="update: forward + backward + Adam",
            height=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("seconds per epoch (modelled)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "epoch_breakdown.png", dpi=140)
    print(f"wrote {out / 'params_vs_epoch.png'} and {out / 'epoch_breakdown.png'}")


if __name__ == "__main__":
    main()
