"""gen-SHARPA zero-shot: the unified policy and the SHARPA-only policy run on gen-SHARPA without training on
it, against gen-SHARPA's own policy. Nominal condition, 1024 envs x 60 s, greedy actions.

goals / episode = goals per minute / episodes per minute, an episode ending in a drop or a timeout.

    .venv_isaacsim/bin/python experiments/02oct_ood_evals/analysis/plot_gen_sharpa_zero_shot.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
R = HERE / "results"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
POLICIES = [("Unified (8 hands)\nep 6400, zero-shot", R / "gen_sharpa_zero_shot/unified_ep_6400", "#2a78d6"),
            ("SHARPA only\nep 3600, zero-shot", R / "gen_sharpa_zero_shot/sharpa_only_ep_3600", "#eb6834"),
            ("gen-SHARPA's own policy\n(trained on it)", HERE.parent / "01oct_embodiment_niches/results/uniform_run2", "#1baf7a")]


def main():
    rows = []
    for label, d, color in POLICIES:
        r = json.loads((d / "gen_sharpa__nominal.json").read_text())
        eps = r["drops_per_min"] + r["timeouts_per_min"]
        rows.append((label, color, r["goals_per_min"] / eps if eps else float("nan"), r["drops_per_min"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE, gridspec_kw={"wspace": 0.3})
    for ax, (k, title) in zip(axes, ((2, "goals / episode  (higher is better)"),
                                     (3, "drops / min  (lower is better)"))):
        ax.set_facecolor(SURFACE)
        for i, row in enumerate(rows):
            ax.bar(i, row[k], 0.62, color=row[1], zorder=3)
            ax.text(i, row[k], f"{row[k]:.2f}" if row[k] < 10 else f"{row[k]:.1f}", ha="center", va="bottom",
                    fontsize=10, color=INK)
        ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(INK2)
        ax.tick_params(colors=INK2, length=0)
        ax.margins(y=0.15)
    fig.suptitle("Zero-shot transfer to gen-SHARPA", x=0.06, ha="left", fontsize=13, fontweight="bold",
                 color=INK, y=1.04)
    fig.text(0.06, 0.95, "nominal condition, 1024 envs × 60 s, greedy actions · an episode ends in a drop or "
             "a timeout", fontsize=9, color=INK2)
    out = HERE / "plots/gen_sharpa_zero_shot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print("->", out)
    for label, _, g, dr in rows:
        print(f"  {label.splitlines()[0]:26s} goals/episode {g:.2f}  drops/min {dr:.2f}")


if __name__ == "__main__":
    main()
