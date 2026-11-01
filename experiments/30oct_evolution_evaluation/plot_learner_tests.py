"""The learner tests of 31 Oct: lr schedule (lr_schedule_test/), PPO vs SAPG (ppo_vs_sapg/), bf16 vs fp16 (amp_bf16_test/).

    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/plot_learner_tests.py

One figure per test, plots/learner_<test>.png. Left: MAP-Elites' best hand g0018_c00 alone, training goals per episode
(episode_final/successes, 50-epoch rolling mean per seed), mean over seeds 100-102 with the slowest-to-fastest band, a
marker at the first epoch the mean reaches 1 goal/episode; log y. Middle: the 30 seed hands under one shared policy
(seed 100 only). Right: the final evaluation (eval_hands_cap.py, cap 200), one dot per seed (the 30-hand run: the mean
over hands). Runs still training are drawn up to their last logged epoch; missing runs and evaluations are skipped.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_training_curves import FRAMES_PER_EPOCH, INK, INK2, SMOOTH, SURFACE, _read, rolling, style  # noqa: E402

LOGS = Path("debug_outputs/train_logs/30oct_evolution_evaluation")
EVALS = (Path("debug_outputs/evolution/lr_test_eval"), Path("debug_outputs/evolution/learner_test_eval"))
OUT = Path(__file__).resolve().parent / "plots"
SEEDS = (100, 101, 102)
EPOCHS = 3180
GRID = np.arange(0, EPOCHS + 1, 10.0)
YMIN = 0.01
# test -> (title, [(run prefix, log folder, label, colour)]); the reference condition (SAPG, adaptive lr capped at 1e-3,
# fp16) is blue in every figure
ADAPT = ("lr_adapt", "lr_schedule_test", "SAPG, adaptive lr (capped at 1e-3), fp16", "#2a78d6")
TESTS = {
    "lr_schedule": ("Learning-rate schedule", [
        ADAPT,
        ("lr_fix3em4", "lr_schedule_test", "fixed lr 3e-4", "#eb6834"),
        ("lr_fix1em4", "lr_schedule_test", "fixed lr 1e-4", "#1baf7a")]),
    "ppo_vs_sapg": ("PPO vs SAPG", [
        ADAPT,
        ("ppo", "ppo_vs_sapg", "plain PPO (one policy, one action noise)", "#eb6834")]),
    "amp_bf16": ("bf16 vs fp16 autocast", [
        ADAPT,
        ("bf16", "amp_bf16_test", "bfloat16 autocast (no loss scaler)", "#eb6834")]),
}


def curve(prefix: str, folder: str, hand: str, seed: int):
    d = glob.glob(str(LOGS / folder / f"*_evo_{prefix}_{hand}_s{seed}_*" / "*" / "summaries"))
    if not d:
        return None
    tags = _read(d[0])
    if "episode_final/successes" not in tags:
        return None
    st, v = tags["episode_final/successes"]
    e = st / FRAMES_PER_EPOCH
    y = rolling(v, SMOOTH)
    return np.where(GRID <= e.max(), np.interp(GRID, e, y), np.nan)


def final(prefix: str, hand: str, seed: int):
    for d in EVALS:
        f = d / f"{prefix}_{hand}_s{seed}.json"
        if f.exists():
            return float(np.mean([h["goals_per_episode"] for h in json.loads(f.read_text())["hands"].values()]))
    return None


def draw(ax, curves, color, label):
    curves = np.array(curves)                 # NaN past a seed's last epoch: draw only where every seed has data
    mean, lo, hi = curves.mean(0), curves.min(0), curves.max(0)
    m = np.isfinite(mean)
    if len(curves) > 1:
        ax.fill_between(GRID[m], np.maximum(lo[m], YMIN), np.maximum(hi[m], YMIN), color=color, alpha=0.18, lw=0)
    hit = np.nonzero(mean >= 1.0)[0]
    lab = label + (f": 1 goal/ep at epoch {GRID[hit[0]]:,.0f}" if len(hit) else "")
    ax.plot(GRID[m], np.maximum(mean[m], YMIN), color=color, lw=2.0, label=lab, zorder=3)
    if len(hit):
        ax.plot([GRID[hit[0]]], [mean[hit[0]]], "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=4)


def figure(test: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title, conds = TESTS[test]
    fig, (ax, cx, bx) = plt.subplots(1, 3, figsize=(16, 6), facecolor=SURFACE,
                                     gridspec_kw={"width_ratios": [2.6, 2.0, 1.1]})
    for a in (ax, cx, bx):
        style(a)
    names = []
    for k, (prefix, folder, label, color) in enumerate(conds):
        cs = [c for c in (curve(prefix, folder, "g0018", s) for s in SEEDS) if c is not None]
        if cs:
            draw(ax, cs, color, label + f" ({len(cs)} seeds)")
        c30 = curve(prefix, folder, "seed30", 100)
        if c30 is not None:
            draw(cx, [c30], color, label)
        pts = [v for v in (final(prefix, "g0018", s) for s in SEEDS) if v is not None]
        if pts:
            bx.scatter([k] * len(pts), pts, s=60, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
            bx.plot([k - 0.25, k + 0.25], [np.mean(pts)] * 2, color=color, lw=3, zorder=4)
            bx.text(k + 0.3, np.mean(pts), f"{np.mean(pts):.0f}", va="center", fontsize=10, color=INK)
        v30 = final(prefix, "seed30", 100)
        if v30 is not None:
            bx.scatter([k], [v30], s=60, marker="D", facecolor=SURFACE, edgecolor=color, lw=1.8, zorder=3)
        names.append(prefix)
    for a, t in ((ax, "MAP-Elites' best hand (g0018_c00) alone"), (cx, "30 seed hands, one shared policy (seed 100)")):
        a.set_yscale("log")
        a.set_ylim(YMIN, 60)
        a.set_xlim(0, EPOCHS)
        a.axhline(1.0, color=INK2, lw=0.8, ls=":")
        a.set_xlabel("training epochs (24,576 envs x 16 steps each)", color=INK2, fontsize=11)
        a.set_title(t, color=INK, fontsize=12.5, loc="left")
        a.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK)
    ax.set_ylabel("training goals per episode (log)", color=INK2, fontsize=11)
    bx.set_xlim(-0.6, len(conds) - 0.1)
    bx.set_ylim(0, 200)
    bx.set_xticks(range(len(conds)))
    bx.set_xticklabels([c[0].replace("lr_", "") for c in conds], color=INK2, fontsize=10)
    bx.set_ylabel("final evaluation: goals per episode (cap 200)", color=INK2, fontsize=11)
    bx.set_title(f"After {EPOCHS:,} epochs", color=INK, fontsize=12.5, loc="left")
    fig.suptitle(title, color=INK, fontsize=14, x=0.01, ha="left")
    fig.text(0.01, 0.01, f"Training: episode_final/successes, {SMOOTH}-epoch rolling mean per seed; line = mean over "
             "seeds, band = slowest to fastest seed, dot = first epoch at 1 goal/episode.\nFinal: eval_hands_cap.py, "
             "20 envs per seed, sampled actions, cap 200; dots = g0018 seeds, bar = mean, open diamond = the 30-hand run "
             "(mean over hands).", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.16, wspace=0.22)
    OUT.mkdir(exist_ok=True)
    out = OUT / f"learner_{test}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def main() -> None:
    for test in (sys.argv[1:] or TESTS):
        figure(test)


if __name__ == "__main__":
    main()
