"""Compare every transformer-debug arm with the stalled base run and the MLP, at matched epochs.

    .venv_isaacsim/bin/python experiments/27oct_transformer_debug/analysis/compare.py [--round r1]

Learning: goals per episode (wandb episode_final/successes) of each arm against the base (job 697372: the arms' own
settings before their one change) and the MLP on the same hand and tokens (job 705905), at fixed epochs, and the
epoch each first reaches 0.1 / 0.25 / 0.5 / 1 goal per episode. Updates: each arm's per-update stats
(<arm>.update_stats.jsonl: KL per action dimension for finger and wrist actions, clip fraction, the value head's
share of the gradient, the lr). Writes plots/<round>_curves.png (one panel per arm) and prints the tables.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
LOGS = REPO / "debug_outputs/train_logs/27oct_transformer_debug"
PROJECT = "kk837/gen_mechanics_jt_debug"
CONTROLS = {"base (697372)": ("kk837/gen_mechanics_pen_repose_v2", "_697372_"),
            "MLP (705905)": ("kk837/gen_mechanics_pen_repose_v2", "_705905_")}
EPOCHS = (250, 500, 750, 1000, 1500)
THRESHOLDS = (0.1, 0.25, 0.5, 1.0)
COLOURS = {"arm": "#2a78d6", "base": "#eb6834", "mlp": "#1baf7a"}   # dataviz reference palette, slots 1-3


def curve(run) -> tuple[np.ndarray, np.ndarray]:
    hs = run.history(keys=["episode_final/successes"], samples=3000, pandas=True)
    he = run.history(keys=["info/epochs"], samples=3000, pandas=True)
    if len(hs) == 0 or len(he) == 0:
        return np.zeros(0), np.zeros(0)
    e = np.interp(hs["_step"].to_numpy(), he["_step"].to_numpy(), he["info/epochs"].to_numpy())
    return e, hs["episode_final/successes"].to_numpy()


def at(e, s, ep, w=40):
    m = (e > ep - w) & (e <= ep)
    return float(np.mean(s[m])) if m.any() else float("nan")


def first(e, s, thr):
    ok = s >= thr
    return int(e[np.argmax(ok)]) if ok.any() else None


def update_summary(tag: str) -> dict | None:
    f = LOGS / f"{tag}.update_stats.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["epoch"] > 20] or rows
    rows = [r for r in rows if np.isfinite(r.get("grad/total", np.nan))]      # AMP overflow steps carry inf norms
    mean = lambda k: float(np.nanmedian([r.get(k, np.nan) for r in rows]))    # noqa: E731
    g = {k: mean(f"grad/{k}") for k in ("hand_head", "wrist_head", "trunk", "value_head", "sigma")}
    gtot = sum(v * v for v in g.values() if v == v) ** 0.5
    return {"kl_finger": mean("kl/hand"), "kl_wrist": mean("kl/wrist"), "clip": mean("clip_frac"),
            "value_share": g["value_head"] / gtot if gtot else float("nan"), "lr": mean("lr")}


def main() -> None:
    import wandb
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="r1")
    args = ap.parse_args()
    api = wandb.Api(timeout=120)
    curves = {}
    for label, (proj, key) in CONTROLS.items():
        r = next((r for r in api.runs(proj) if key in r.name), None)
        if r is not None:
            curves[label] = curve(r)
    arms = {}
    for r in api.runs(PROJECT):
        m = re.search(r"_jd_(.+?)_\d+_\d{8}_\d{6}$", r.name)
        tag = m.group(1) if m else None
        if tag and tag.startswith(args.round + "_"):
            arms[tag] = (curve(r), r.state)

    print(f"goals per episode at matched epochs (mean over the 40 epochs up to each)\n{'':24s}"
          + "".join(f"{('ep' + str(x)):>8s}" for x in EPOCHS) + "   first epoch at " + "/".join(map(str, THRESHOLDS)))
    rows = list(curves.items()) + [(t, c) for t, (c, _) in sorted(arms.items())]
    for name, (e, s) in rows:
        if not len(e):
            print(f"{name:24s} no data yet")
            continue
        print(f"{name:24s}" + "".join(f"{at(e, s, x):8.3f}" for x in EPOCHS)
              + f"   {'/'.join(str(first(e, s, t) or '-') for t in THRESHOLDS)}   (now epoch {int(e[-1])})")

    print("\nper-update stats (epochs > 20): KL per finger / wrist action dimension, clip fraction, value head's "
          "share of the gradient norm, lr")
    for tag in sorted(arms):
        u = update_summary(tag)
        if u:
            print(f"{tag:24s} KL finger {u['kl_finger']:.5f}  wrist {u['kl_wrist']:.5f}  clip {u['clip']:.2f}  "
                  f"value share {u['value_share']:.2f}  lr {u['lr']:.2e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tags = sorted(arms)
    if not tags:
        return
    cols = 4
    nrow = (len(tags) + cols - 1) // cols
    fig, axes = plt.subplots(nrow, cols, figsize=(3.4 * cols, 2.6 * nrow), sharex=True, sharey=True, squeeze=False)
    base, mlp = curves.get("base (697372)"), curves.get("MLP (705905)")
    for ax, tag in zip(axes.flat, tags):
        (e, s), state = arms[tag]
        for (ce, cs), colour, lab in ((mlp, COLOURS["mlp"], "MLP"), (base, COLOURS["base"], "base JT"),
                                      ((e, s), COLOURS["arm"], tag.split("_", 1)[1])):
            if ce is not None and len(ce):
                m = ce <= 1600
                ax.plot(ce[m], cs[m], color=colour, lw=2, label=lab)
        ax.set_title(f"{tag.split('_', 1)[1]}" + ("" if state == "finished" else f" ({state})"), fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(tags):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("goals / episode")
    handles = [plt.Line2D([], [], color=COLOURS[k], lw=2) for k in ("arm", "base", "mlp")]
    fig.legend(handles, ["the arm", "base transformer (697372)", "MLP, same tokens (705905)"], loc="upper center",
               ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / "plots" / f"{args.round}_curves.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
