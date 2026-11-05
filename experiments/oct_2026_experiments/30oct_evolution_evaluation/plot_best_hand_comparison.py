"""MAP-Elites' best hand vs greedy evolution's best, one image, 1:1.

    .venv_isaacsim/bin/python experiments/oct_2026_experiments/30oct_evolution_evaluation/plot_best_hand_comparison.py \
        [--map-elites-hand g0058_c00] [--baseline-hand g0047_c06]      (default: each run's best)

The best hand of each run's final archive re-evaluation (final_archive_eval.py: every niche's top candidates under the
run's last policy, 20 envs, 50-goal cap), drawn as plot_niches.py draws hands (home pose, palm down as mounted, seen
from below, one shared scale), with its design and score and the run's final archive numbers underneath. If
--with-sim and render_archive_videos.py has filmed both hands, a frame of each in sim (mid-clip) is added below.
Writes experiments/oct_2026_experiments/30oct_evolution_evaluation/plots/best_hand_map_elites_vs_greedy.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_niches import LIGHT, REPO, _wrist, hand_faces, project  # noqa: E402

FINAL = REPO / "debug_outputs/evolution/final_archive_eval"
VIDEOS = REPO / "debug_outputs/evolution/videos/final_archive"
OUT = HERE / "plots" / "best_hand_map_elites_vs_greedy.png"
RUNS = (("map_elites", "MAP-Elites", "#1baf7a"), ("baseline", "Greedy evolution", "#eb6834"))
INK, INK2 = "#0b0b0b", "#52514e"
LENGTH = {"j0": "short", "j1": "medium", "j2": "long"}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-elites-hand", default=None)
    ap.add_argument("--baseline-hand", default=None)
    ap.add_argument("--map-elites-label", default=None, help="replaces the rank in brackets, e.g. 'best design'")
    ap.add_argument("--baseline-label", default=None)
    ap.add_argument("--scores", choices=("archive", "top100"), default="archive",
                    help="top100: the 100-env, table-safe-reset scores of top_hands_100env.py (rank among its shortlist)")
    ap.add_argument("--summary-tag", default="", help="final_archive_eval.py output tag, e.g. _it73_100env")
    ap.add_argument("--iteration", type=int, default=None,
                    help="hands and scores from that iteration's loop evaluation (gen_<N>/fitness.json, 10 envs) instead")
    ap.add_argument("--with-sim", action="store_true", help="add a row of in-sim frames (render_archive_videos.py)")
    args = ap.parse_args()
    pick = {"map_elites": args.map_elites_hand, "baseline": args.baseline_hand}
    label_of = {"map_elites": args.map_elites_label, "baseline": args.baseline_label}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from coevolution.evolution.descriptors import summary as design_summary
    summary = json.loads((FINAL / f"summary{args.summary_tag}.json").read_text())
    wrist = _wrist()
    light = LIGHT / np.linalg.norm(LIGHT)
    panels = []
    for run, label, color in RUNS:
        s = summary[run]
        winners = [(n, d["best"]) for n, d in s["per_niche"].items() if d["best"]]
        if args.iteration is not None:          # the loop's own evaluation of that iteration (every trained hand)
            from coevolution.evolution.descriptors import niche as niche_of
            g = REPO / "debug_outputs/evolution" / run / f"gen_{args.iteration:04d}"
            fit = json.loads((g / "fitness.json").read_text())["hands"]
            gm = {h["spec"]: h["id"] for h in json.loads((g / "manifest.json").read_text())["hands"]}
            winners = []
            for spec_path, v in fit.items():
                d = json.loads((REPO / spec_path).with_name(Path(spec_path).name.replace(".spec.json", ".design.json")).read_text())
                winners.append((niche_of(d), {"hand": gm[spec_path], "final_goals": v["goals_per_episode"], "spec": spec_path}))
        if pick[run]:
            niche, best = next((n, b) for n, b in winners if b["hand"] == pick[run])
        else:
            niche, best = max(winners, key=lambda t: t[1]["final_goals"])
        if "spec" in best:
            spec = REPO / best["spec"]
        else:
            man = json.loads((FINAL / (f"{run}{args.summary_tag}_manifest.json" if args.summary_tag else f"{run}_manifest.json")).read_text())
            spec = REPO / next(h["spec"] for h in man["hands"] if h["id"] == best["hand"])
        design = json.loads(spec.with_name(spec.name.replace(".spec.json", ".design.json")).read_text())
        frame = np.asarray(json.loads(spec.read_text())["palm_frame"], float)
        tris, cols = hand_faces(spec, wrist.palm_down_rotation(frame))
        xy, depth = project(tris)
        nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        order = np.argsort(-depth.mean(1))
        video = VIDEOS / run / f"{best['hand']}.mp4"
        rank = 1 + sum(1 for _, b in winners if b["final_goals"] > best["final_goals"])
        if args.scores == "top100":
            top = json.loads((FINAL / "top_hands_100env.json").read_text())[run]
            mode = "table-safe (0.1)"
            best = dict(best, final_goals=top[best["hand"]][mode]["goals"], sem=top[best["hand"]][mode]["sem"])
            rank = 1 + sum(1 for d in top.values() if d[mode]["goals"] > best["final_goals"])
        panels.append(dict(run=run, label=label, color=color, niche=niche, best=best, design=design_summary(design),
                           rank=rank, n_winners=len(winners),
                           xy=xy[order], col=cols[order] * (0.35 + 0.65 * np.abs(nrm @ light))[order, None],
                           s=s, video=video if video.exists() else None))
    allxy = np.concatenate([p["xy"].reshape(-1, 2) for p in panels])
    lo, hi = allxy.min(0) - 0.012, allxy.max(0) + 0.012
    with_sim = args.with_sim and all(p["video"] for p in panels)

    rows = 2 if with_sim else 1
    fig, axes = plt.subplots(rows, 2, figsize=(12, 6.8 * rows), facecolor="white", squeeze=False)
    for k, p in enumerate(panels):
        ax = axes[0, k]
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.add_collection(PolyCollection(p["xy"], facecolors=p["col"], edgecolors="none"))
        f, cut, j = p["niche"].split("_")
        ax.set_title(p["label"], fontsize=17, color=INK, loc="left", pad=22, fontweight="bold")
        ax.plot([0, 1], [1.035, 1.035], color=p["color"], lw=6, transform=ax.transAxes, clip_on=False,
                solid_capstyle="butt")
        d = p["design"]
        rank = "the run's best" if p["rank"] == 1 else f"#{p['rank']} in the run"
        rank = label_of[p["run"]] or rank
        pm = f" ± {p['best']['sem']:.1f}" if p["best"].get("sem") else ""
        lines = [(f"{p['best']['final_goals']:.1f}{pm} goals / episode   ({rank})", 15, INK, "bold")]
        y = -0.03
        for text, size, c, w in lines:
            ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", fontsize=size, color=c, fontweight=w)
            y -= 0.075 if size > 12 else 0.06
        if with_sim:
            import imageio
            frames = imageio.mimread(p["video"], memtest=False)
            img = np.asarray(frames[int(0.6 * len(frames))])[62:, :, :3]      # mid-clip, below the caption bar
            bx = axes[1, k]
            bx.imshow(img)
            bx.set_xticks([]); bx.set_yticks([])
            for sp in bx.spines.values():
                sp.set_color("#cccccc")
            bx.set_title("in simulation, under the run's final policy (pen repose)", fontsize=11, color=INK2, loc="left")
    note = ("goals per episode over 100 envs with table-safe finger resets,\nunder the run's final policy"
            if args.summary_tag.endswith("100env") else None) or (f"scores from iteration {args.iteration}'s loop evaluation (10 envs, first episode each),\nunder that iteration's policy"
            if args.iteration is not None else None) or ("goals per episode over 100 envs (± standard error) with table-safe finger resets,\nunder the run's final policy"
            if args.scores == "top100" else None)
    head = (f"Each run's best hand at iteration {args.iteration} (table-safe resets since iteration 64); "
            if args.iteration is not None else "Hands from each run's final archive after 74 iterations (18.5B steps); "
            if args.summary_tag.startswith("_it73") else "Hands from each run's final archive after 64 iterations (16B steps); ")
    fig.text(0.01, 0.005, (head + note + ", sampled actions, "
             "50-goal cap. Hands drawn at their home pose, palm down as mounted, seen from below; one scale.") if note else
             "Hands from each run's final archive after 64 iterations (16B steps): every niche's top candidates "
             "re-evaluated "
             "under the run's final policy (20 envs, sampled actions, 50-goal cap).\nHands drawn at their home pose, palm "
             "down as mounted, seen from below; one scale.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9 if not with_sim else 0.94, bottom=0.13 if not with_sim else 0.06,
                        wspace=0.12, hspace=0.42)
    out = OUT if not any(pick.values()) else OUT.with_name(
        f"hand_{panels[0]['best']['hand']}_vs_{panels[1]['best']['hand']}{'_100env' if args.scores == 'top100' else ''}.png")
    if args.summary_tag:
        out = OUT.with_name(f"best_hand{args.summary_tag}_{panels[0]['best']['hand']}_vs_{panels[1]['best']['hand']}.png")
    if args.iteration is not None:
        out = OUT.with_name(f"best_hand_iter{args.iteration}_{panels[0]['best']['hand']}_vs_{panels[1]['best']['hand']}.png")
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(out, "(with sim frames)" if with_sim else "(designs only)")


if __name__ == "__main__":
    main()
