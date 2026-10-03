#!/usr/bin/env python3
"""Grammar-solo comparison: analysis (step 4 of the design in
project-notes/grammar/experiments/grammar_solo/README.md).

Reads, per design, ``<run-root>/train/<group>/<design_index>/windows.jsonl``
(appended by ``tools/poll_design_scores.py`` from the live
``per_design_scores_rank0.json``; schema: ``design_scoring._snapshot_payload``)
and ``design_meta.json`` (group/source/sha256, written by ``run_solo.sh``),
plus ``<run-root>/viable/<group>/viability.json`` (candidates drawn, CPU
pre-filter and grasp-search pass rates, written by
``build_viable_population.py``/``viable.py``).

CPU-only: stdlib + numpy + scipy.stats + matplotlib (Agg). Never imports
isaacsimenvs/isaaclab/hand_sampler, never touches Kit or the GPU.

    python3 experiments/grammar_solo/analyze_solo.py \\
        --run-root /data/pulkitag/users/mpeticco/runs/grammar_solo \\
        --out-dir project-notes/grammar/experiments/grammar_solo

Honest statistics: variants have (at most) 16 designs each; this script
reports median/IQR/best plus a two-sided Mann-Whitney U per pairwise
comparison, uncorrected for multiple comparisons (3 variant pairs x 3
metrics = 9 tests) -- treat borderline p-values accordingly, and say so in
any write-up. A design with no windows.jsonl (training produced no scoring
window at all, e.g. a boot failure or a timeout before the first write
window) is excluded from the metric tables and listed under "Data-quality
warnings" rather than silently dropped.

Output (under --out-dir):
    summary.md       human-readable report
    summary.json     the same data, structured
    summary.csv       one row per design
    plots/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import mannwhitneyu
except ImportError:  # pragma: no cover
    mannwhitneyu = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS: Tuple[str, ...] = ("G_V1", "G_V2S", "G_V3S")
CONTROLS_GROUP = "controls"
LEARN_ROTATIONS_THRESHOLD = 0.3  # "learns at all": rotations/episode > this, at the final window

# dataviz skill's categorical palette, slots 1/2/3/4/5 in fixed order (never re-cycled):
# blue/orange/aqua for the three grammar variants, yellow/magenta for the two controls.
COLOR: Dict[str, str] = {
    "G_V1": "#2a78d6",
    "G_V2S": "#eb6834",
    "G_V3S": "#1baf7a",
    "allegro_right": "#eda100",
    "sharpa_left_on_iiwa14": "#e87ba4",
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#dddad2"

WARNINGS: List[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"[analyze_solo] WARNING: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_windows(path: Path) -> List[dict]:
    out = []
    if not path.exists():
        return out
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            warn(f"{path}: malformed JSON on line {i + 1}, skipped")
    return out


def design_series(windows: List[dict]) -> List[dict]:
    """One row per window for design slot '0' (solo training: one design per
    job, so it is always slot 0): elapsed_s, time_held_s, rotations
    (rotation_progress_mean_rad / 2pi), rad_s, fitness, episodes, has_grasp."""
    out = []
    for w in windows:
        d0 = (w.get("designs") or {}).get("0")
        if not d0:
            continue
        th = float(d0.get("time_held_mean_s", 0.0))
        rot_rad = float(d0.get("rotation_progress_mean_rad", 0.0))
        out.append({
            "elapsed_s": w.get("elapsed_s"),
            "time_held_s": th,
            "rotations": rot_rad / (2.0 * math.pi),
            "rad_s": (rot_rad / th) if th > 1e-6 else 0.0,
            "fitness": float(d0.get("graded_fitness", 0.0)),
            "episodes": int(d0.get("episodes", 0)),
            "has_grasp": d0.get("has_stable_grasp"),
        })
    return out


def load_design(run_root: Path, group: str, design_index: str) -> Optional[dict]:
    d = run_root / "train" / group / str(design_index)
    meta_path = d / "design_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    series = design_series(load_windows(d / "windows.jsonl"))
    if not series:
        warn(f"{group}/{design_index} ({meta.get('source', '?')}): no scoring windows under {d} "
             f"(train.log: {d / 'train.log'})")
        return None
    final = series[-1]
    return {
        "group": group, "design_index": str(design_index), "source": meta.get("source", "?"),
        "sha256": meta.get("sha256", "?"), "n_windows": len(series), "series": series,
        "final_time_held_s": final["time_held_s"], "final_rotations": final["rotations"],
        "final_rad_s": final["rad_s"], "final_fitness": final["fitness"],
        "final_episodes": final["episodes"], "learned": final["rotations"] > LEARN_ROTATIONS_THRESHOLD,
    }


def discover_designs(run_root: Path, group: str) -> List[str]:
    d = run_root / "train" / group
    if not d.exists():
        return []
    return sorted((p.name for p in d.iterdir() if p.is_dir()), key=lambda s: (len(s), s))


def load_viability(run_root: Path, group: str) -> Optional[dict]:
    path = run_root / "viable" / group / "viability.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


def iqr_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None, "best": None, "mean": None}
    a = np.asarray(values, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"n": len(a), "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "best": float(a.max()), "mean": float(a.mean())}


def mw_test(a: List[float], b: List[float]) -> Dict[str, Optional[float]]:
    if mannwhitneyu is None or len(a) < 1 or len(b) < 1:
        return {"u": None, "p": None, "note": "scipy unavailable or an empty group"}
    try:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return {"u": float(u), "p": float(p)}
    except ValueError as e:
        return {"u": None, "p": None, "note": str(e)}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


def _style_axes(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)


def plot_distribution(groups: Dict[str, List[dict]], metric: str, ylabel: str, title: str, out_path: Path,
                       controls: Dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    labels, data = [], []
    for v in VARIANTS:
        vals = [d[metric] for d in groups.get(v, [])]
        if vals:
            labels.append(v)
            data.append(vals)
    bp = ax.boxplot(data, positions=range(len(data)), widths=0.45, patch_artist=True, showfliers=False,
                     medianprops={"color": TEXT_PRIMARY, "linewidth": 2})
    for i, (box, v) in enumerate(zip(bp["boxes"], labels)):
        box.set_facecolor(COLOR.get(v, "#999999"))
        box.set_alpha(0.35)
        box.set_edgecolor(COLOR.get(v, "#999999"))
    for i, v in enumerate(labels):
        vals = [d[metric] for d in groups[v]]
        jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.25
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=COLOR.get(v, "#999999"), s=18,
                   edgecolors="white", linewidths=0.5, zorder=3)
    for hand_id, d in controls.items():
        ax.axhline(d[metric], color=COLOR.get(hand_id, "#999999"), linestyle="--", linewidth=1.5, zorder=2)
        ax.text(len(labels) - 0.5, d[metric], f" {hand_id}", color=COLOR.get(hand_id, "#999999"),
               fontsize=8, va="center")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=TEXT_PRIMARY)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_viability_yield(viability: Dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    labels = [v for v in VARIANTS if v in viability]
    drawn = [viability[v]["candidates_drawn"] for v in labels]
    searched = [viability[v]["candidates_searched"] for v in labels]
    rejected = [viability[v]["prefilter_rejected"] for v in labels]
    viable = [viability[v]["n_designs"] - viability[v].get("n_elites", 0) for v in labels]
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, drawn, width=w, color="#c3c2b7", label="candidates drawn")
    ax.bar(x, searched, width=w, color="#86b6ef", label="grasp-searched (post pre-filter)")
    ax.bar(x + w, viable, width=w, color=[COLOR.get(v, "#999999") for v in labels], label="viable (used)")
    for i, v in enumerate(labels):
        rate = viability[v]["viability_rate"]
        ax.text(i, drawn[i] + max(drawn) * 0.02, f"{rate * 100:.1f}%" if rate is not None else "n/a",
               ha="center", color=TEXT_PRIMARY, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT_PRIMARY)
    ax.set_ylabel("designs")
    ax.set_title("Viability yield per variant (founders, generation 0)", loc="left", fontsize=11,
                fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_learning_curves(groups: Dict[str, List[dict]], metric: str, ylabel: str, title: str,
                         out_path: Path, bin_s: float = 120.0) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for v in VARIANTS:
        designs = groups.get(v, [])
        if not designs:
            continue
        max_t = max((s["elapsed_s"] or 0) for d in designs for s in d["series"])
        if max_t <= 0:
            continue
        bins = np.arange(0, max_t + bin_s, bin_s)
        med, lo, hi, centers = [], [], [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            vals = [s[metric] for d in designs for s in d["series"]
                   if s["elapsed_s"] is not None and b0 <= s["elapsed_s"] < b1]
            if not vals:
                continue
            centers.append((b0 + b1) / 2.0 / 60.0)
            med.append(float(np.median(vals)))
            lo.append(float(np.percentile(vals, 25)))
            hi.append(float(np.percentile(vals, 75)))
        if not centers:
            continue
        ax.plot(centers, med, color=COLOR.get(v, "#999999"), linewidth=2, label=v)
        ax.fill_between(centers, lo, hi, color=COLOR.get(v, "#999999"), alpha=0.18, linewidth=0)
    ax.set_xlabel("training wall time (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_report(run_root: Path, out_dir: Path) -> dict:
    groups: Dict[str, List[dict]] = {}
    for v in VARIANTS:
        rows = []
        for idx in discover_designs(run_root, v):
            d = load_design(run_root, v, idx)
            if d is not None:
                rows.append(d)
        groups[v] = rows

    controls: Dict[str, dict] = {}
    for idx in discover_designs(run_root, CONTROLS_GROUP):
        d = load_design(run_root, CONTROLS_GROUP, idx)
        if d is not None:
            hand_id = d["source"].split(":")[-1] if ":" in d["source"] else d["source"]
            controls[hand_id] = d

    viability: Dict[str, dict] = {}
    for v in VARIANTS:
        vj = load_viability(run_root, v)
        if vj is not None:
            viability[v] = vj
        else:
            warn(f"{v}: no viability.json under {run_root / 'viable' / v}")

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_distribution(groups, "final_time_held_s", "holding time (s)",
                      "Final holding time by variant (+ control reference lines)",
                      plots_dir / "holding_time.png", controls)
    plot_distribution(groups, "final_rotations", "rotations / episode",
                      "Final rotations per episode by variant (+ control reference lines)",
                      plots_dir / "rotations.png", controls)
    plot_distribution(groups, "final_fitness", "graded fitness",
                      "Final graded fitness by variant (+ control reference lines)",
                      plots_dir / "fitness.png", controls)
    if viability:
        plot_viability_yield(viability, plots_dir / "viability_yield.png")
    plot_learning_curves(groups, "time_held_s", "holding time (s)",
                         "Holding time over training (median, IQR band)", plots_dir / "curve_holding_time.png")
    plot_learning_curves(groups, "rotations", "rotations / episode",
                         "Rotations/episode over training (median, IQR band)", plots_dir / "curve_rotations.png")

    metrics = ("final_time_held_s", "final_rotations", "final_fitness")
    distributions = {v: {m: iqr_stats([d[m] for d in groups[v]]) for m in metrics} for v in VARIANTS}
    learn_fraction = {v: (sum(1 for d in groups[v] if d["learned"]) / len(groups[v])) if groups[v] else None
                      for v in VARIANTS}

    pairwise = {}
    for i, v1 in enumerate(VARIANTS):
        for v2 in VARIANTS[i + 1:]:
            key = f"{v1}_vs_{v2}"
            pairwise[key] = {m: mw_test([d[m] for d in groups[v1]], [d[m] for d in groups[v2]]) for m in metrics}

    rows = []
    for v in VARIANTS:
        for d in groups[v]:
            rows.append(d)
    for hand_id, d in controls.items():
        rows.append(d)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        fieldnames = ["group", "design_index", "source", "sha256", "n_windows", "final_time_held_s",
                     "final_rotations", "final_rad_s", "final_fitness", "final_episodes", "learned"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    report = {
        "variants": VARIANTS, "n_designs": {v: len(groups[v]) for v in VARIANTS},
        "distributions": distributions, "learn_fraction_gt_0p3_rot": learn_fraction,
        "mann_whitney_two_sided_uncorrected": pairwise,
        "viability": {v: {k: viability[v].get(k) for k in (
            "candidates_drawn", "candidates_searched", "prefilter_rejected", "founder_viability_rate",
            "viability_rate", "batches", "grasp_search_s", "short_by")} for v in viability},
        "controls": {hand_id: {m: d[m] for m in metrics} for hand_id, d in controls.items()},
        "warnings": list(WARNINGS),
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    write_markdown(report, groups, controls, out_dir / "summary.md")
    return report


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def write_markdown(report: dict, groups: Dict[str, List[dict]], controls: Dict[str, dict], out_path: Path) -> None:
    lines = ["# Grammar-solo comparison: analysis", ""]
    lines.append("Per-design SOLO RL fitness under the HORA in-hand-rotation task: each viable design "
                 "trains its own controller from scratch, 4096 envs, a fixed seed. See this directory's "
                 "README.md for the full design and caveats.")
    lines.append("")
    lines.append("## Viability yield")
    lines.append("")
    lines.append("| Variant | Candidates drawn | CPU pre-filter pass rate | Grasp-search pass rate "
                 "(of searched) | Founder viability (of drawn) | Batches | Search (s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        vi = report["viability"].get(v)
        if not vi:
            lines.append(f"| {v} | - | - | - | - | - | - |")
            continue
        drawn = vi["candidates_drawn"]
        rejected = vi["prefilter_rejected"]
        searched = vi["candidates_searched"]
        prefilter_rate = (drawn - rejected) / drawn if drawn else None
        viable_of_searched = (report["n_designs"][v] / searched) if searched else None
        lines.append(f"| {v} | {drawn} | {pct(prefilter_rate)} | {pct(viable_of_searched)} | "
                     f"{pct(vi.get('founder_viability_rate'))} | {vi['batches']} | {vi['grasp_search_s']:.0f} |")
    lines.append("")
    lines.append("![viability yield](plots/viability_yield.png)")
    lines.append("")
    lines.append("## Final-window distributions (median [IQR], best, n)")
    lines.append("")
    lines.append("| Variant | n | Holding time (s) | Rotations/episode | Graded fitness | "
                 "Fraction learning (rot/ep > 0.3) |")
    lines.append("|---|---|---|---|---|---|")
    for v in VARIANTS:
        n = report["n_designs"][v]
        th = report["distributions"][v]["final_time_held_s"]
        rot = report["distributions"][v]["final_rotations"]
        fit = report["distributions"][v]["final_fitness"]
        lf = report["learn_fraction_gt_0p3_rot"][v]

        def fmt(s):
            if s["median"] is None:
                return "n/a"
            return f"{s['median']:.2f} [{s['q1']:.2f}, {s['q3']:.2f}], best {s['best']:.2f}"
        lf_str = "n/a" if lf is None else f"{lf * 100:.0f}%"
        lines.append(f"| {v} | {n} | {fmt(th)} | {fmt(rot)} | {fmt(fit)} | {lf_str} |")
    lines.append("")
    lines.append("![holding time](plots/holding_time.png)")
    lines.append("")
    lines.append("![rotations](plots/rotations.png)")
    lines.append("")
    lines.append("![fitness](plots/fitness.png)")
    lines.append("")
    lines.append("## Controls (projected commercial hands, n=1 each -- point values, not distributions)")
    lines.append("")
    lines.append("| Hand | Holding time (s) | Rotations/episode | Graded fitness |")
    lines.append("|---|---|---|---|")
    for hand_id, d in controls.items():
        lines.append(f"| {hand_id} | {d['final_time_held_s']:.2f} | {d['final_rotations']:.2f} | "
                     f"{d['final_fitness']:.2f} |")
    lines.append("")
    lines.append("## Rank comparison (Mann-Whitney U, two-sided, uncorrected for multiple comparisons)")
    lines.append("")
    lines.append("3 variant pairs x 3 metrics = 9 tests; no multiple-comparison correction applied -- "
                 "treat p-values near 0.05 accordingly.")
    lines.append("")
    lines.append("| Pair | Metric | U | p |")
    lines.append("|---|---|---|---|")
    for pair, metrics in report["mann_whitney_two_sided_uncorrected"].items():
        for metric, res in metrics.items():
            u = f"{res['u']:.1f}" if res.get("u") is not None else "n/a"
            p = f"{res['p']:.4f}" if res.get("p") is not None else "n/a"
            lines.append(f"| {pair} | {metric} | {u} | {p} |")
    lines.append("")
    lines.append("![holding time learning curve](plots/curve_holding_time.png)")
    lines.append("")
    lines.append("![rotations learning curve](plots/curve_rotations.png)")
    lines.append("")
    if report["warnings"]:
        lines.append("## Data-quality warnings")
        lines.append("")
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    report = build_report(Path(a.run_root), Path(a.out_dir))
    print(f"[analyze_solo] wrote {a.out_dir}/summary.{{md,json,csv}} and plots/")
    for v in VARIANTS:
        print(f"  {v}: n={report['n_designs'][v]}")
    if WARNINGS:
        print(f"  {len(WARNINGS)} warning(s), see summary.md's 'Data-quality warnings'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
