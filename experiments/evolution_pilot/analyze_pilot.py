#!/usr/bin/env python3
"""Analysis tooling for the evolution pilot (E-R2', plan-rl-grammar-tuning.md's
"Revision, 2026-09-27"): per-generation curves, a final-generation table, an
archive-occupancy heatmap and lineage, across one or more MAP-Elites runs
(`isaacsimenvs.inhand_reorient.evolution.driver`), grouped by grammar variant.

CPU-only, stdlib + numpy + matplotlib -- reads each run directory's
``generations.jsonl``/``state.json`` (see `evolution/README.md`'s "File
layout" and `evolution/archive.py`'s module docstring for the schema); never
imports `isaacsimenvs`/`isaaclab`/`hand_sampler`, never boots Kit, never
touches the GPU or the cluster.

    python3 experiments/evolution_pilot/analyze_pilot.py \\
        outputs/evolution_pilot/pilot/G_V2S_s0 outputs/evolution_pilot/pilot/G_V2S_s1 \\
        --out-dir outputs/evolution_pilot/analysis

Variant/seed grouping: each run directory's ``state.json["config"]`` (written
by the driver on every run) is the primary source for ``variant``/``seed``;
a directory named ``<variant>_s<seed>`` (e.g. ``G_V2S_s0``) is the fallback
when ``state.json`` is missing or lacks a ``variant`` key.

Honest statistics: with (at most) 2 seeds per variant, this script reports
per-seed values, the mean and the range (max - min) -- never a standard
error or a confidence interval, and never a significance claim. Say so
plainly wherever a per-variant number is reported.

Robustness: a run missing ``generations.jsonl``/``state.json``, a malformed
JSON line, a missing field, a truncated final generation (nonzero
``returncode`` or a single training window), or seeds with unequal
generation counts are all reported as warnings (printed and collected in
``summary.json``/`summary.md`'s "Data-quality warnings" section) rather than
raising -- the script does its best with whatever is present in each run.

Output (written under ``--out-dir``):
    summary.md      human-readable report (curves, tables, heatmaps, lineage,
                     warnings), one section per variant.
    summary.json     the same data, structured, for programmatic use.
    summary.csv      the final-generation table, one row per run.
    plots/<variant>/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Constants (mirror isaacsimenvs/inhand_reorient/evolution/archive.py)
# --------------------------------------------------------------------------

DIGIT_BINS: Tuple[int, ...] = (1, 2, 3, 4, 5)
JOINT_BIN_LABELS: Tuple[str, ...] = ("1-5", "6-10", "11-15", "16-20", "21-25", "26-32")
CELL_LABEL_RE = re.compile(r"^d(?P<digit>\d+)_j(?P<joint>\d+-\d+)$")

PROBE_HANDS_DEFAULT: Tuple[str, ...] = ("allegro_right", "dclaw", "sharpa_left_on_iiwa14", "leap_right")

VARIANT_SEED_RE = re.compile(r"^(?P<variant>.+)_s(?P<seed>\d+)$")

# Fields carried straight off a generations.jsonl row into the final table /
# per-generation numeric curves (excludes nested "cells"/"probes"/"timings").
SCALAR_METRICS: Tuple[str, ...] = (
    "coverage", "qd_score", "best_fitness", "mean_fitness", "median_fitness",
    "n_distinct_founders", "max_founder_share", "mean_joint_count", "max_joint_count",
)

WARNINGS: List[str] = []


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"[analyze_pilot] WARNING: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class RunData:
    run_dir: Path
    name: str
    variant: str
    seed: Optional[int]
    generations: List[dict] = field(default_factory=list)
    state: Optional[dict] = None


def parse_variant_seed(run_dir: Path, config: Optional[dict]) -> Tuple[str, Optional[int]]:
    if config:
        v = config.get("variant")
        if v is not None:
            s = config.get("seed")
            return str(v), (int(s) if s is not None else None)
    m = VARIANT_SEED_RE.match(run_dir.name)
    if m:
        return m.group("variant"), int(m.group("seed"))
    warn(f"{run_dir}: could not parse a variant/seed from state.json config or the directory name "
         f"'{run_dir.name}'; grouping it under its own directory name as the variant")
    return run_dir.name, None


def load_run(run_dir: Path) -> RunData:
    name = run_dir.name
    gen_path = run_dir / "generations.jsonl"
    state_path = run_dir / "state.json"

    generations: List[dict] = []
    if gen_path.exists():
        with open(gen_path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    generations.append(json.loads(line))
                except json.JSONDecodeError as e:
                    warn(f"{run_dir}: malformed JSON at generations.jsonl:{lineno}: {e}")
    else:
        warn(f"{run_dir}: generations.jsonl is missing; this run contributes nothing to curves/tables")

    if any("generation" not in row for row in generations):
        warn(f"{run_dir}: some generations.jsonl rows are missing a 'generation' index; "
             f"sorting/plotting may be unreliable for this run")
    generations.sort(key=lambda r: r.get("generation", -1))

    state: Optional[dict] = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError as e:
            warn(f"{run_dir}: malformed state.json: {e}")
    else:
        warn(f"{run_dir}: state.json is missing (no config, no cross-check of generation_completed)")

    config = (state or {}).get("config") if state else None
    variant, seed = parse_variant_seed(run_dir, config)

    if state is not None and generations:
        gc = state.get("generation_completed")
        last_gen = generations[-1].get("generation")
        if gc is not None and last_gen is not None and gc != last_gen:
            warn(f"{run_dir}: state.json generation_completed={gc} does not match the last "
                 f"generations.jsonl row's generation={last_gen}")

    return RunData(run_dir=run_dir, name=name, variant=variant, seed=seed, generations=generations, state=state)


def group_by_variant(runs: Sequence[RunData]) -> Dict[str, List[RunData]]:
    groups: Dict[str, List[RunData]] = {}
    for r in runs:
        groups.setdefault(r.variant, []).append(r)
    for v, rs in groups.items():
        rs.sort(key=lambda r: (r.seed if r.seed is not None else -1, r.name))
        seed_counts: Dict[Optional[int], int] = {}
        for r in rs:
            seed_counts[r.seed] = seed_counts.get(r.seed, 0) + 1
        dupes = {s: n for s, n in seed_counts.items() if n > 1 and s is not None}
        if dupes:
            warn(f"variant {v}: multiple run directories share the same seed {dupes}; "
                 f"they will all be plotted/averaged together as if independent seeds")
    return groups


def check_unequal_lengths(variant: str, runs: Sequence[RunData]) -> None:
    lens = {r.name: len(r.generations) for r in runs}
    if len(set(lens.values())) > 1:
        warn(f"variant {variant}: seeds have unequal generation counts {lens}; per-generation means "
             f"beyond the shortest run's length are averaged over fewer seeds, and final-generation "
             f"comparisons below are NOT at the same generation for every seed")


# --------------------------------------------------------------------------
# Metric extraction
# --------------------------------------------------------------------------


def metric_series(run: RunData, key: str) -> Tuple[List[int], List[Optional[float]]]:
    xs: List[int] = []
    ys: List[Optional[float]] = []
    for row in run.generations:
        xs.append(row.get("generation"))
        ys.append(row.get(key))
    return xs, ys


def probe_series(run: RunData, hand: str) -> Tuple[List[int], List[Optional[float]]]:
    xs: List[int] = []
    ys: List[Optional[float]] = []
    for row in run.generations:
        xs.append(row.get("generation"))
        probes = row.get("probes") or {}
        entry = probes.get(hand)
        ys.append(entry.get("fitness") if isinstance(entry, dict) else None)
    return xs, ys


def digit_distribution_series(run: RunData) -> Dict[int, Tuple[List[int], List[int]]]:
    """Per digit count d in 1..5: (generations, count of elites with that digit_count)."""
    out: Dict[int, Tuple[List[int], List[int]]] = {}
    per_d: Dict[int, List[int]] = {d: [] for d in DIGIT_BINS}
    gens: List[int] = []
    for row in run.generations:
        gens.append(row.get("generation"))
        cells = row.get("cells") or {}
        counts = {d: 0 for d in DIGIT_BINS}
        for c in cells.values():
            dc = c.get("digit_count")
            if dc in counts:
                counts[dc] += 1
            elif dc is not None:
                warn(f"{run.name} gen {row.get('generation')}: elite with out-of-range digit_count={dc} "
                     f"(expected 1-5); excluded from the digit-distribution plot")
        for d in DIGIT_BINS:
            per_d[d].append(counts[d])
    for d in DIGIT_BINS:
        out[d] = (gens, per_d[d])
    return out


def mean_over_seeds(series_list: Sequence[Tuple[List[int], List[Optional[float]]]]) -> Tuple[List[int], List[float]]:
    by_gen: Dict[int, List[float]] = {}
    for xs, ys in series_list:
        for x, y in zip(xs, ys):
            if x is None or y is None:
                continue
            by_gen.setdefault(x, []).append(float(y))
    gens = sorted(by_gen)
    means = [float(np.mean(by_gen[g])) for g in gens]
    return gens, means


SEED_LINESTYLES = ("-", "--", "-.", ":")


def _seed_label(run: RunData) -> str:
    return f"seed {run.seed}" if run.seed is not None else run.name


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def _finish_axes(ax, title: str, ylabel: str) -> None:
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def plot_scalar(ax, runs: Sequence[RunData], key: str, title: str, ylabel: str) -> None:
    series_list = []
    for run in runs:
        xs, ys = metric_series(run, key)
        series_list.append((xs, ys))
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, alpha=0.7, label=_seed_label(run))
    if len(runs) > 1:
        gens, means = mean_over_seeds(series_list)
        ax.plot(gens, means, marker="o", markersize=3, linewidth=2.2, color="black", label="mean")
    _finish_axes(ax, title, ylabel)


def save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_single(variant: str, runs: Sequence[RunData], key: str, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_scalar(ax, runs, key, f"{variant}: {label}", label)
    save_fig(fig, out_path)


def plot_pair(variant: str, runs: Sequence[RunData], specs: Sequence[Tuple[str, str]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(specs), figsize=(6 * len(specs), 4))
    if len(specs) == 1:
        axes = [axes]
    for ax, (key, label) in zip(axes, specs):
        plot_scalar(ax, runs, key, f"{variant}: {label}", label)
    save_fig(fig, out_path)


def plot_digit_distribution(variant: str, runs: Sequence[RunData], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    colors = {d: cmap(i / (len(DIGIT_BINS) - 1)) for i, d in enumerate(DIGIT_BINS)}
    for si, run in enumerate(runs):
        style = SEED_LINESTYLES[si % len(SEED_LINESTYLES)]
        per_d = digit_distribution_series(run)
        for d in DIGIT_BINS:
            gens, counts = per_d[d]
            label = f"{d} digit{'s' if d != 1 else ''} ({_seed_label(run)})" if len(runs) > 1 else f"{d} digit{'s' if d != 1 else ''}"
            ax.plot(gens, counts, linestyle=style, color=colors[d], marker="o", markersize=2.5,
                     linewidth=1.4, alpha=0.85, label=label)
    ax.set_xlabel("generation")
    ax.set_ylabel("elites with this digit count")
    ax.set_title(f"{variant}: digit-count distribution of the elites", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    save_fig(fig, out_path)


def plot_probes(variant: str, runs: Sequence[RunData], probe_hands: Sequence[str], out_path: Path) -> None:
    n = len(probe_hands)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for i, hand in enumerate(probe_hands):
        ax = axes[i // ncols][i % ncols]
        series_list = []
        for run in runs:
            xs, ys = probe_series(run, hand)
            series_list.append((xs, ys))
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, alpha=0.7, label=_seed_label(run))
        if len(runs) > 1:
            gens, means = mean_over_seeds(series_list)
            ax.plot(gens, means, marker="o", markersize=3, linewidth=2.2, color="black", label="mean")
        _finish_axes(ax, f"probe: {hand}", "fitness")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"{variant}: probe-hand fitness", fontsize=12)
    save_fig(fig, out_path)


def plot_all_curves(variant: str, runs: Sequence[RunData], probe_hands: Sequence[str], plots_dir: Path) -> Dict[str, str]:
    """Writes every per-generation PNG for one variant; returns {name: relpath}."""
    vdir = plots_dir / variant
    paths: Dict[str, Path] = {
        "coverage": vdir / "coverage.png",
        "qd_score": vdir / "qd_score.png",
        "fitness": vdir / "fitness.png",
        "founders": vdir / "founders.png",
        "joints": vdir / "joints.png",
        "digit_distribution": vdir / "digit_distribution.png",
        "probes": vdir / "probes.png",
    }
    plot_single(variant, runs, "coverage", "archive coverage (cells filled of 30)", paths["coverage"])
    plot_single(variant, runs, "qd_score", "QD-score", paths["qd_score"])
    plot_pair(variant, runs, [("best_fitness", "best elite fitness"), ("mean_fitness", "mean elite fitness")], paths["fitness"])
    plot_pair(variant, runs, [("n_distinct_founders", "distinct founders"), ("max_founder_share", "max founder share")], paths["founders"])
    plot_pair(variant, runs, [("mean_joint_count", "mean joint count"), ("max_joint_count", "max joint count")], paths["joints"])
    plot_digit_distribution(variant, runs, paths["digit_distribution"])
    plot_probes(variant, runs, probe_hands, paths["probes"])
    return {name: str(p.relative_to(plots_dir.parent)) for name, p in paths.items()}


def cell_label_to_indices(label: str) -> Tuple[int, int]:
    m = CELL_LABEL_RE.match(label)
    if not m:
        raise ValueError(f"unrecognized cell label {label!r}")
    d = int(m.group("digit")) - 1
    j = JOINT_BIN_LABELS.index(m.group("joint"))
    return d, j


def build_variant_heatmap_data(runs: Sequence[RunData]) -> Tuple[np.ndarray, np.ndarray]:
    """Mean final-generation elite fitness per (digit, joint-bin) cell, and how
    many of the variant's seeds occupy it, aggregated over `runs`."""
    sum_grid = np.zeros((len(DIGIT_BINS), len(JOINT_BIN_LABELS)))
    count_grid = np.zeros_like(sum_grid, dtype=int)
    for run in runs:
        if not run.generations:
            continue
        cells = run.generations[-1].get("cells") or {}
        for label, c in cells.items():
            try:
                d, j = cell_label_to_indices(label)
            except ValueError as e:
                warn(f"{run.name}: {e}; skipped in the archive heatmap")
                continue
            fit = c.get("fitness")
            if fit is None:
                continue
            sum_grid[d, j] += float(fit)
            count_grid[d, j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_grid = np.where(count_grid > 0, sum_grid / np.maximum(count_grid, 1), np.nan)
    return mean_grid, count_grid


def plot_archive_heatmap(variant: str, runs: Sequence[RunData], out_path: Path) -> None:
    mean_grid, count_grid = build_variant_heatmap_data(runs)
    masked = np.ma.masked_invalid(mean_grid)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#e6e6e6")
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(JOINT_BIN_LABELS)))
    ax.set_xticklabels(JOINT_BIN_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(DIGIT_BINS)))
    ax.set_yticklabels([str(d) for d in DIGIT_BINS])
    ax.set_xlabel("joint-count bin")
    ax.set_ylabel("digit count")
    n_seeds = len(runs)
    ax.set_title(f"{variant}: final-generation archive occupancy\n(mean elite fitness across seeds; "
                 f"n/{n_seeds} = seeds occupying that cell)", fontsize=10)
    vmax = float(np.nanmax(mean_grid)) if np.any(~np.isnan(mean_grid)) else 1.0
    for d in range(len(DIGIT_BINS)):
        for j in range(len(JOINT_BIN_LABELS)):
            if count_grid[d, j] > 0:
                val = mean_grid[d, j]
                color = "white" if val < 0.6 * vmax else "black"
                ax.text(j, d, f"{val:.3f}\n({count_grid[d, j]}/{n_seeds})", ha="center", va="center",
                        fontsize=7, color=color)
    fig.colorbar(im, ax=ax, label="mean elite fitness (blank = unoccupied by any seed)")
    save_fig(fig, out_path)


# --------------------------------------------------------------------------
# Final-generation table
# --------------------------------------------------------------------------


def final_generation_row(run: RunData, probe_hands: Sequence[str]) -> Optional[dict]:
    if not run.generations:
        warn(f"{run.name}: no generations logged; excluded from the final-generation table")
        return None
    last = run.generations[-1]

    total_boot = total_train = total_select = 0.0
    missing_timing_gens: List[Any] = []
    for row in run.generations:
        t = row.get("timings")
        if not t:
            missing_timing_gens.append(row.get("generation"))
            t = {}
        total_boot += float(t.get("boot_s") or 0.0)
        total_train += float(t.get("train_s") or 0.0)
        total_select += float(t.get("select_s") or 0.0)
    if missing_timing_gens:
        warn(f"{run.name}: missing 'timings' for generation(s) {missing_timing_gens}; "
             f"total wall time/GPU-hours for this run are undercounted")
    total_wall_s = total_boot + total_train + total_select

    n_windows = last.get("n_windows")
    returncode = last.get("returncode")
    truncated = (returncode not in (0, None)) or (isinstance(n_windows, (int, float)) and n_windows is not None and n_windows <= 1)
    if truncated:
        warn(f"{run.name}: final generation {last.get('generation')} looks truncated/failed "
             f"(returncode={returncode}, n_windows={n_windows}); its fitness/probe numbers rest on "
             f"very few episodes and its row in the final-generation table is flagged accordingly")

    probes = last.get("probes") or {}
    row_out: Dict[str, Any] = {
        "run": run.name, "run_dir": str(run.run_dir), "variant": run.variant, "seed": run.seed,
        "final_generation": last.get("generation"),
        "n_generations_logged": len(run.generations),
        "coverage": last.get("coverage"), "n_cells_total": last.get("n_cells_total"),
        "qd_score": last.get("qd_score"),
        "best_fitness": last.get("best_fitness"), "mean_fitness": last.get("mean_fitness"),
        "median_fitness": last.get("median_fitness"),
        "n_distinct_founders": last.get("n_distinct_founders"),
        "max_founder_share": last.get("max_founder_share"),
        "mean_joint_count": last.get("mean_joint_count"), "max_joint_count": last.get("max_joint_count"),
        "total_wall_time_s": total_wall_s, "total_wall_time_h": total_wall_s / 3600.0,
        "gpu_hours_approx": total_wall_s / 3600.0,
        "final_gen_returncode": returncode, "final_gen_n_windows": n_windows,
        "final_gen_possibly_truncated": truncated,
    }
    for hand in probe_hands:
        entry = probes.get(hand)
        if entry is None:
            warn(f"{run.name}: probe hand {hand!r} missing from the final generation's probe report")
        row_out[f"probe_{hand}_fitness"] = (entry or {}).get("fitness")
        row_out[f"probe_{hand}_episodes"] = (entry or {}).get("episodes")
        row_out[f"probe_{hand}_low_confidence"] = (entry or {}).get("low_confidence")
    for hand in probes:
        if hand not in probe_hands:
            warn(f"{run.name}: unexpected probe hand {hand!r} not in the requested probe set {probe_hands}")
            row_out[f"probe_{hand}_fitness"] = probes[hand].get("fitness")
    return row_out


NUMERIC_AGG_FIELDS_BASE: Tuple[str, ...] = (
    "coverage", "qd_score", "best_fitness", "mean_fitness", "median_fitness",
    "n_distinct_founders", "max_founder_share", "mean_joint_count", "max_joint_count",
    "total_wall_time_h", "gpu_hours_approx",
)


def aggregate_variant(rows: Sequence[dict], probe_hands: Sequence[str]) -> dict:
    fields = list(NUMERIC_AGG_FIELDS_BASE) + [f"probe_{h}_fitness" for h in probe_hands]
    agg: Dict[str, Any] = {"n_seeds": len(rows), "seeds": [r.get("seed") for r in rows]}
    for f_ in fields:
        vals = [r[f_] for r in rows if r.get(f_) is not None]
        if not vals:
            agg[f_] = {"values": [], "mean": None, "min": None, "max": None, "range": None}
            continue
        agg[f_] = {
            "values": vals, "mean": float(np.mean(vals)), "min": float(min(vals)),
            "max": float(max(vals)), "range": float(max(vals) - min(vals)),
        }
    return agg


# --------------------------------------------------------------------------
# Lineage
# --------------------------------------------------------------------------


def build_lineage_registry(run: RunData) -> Dict[str, dict]:
    """design_id -> {parent_id, founder_id, generation_born, digit_count,
    joint_count}, scanned across EVERY generation this run logged (not just
    the final one) -- a design that has since been evicted from the archive
    is still findable here if it was ever an elite in some earlier
    generation's 'cells' snapshot (every generation logs the full archive)."""
    registry: Dict[str, dict] = {}
    for row in run.generations:
        cells = row.get("cells") or {}
        for c in cells.values():
            did = c.get("design_id")
            if did is None:
                continue
            registry[did] = {
                "parent_id": c.get("parent_id"), "founder_id": c.get("founder_id"),
                "generation_born": c.get("generation_born"),
                "digit_count": c.get("digit_count"), "joint_count": c.get("joint_count"),
            }
    return registry


def compute_depth(design_id: str, registry: Dict[str, dict], max_hops: int = 500) -> Tuple[Optional[int], str]:
    """Number of parent-hops from `design_id` back to a founder (parent_id
    None). Returns (None, note) if the chain is broken (a parent_id that
    never appears as an elite in this run's logged history -- shouldn't
    happen given how offspring are sampled, but the archive is silent about
    non-elite candidates, so this is defensive) or cycles."""
    seen = set()
    cur = design_id
    hops = 0
    while True:
        if cur in seen:
            return None, f"cycle detected after {hops} hop(s)"
        seen.add(cur)
        rec = registry.get(cur)
        if rec is None:
            if hops == 0:
                return None, "design_id not found in this run's logged history"
            return None, f"chain broken after {hops} hop(s): parent {cur!r} not found in this run's logged history"
        parent = rec.get("parent_id")
        if parent is None:
            return hops, "ok"
        cur = parent
        hops += 1
        if hops > max_hops:
            return None, f"exceeded {max_hops} hops (likely a cycle)"


def lineage_table(run: RunData) -> List[dict]:
    if not run.generations:
        return []
    registry = build_lineage_registry(run)
    cells = run.generations[-1].get("cells") or {}
    rows = []
    for label, c in sorted(cells.items()):
        did = c.get("design_id")
        if did is None:
            depth, note = None, "missing design_id"
        else:
            depth, note = compute_depth(did, registry)
        if depth is None:
            warn(f"{run.name}: lineage depth for cell {label} (design {did}) could not be resolved: {note}")
        rows.append({
            "cell": label, "design_id": did, "founder_id": c.get("founder_id"),
            "parent_id": c.get("parent_id"), "generation_born": c.get("generation_born"),
            "digit_count": c.get("digit_count"), "joint_count": c.get("joint_count"),
            "fitness": c.get("fitness"), "depth_generations": depth, "depth_note": note,
        })
    return rows


# --------------------------------------------------------------------------
# Markdown / CSV / JSON writers
# --------------------------------------------------------------------------


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def write_final_table_md(rows: Sequence[dict], probe_hands: Sequence[str]) -> List[str]:
    lines = []
    fields = ["seed", "final_generation", "coverage", "qd_score", "best_fitness", "mean_fitness",
              "median_fitness", "n_distinct_founders", "max_founder_share", "mean_joint_count",
              "max_joint_count", "total_wall_time_h", "gpu_hours_approx"]
    fields += [f"probe_{h}_fitness" for h in probe_hands]
    header = "| metric | " + " | ".join(f"seed {r.get('seed')}" for r in rows) + " | mean | range |"
    sep = "|---" * (len(rows) + 3) + "|"
    lines.append(header)
    lines.append(sep)
    for f_ in fields:
        if f_ == "seed":
            continue
        vals = [r.get(f_) for r in rows]
        numeric = [v for v in vals if isinstance(v, (int, float))]
        mean_s = _fmt(float(np.mean(numeric))) if numeric else "-"
        range_s = _fmt(float(max(numeric) - min(numeric))) if numeric else "-"
        lines.append(f"| {f_} | " + " | ".join(_fmt(v) for v in vals) + f" | {mean_s} | {range_s} |")
    trunc_flags = [r.get("final_gen_possibly_truncated") for r in rows]
    if any(trunc_flags):
        which = [r.get("run") for r, t in zip(rows, trunc_flags) if t]
        lines.append("")
        lines.append(f"**Warning:** final generation possibly truncated/failed for: {', '.join(which)} "
                      f"(nonzero exit code or a single training window) -- treat that seed's final-generation "
                      f"numbers as low-confidence.")
    return lines


def write_lineage_md(run: RunData, rows: Sequence[dict]) -> List[str]:
    if not rows:
        return ["_(no elites / no generations logged)_"]
    lines = ["| cell | design_id | founder_id | generation_born | depth (generations) | fitness |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        depth_s = _fmt(r["depth_generations"]) if r["depth_generations"] is not None else f"unresolved ({r['depth_note']})"
        lines.append(f"| {r['cell']} | {r['design_id']} | {r['founder_id']} | {r['generation_born']} | "
                     f"{depth_s} | {_fmt(r['fitness'])} |")
    depths = [r["depth_generations"] for r in rows if r["depth_generations"] is not None]
    if depths:
        lines.append("")
        lines.append(f"_Depth: mean {np.mean(depths):.2f}, max {max(depths)}, "
                     f"{sum(1 for d in depths if d == 0)}/{len(rows)} elites are still their own founder (depth 0)._")
    unresolved = sum(1 for r in rows if r["depth_generations"] is None)
    if unresolved:
        lines.append(f"_{unresolved}/{len(rows)} elite(s) have an unresolved depth -- see summary.json's "
                     f"'depth_note' field._")
    return lines


def build_markdown(run_dirs_input: Sequence[str], groups: Dict[str, List[RunData]],
                    plot_paths: Dict[str, Dict[str, str]], final_rows_by_variant: Dict[str, List[dict]],
                    aggregates: Dict[str, dict], lineage_by_run: Dict[str, List[dict]],
                    probe_hands: Sequence[str]) -> str:
    lines: List[str] = []
    lines.append("# Evolution pilot analysis")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z by "
                 f"`experiments/evolution_pilot/analyze_pilot.py`.")
    lines.append("")
    lines.append("Run directories analyzed:")
    for d in run_dirs_input:
        lines.append(f"- `{d}`")
    lines.append("")
    lines.append("**Honest statistics.** Variants here have at most 2 seeds. Every per-variant number below "
                 "is reported as the per-seed values, their mean, and their range (max - min) -- never a "
                 "standard error, a confidence interval, or a significance claim. With n=2, a range is all "
                 "the data supports.")
    lines.append("")

    if WARNINGS:
        lines.append("## Data-quality warnings")
        lines.append("")
        for w in WARNINGS:
            lines.append(f"- {w}")
        lines.append("")

    for variant in sorted(groups):
        runs = groups[variant]
        lines.append(f"## Variant `{variant}` ({len(runs)} seed(s): {[r.seed for r in runs]})")
        lines.append("")

        lines.append("### Per-generation curves")
        lines.append("")
        pp = plot_paths[variant]
        captions = [
            ("coverage", "Archive coverage (cells filled of 30)"),
            ("qd_score", "QD-score"),
            ("fitness", "Best and mean elite fitness"),
            ("founders", "Distinct founders and max founder share"),
            ("joints", "Mean and max joint count of the elites"),
            ("digit_distribution", "Digit-count distribution of the elites"),
            ("probes", "Probe-hand fitness"),
        ]
        for key, caption in captions:
            lines.append(f"**{caption}**")
            lines.append("")
            lines.append(f"![{caption}]({pp[key]})")
            lines.append("")

        lines.append("### Final-generation table")
        lines.append("")
        rows = final_rows_by_variant.get(variant, [])
        if not rows:
            lines.append("_no runs with logged generations in this variant_")
        else:
            lines.extend(write_final_table_md(rows, probe_hands))
        lines.append("")

        lines.append("### Final archive occupancy (heatmap)")
        lines.append("")
        lines.append(f"![archive heatmap]({pp['heatmap']})")
        lines.append("")

        lines.append("### Lineage (final archive)")
        lines.append("")
        for run in runs:
            lines.append(f"**{run.name}** (seed {run.seed})")
            lines.append("")
            lines.extend(write_lineage_md(run, lineage_by_run.get(run.name, [])))
            lines.append("")

    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="one or more evolution-pilot run directories "
                    "(each holding generations.jsonl/state.json)")
    ap.add_argument("--out-dir", required=True, help="output directory for summary.md/json/csv and plots/ "
                    "(created if missing; should be git-ignored -- do not point this at a tracked path)")
    ap.add_argument("--probes", default=",".join(PROBE_HANDS_DEFAULT),
                    help="comma-separated probe hand ids expected in every generation's 'probes' report "
                    f"(default: {','.join(PROBE_HANDS_DEFAULT)})")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    probe_hands = tuple(h for h in args.probes.split(",") if h)
    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: List[RunData] = []
    for d in args.run_dirs:
        run_dir = Path(d)
        if not run_dir.is_dir():
            warn(f"{run_dir}: not a directory; skipped")
            continue
        runs.append(load_run(run_dir))

    if not runs:
        warn("no valid run directories given; nothing to analyze")

    groups = group_by_variant(runs)

    plot_paths: Dict[str, Dict[str, str]] = {}
    final_rows_by_variant: Dict[str, List[dict]] = {}
    aggregates: Dict[str, dict] = {}
    lineage_by_run: Dict[str, List[dict]] = {}
    all_final_rows: List[dict] = []

    for variant, vruns in groups.items():
        check_unequal_lengths(variant, vruns)

        paths = plot_all_curves(variant, vruns, probe_hands, plots_dir)
        heatmap_path = plots_dir / variant / "archive_heatmap.png"
        plot_archive_heatmap(variant, vruns, heatmap_path)
        paths["heatmap"] = str(heatmap_path.relative_to(plots_dir.parent))
        plot_paths[variant] = paths

        rows = [r for r in (final_generation_row(run, probe_hands) for run in vruns) if r is not None]
        final_rows_by_variant[variant] = rows
        all_final_rows.extend(rows)
        aggregates[variant] = aggregate_variant(rows, probe_hands)

        for run in vruns:
            lineage_by_run[run.name] = lineage_table(run)

    md = build_markdown(args.run_dirs, groups, plot_paths, final_rows_by_variant, aggregates,
                         lineage_by_run, probe_hands)
    (out_dir / "summary.md").write_text(md)

    write_csv(out_dir / "summary.csv", all_final_rows)

    summary_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z",
        "run_dirs_input": list(args.run_dirs),
        "probe_hands": list(probe_hands),
        "warnings": list(WARNINGS),
        "variants": {
            variant: {
                "seeds": [r.seed for r in vruns],
                "run_dirs": [str(r.run_dir) for r in vruns],
                "generations": {r.name: r.generations for r in vruns},
                "final_generation_rows": final_rows_by_variant.get(variant, []),
                "aggregate": aggregates.get(variant, {}),
                "lineage": {r.name: lineage_by_run.get(r.name, []) for r in vruns},
                "plots": plot_paths.get(variant, {}),
            }
            for variant, vruns in groups.items()
        },
        "final_generation_rows_all": all_final_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=1, sort_keys=True))

    print(f"[analyze_pilot] wrote {out_dir}/summary.{{md,json,csv}} and {plots_dir}/ "
          f"({len(groups)} variant(s), {len(runs)} run(s), {len(WARNINGS)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
