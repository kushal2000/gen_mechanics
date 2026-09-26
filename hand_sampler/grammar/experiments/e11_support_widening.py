"""E11: how far the grammar's own sampling support has to widen, one named
relaxation at a time, before every real hand in ``grammar_bench/manifest.json``
falls inside it -- and what that widening costs in design-space bits per
joint.

Uses ``hand_sampler.grammar.coverage.coverage(model, dist, relax=...)``'s E11
``relax`` argument (see that module for the full set of named relaxations and
their semantics). This script never changes ``rules.py``/``distributions.py``
sampling behaviour; it only asks, after the fact, how much of the real-hand
gap each named relaxation closes.

Cumulative levels (L0..L8), exactly as specified for E11:
    L0 = {}                                            (current grammar)
    L1 = L0 + {rest_bend}
    L2 = L1 + {limits_continuous}
    L3 = L2 + {length_grid_1mm}
    L4 = L3 + {axis_grid_5deg}
    L5 = L4 + {coupling_continuous}
    L6 = L5 + {length_continuous, axis_continuous}
    L7 = L6 + {limits_range_x1.5, length_range_x1.5}
    L8 = L7 + {fixed_in_digit_ok, children_unbounded}

Outputs (under ``project-notes/grammar/experiments/E11_support_widening/``):
    ``result.json``/``summary.md`` -- via ``runner.run_experiment`` (git
    provenance, dirty-tree gate).
    ``e11_report.md`` -- the three required tables (hands x levels,
    single-relaxation unblocking, bits-per-joint) plus the hardest hands'
    remaining items at L8, computed once (independent of the seed
    ``run_experiment`` maps over -- the hands table and the bits table are
    deterministic; only the 200-seed generated-design check consumes RNG
    seeds).

Real hands are resolved exactly as ``grammar_bench/evaluate.py`` does
(``fixture_path``, then ``REPO:``-prefixed ``source_path``, then a
local-only ``source_path`` under the manifest's ``source_root`` -- overridable
here with the ``HAND_URDF_ROOT`` environment variable, default
``/home/singularity/karma/karma-data/all_urdfs/full_models_as_downloaded``,
matching this machine's manifest ``source_root``). ``excluded``-split hands
are never imported. Held-out hands are reported exactly like dev-split hands
-- this script never changes a rule/range/tolerance in response to either.

Stdlib + numpy + this project's own ``hand_sampler`` package only.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

from ..adapters.urdf import load_urdf
from ..coverage import (
    CoverageResult,
    LENGTH_GRID_1MM_M,
    RELAX_NAMES,
    _widen_range,
    coverage,
    global_prismatic_limit_range_m,
    global_revolute_limit_range_deg,
)
from ..derive import generate
from ..distributions import ANGLE_STEP_DEG, DEFAULT_DISTRIBUTION, DEG, Distribution, N_ANGLE_STEPS, N_ELEVATION_STEPS
from .runner import register, run_experiment

BENCH_DIR = Path(__file__).resolve().parents[3] / "hand_sampler" / "grammar_bench"
# I16 fix: this file lives at <repo>/hand_sampler/grammar/experiments/, so
# ``parents[3]`` (matching BENCH_DIR's own reference point above) is the
# repo root -- previously ``parents[4]`` (one level too high, the PARENT of
# the repo), which silently made every ``REPO:``-prefixed manifest source
# (SHARPA) resolve to a nonexistent path and be reported "unavailable",
# unlike ``grammar_bench/evaluate.py``'s own ``REPO_ROOT = BENCH_DIR.parent.parent``
# (== this same ``parents[3]``), which this now matches exactly.
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = BENCH_DIR / "manifest.json"
DEFAULT_HAND_URDF_ROOT = "/home/singularity/karma/karma-data/all_urdfs/full_models_as_downloaded"

N_GENERATED_SEEDS_DEFAULT = 200

# Cumulative relaxation levels, in the exact order specified for E11.
LEVEL_ADDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("L0", ()),
    ("L1", ("rest_bend",)),
    ("L2", ("limits_continuous",)),
    ("L3", ("length_grid_1mm",)),
    ("L4", ("axis_grid_5deg",)),
    ("L5", ("coupling_continuous",)),
    ("L6", ("length_continuous", "axis_continuous")),
    ("L7", ("limits_range_x1.5", "length_range_x1.5")),
    ("L8", ("fixed_in_digit_ok", "children_unbounded")),
)


def cumulative_levels() -> Dict[str, FrozenSet[str]]:
    """``{"L0": frozenset(), "L1": frozenset({"rest_bend"}), ...}``, each
    level's relax set being the union of its own row and every row above it
    in ``LEVEL_ADDS`` -- i.e. genuinely cumulative, as specified."""
    out: Dict[str, FrozenSet[str]] = {}
    acc: set = set()
    for name, adds in LEVEL_ADDS:
        acc |= set(adds)
        out[name] = frozenset(acc)
    return out


LEVELS: Dict[str, FrozenSet[str]] = cumulative_levels()
LEVEL_ORDER: Tuple[str, ...] = tuple(name for name, _ in LEVEL_ADDS)


# ---------------------------------------------------------------------------
# Hand resolution -- mirrors grammar_bench/evaluate.py::_resolve_hand exactly,
# with the addition of a HAND_URDF_ROOT env override for the local-only
# source_root (evaluate.py always reads manifest["source_root"] only; this
# script is explicitly asked to honour an env override too).
# ---------------------------------------------------------------------------


def _resolve_hand(hand: dict, manifest: dict) -> Tuple[Optional[Path], str, Optional[str]]:
    if hand.get("split") == "excluded":
        return None, "excluded", hand.get("notes") or "excluded split: never imported"

    fixture_path = hand.get("fixture_path")
    if fixture_path:
        p = BENCH_DIR / fixture_path
        if p.is_file():
            return p, "available", None
        return None, "unavailable", f"fixture_path listed in manifest but missing on disk: {fixture_path}"

    source_path = hand.get("source_path")
    if isinstance(source_path, str) and source_path.startswith("REPO:"):
        p = REPO_ROOT / source_path[len("REPO:"):]
        if p.is_file():
            return p, "available", None
        return None, "unavailable", f"REPO path listed in manifest but missing on disk: {source_path}"

    if isinstance(source_path, str) and source_path:
        source_root = os.environ.get("HAND_URDF_ROOT") or manifest.get("source_root") or DEFAULT_HAND_URDF_ROOT
        p = Path(source_root) / source_path
        if p.is_file():
            return p, "available", None
        return None, "unavailable", f"local-only:{hand['id']} source not present under HAND_URDF_ROOT={source_root!r}"

    return None, "unavailable", "no fixture_path/source_path in manifest"


def _load_hand_models() -> List[dict]:
    """Every manifest hand, resolved and imported once. Returns a list of
    dicts with id/family/split/model (model is ``None`` when unavailable, with
    ``reason`` explaining why -- never scored, exactly like evaluate.py)."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    out = []
    for hand in manifest["hands"]:
        entry = {"id": hand["id"], "family": hand.get("family"), "split": hand.get("split"), "model": None, "reason": None}
        path, availability, reason = _resolve_hand(hand, manifest)
        if availability != "available":
            entry["reason"] = f"{availability}: {reason}"
            out.append(entry)
            continue
        try:
            result = load_urdf(path, hand_root=hand.get("hand_root"))
        except Exception as exc:  # noqa: BLE001
            entry["reason"] = f"import failed: {exc}"
            out.append(entry)
            continue
        entry["model"] = result.model
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Hands x levels table
# ---------------------------------------------------------------------------


def _remaining_count(res: CoverageResult) -> int:
    return len(res.missing_constructs) + len(res.out_of_support)


def hands_x_levels(dist: Distribution = DEFAULT_DISTRIBUTION) -> dict:
    hands = _load_hand_models()
    rows = []
    for entry in hands:
        row = {"id": entry["id"], "family": entry["family"], "split": entry["split"]}
        if entry["model"] is None:
            row["available"] = False
            row["reason"] = entry["reason"]
            row["levels"] = None
            rows.append(row)
            continue
        row["available"] = True
        levels = {}
        for level_name in LEVEL_ORDER:
            res = coverage(entry["model"], dist, relax=LEVELS[level_name])
            levels[level_name] = {
                "in_support": res.in_support,
                "topology_expressible": res.topology_expressible,
                "remaining": _remaining_count(res),
                "missing_constructs": list(res.missing_constructs),
                "out_of_support": list(res.out_of_support),
            }
        row["levels"] = levels
        rows.append(row)
    return {"levels": LEVEL_ORDER, "relax_by_level": {k: sorted(v) for k, v in LEVELS.items()}, "hands": rows}


# ---------------------------------------------------------------------------
# Single-relaxation unblocking table: for each of the 11 named relaxations
# applied ALONE (never combined with any other), how many available hands
# flip from not-in-support (at L0, i.e. no relaxation) to in-support?
# ---------------------------------------------------------------------------


def unblocking_table(dist: Distribution = DEFAULT_DISTRIBUTION) -> dict:
    hands = [e for e in _load_hand_models() if e["model"] is not None]
    baseline = {e["id"]: coverage(e["model"], dist, relax=frozenset()).in_support for e in hands}
    n_blocked_at_l0 = sum(1 for v in baseline.values() if not v)
    counts = {}
    unblocked_by = {}
    for name in RELAX_NAMES:
        unblocked = [
            e["id"] for e in hands
            if not baseline[e["id"]] and coverage(e["model"], dist, relax=frozenset({name})).in_support
        ]
        counts[name] = len(unblocked)
        unblocked_by[name] = unblocked
    ranked = sorted(RELAX_NAMES, key=lambda n: (-counts[n], n))

    # From-L0 unblocking is, in practice, uninformative on its own: EVERY
    # available hand also fails on continuation_pose (mid-digit rpy/lateral
    # offset), so no relaxation OTHER than rest_bend can ever flip a hand to
    # in_support by itself (topology_expressible stays False regardless of
    # what the parameter-support checks say) -- see ``counts`` above, which
    # is honestly all zero except for "rest_bend" itself. The more
    # actionable question is which SINGLE additional relaxation, once
    # rest_bend (the universal prerequisite) is already applied, unblocks
    # the most hands -- computed the same way but from an L1 baseline.
    baseline_l1 = {e["id"]: coverage(e["model"], dist, relax=frozenset({"rest_bend"})).in_support for e in hands}
    n_blocked_at_l1 = sum(1 for v in baseline_l1.values() if not v)
    counts_from_l1, unblocked_by_from_l1 = {}, {}
    other_names = [n for n in RELAX_NAMES if n != "rest_bend"]
    for name in other_names:
        unblocked = [
            e["id"] for e in hands
            if not baseline_l1[e["id"]]
            and coverage(e["model"], dist, relax=frozenset({"rest_bend", name})).in_support
        ]
        counts_from_l1[name] = len(unblocked)
        unblocked_by_from_l1[name] = unblocked
    ranked_from_l1 = sorted(other_names, key=lambda n: (-counts_from_l1[n], n))

    return {
        "n_available_hands": len(hands),
        "n_blocked_at_l0": n_blocked_at_l0,
        "counts": counts,
        "unblocked_hand_ids": unblocked_by,
        "ranked": ranked,
        "n_blocked_at_l1": n_blocked_at_l1,
        "counts_from_l1": counts_from_l1,
        "unblocked_hand_ids_from_l1": unblocked_by_from_l1,
        "ranked_from_l1": ranked_from_l1,
    }


# ---------------------------------------------------------------------------
# Generated-design check: every one of N seeded grammar derivations must stay
# in_support under every cumulative level (monotonicity means it is enough
# that they are in_support at L0; every wider level can only keep them in).
# ---------------------------------------------------------------------------


def generated_designs_check(seed0: int = 0, n: int = N_GENERATED_SEEDS_DEFAULT,
                             dist: Distribution = DEFAULT_DISTRIBUTION) -> dict:
    per_level_all_in_support = {name: True for name in LEVEL_ORDER}
    counterexamples: Dict[str, List[int]] = {name: [] for name in LEVEL_ORDER}
    for i in range(n):
        seed = seed0 + i
        _derivation, model = generate(seed, dist)
        for level_name in LEVEL_ORDER:
            res = coverage(model, dist, relax=LEVELS[level_name])
            if not res.in_support:
                per_level_all_in_support[level_name] = False
                counterexamples[level_name].append(seed)
    return {
        "n_seeds": n,
        "seed0": seed0,
        "all_in_support_by_level": per_level_all_in_support,
        "counterexample_seeds_by_level": counterexamples,
    }


# ---------------------------------------------------------------------------
# Bits-per-joint design-space-size cost of each level.
# ---------------------------------------------------------------------------


def _bits(n_choices: float) -> float:
    return math.log2(n_choices)


def bits_per_joint_table(dist: Distribution = DEFAULT_DISTRIBUTION) -> dict:
    """One row per level: the number of discrete choices (or 'continuous')
    for each of the four sampled-parameter families a relaxation can touch --
    revolute-joint limit pairs, prismatic-joint limit pairs, per-body link
    length, and joint axis direction -- and the coupling multiplier/offset
    pair. Lengths are reported for ``dist.link_length_range_m`` (phalanx
    bodies; palm bodies draw from a slightly narrower ``palm_length_range_m``
    and would report a marginally different grid count under the same
    relaxations, omitted here for brevity)."""
    revolute_lo, revolute_hi = global_revolute_limit_range_deg(dist)
    n_revolute_choices = len(set(dist.revolute_limit_choices_deg) | set(dist.palm_joint_limit_choices_deg))
    prismatic_lo, prismatic_hi = global_prismatic_limit_range_m(dist)
    n_prismatic_choices = len(dist.prismatic_limit_choices_m)

    link_lo, link_hi = dist.link_length_range_m
    n_length_choices_5mm = int(round((link_hi - link_lo) / dist.link_length_grid_m)) + 1
    n_length_choices_1mm = int(round((link_hi - link_lo) / LENGTH_GRID_1MM_M)) + 1

    n_axis_choices_15deg = N_ANGLE_STEPS * N_ELEVATION_STEPS
    n_axis_choices_5deg = (360 // 5) * (180 // 5 + 1)

    n_coupling_mult_choices = len(dist.coupling_multiplier_choices)
    n_coupling_offset_choices = len(dist.coupling_offset_choices_rad)

    def _limits_bits(level: str) -> Tuple[str, str]:
        rx = LEVELS[level]
        if "limits_continuous" in rx or "limits_range_x1.5" in rx:
            lo, hi = revolute_lo, revolute_hi
            if "limits_range_x1.5" in rx:
                lo, hi = _widen_range(lo, hi, 1.5)
            return f"continuous in [{lo:.1f}, {hi:.1f}] deg (2 continuous DOF: lo, hi)", "continuous"
        return f"{n_revolute_choices} choices ({_bits(n_revolute_choices):.3f} bits)", f"{n_prismatic_choices} choices ({_bits(n_prismatic_choices):.3f} bits)"

    def _length_bits(level: str) -> str:
        rx = LEVELS[level]
        lo, hi = link_lo, link_hi
        if "length_range_x1.5" in rx:
            lo, hi = _widen_range(lo, hi, 1.5)
        if "length_continuous" in rx:
            return f"continuous in [{lo:.4f}, {hi:.4f}] m"
        grid = LENGTH_GRID_1MM_M if "length_grid_1mm" in rx else dist.link_length_grid_m
        n = int(round((hi - lo) / grid)) + 1
        return f"{n} choices ({_bits(n):.3f} bits), grid={grid*1000:.0f}mm"

    def _axis_bits(level: str) -> str:
        rx = LEVELS[level]
        if "axis_continuous" in rx:
            return "continuous (unit sphere, 2 continuous DOF)"
        n = n_axis_choices_5deg if "axis_grid_5deg" in rx else n_axis_choices_15deg
        step = 5.0 if "axis_grid_5deg" in rx else ANGLE_STEP_DEG
        return f"{n} choices ({_bits(n):.3f} bits), grid={step:.0f}deg"

    def _coupling_bits(level: str) -> str:
        rx = LEVELS[level]
        if "coupling_continuous" in rx:
            return "continuous: multiplier in [-2,2], offset in [-1,1] rad"
        return (
            f"{n_coupling_mult_choices} mult choices ({_bits(n_coupling_mult_choices):.3f} bits) x "
            f"{n_coupling_offset_choices} offset choices ({_bits(n_coupling_offset_choices):.3f} bits)"
        )

    rows = []
    for level in LEVEL_ORDER:
        rev_str, pris_str = _limits_bits(level)
        rows.append({
            "level": level,
            "relax": sorted(LEVELS[level]),
            "revolute_limits": rev_str,
            "prismatic_limits": pris_str,
            "link_length": _length_bits(level),
            "axis": _axis_bits(level),
            "coupling": _coupling_bits(level),
        })
    return {
        "baseline": {
            "revolute_limit_choices": n_revolute_choices,
            "palm_joint_limit_choices": len(dist.palm_joint_limit_choices_deg),
            "prismatic_limit_choices": n_prismatic_choices,
            "link_length_choices_5mm_grid": n_length_choices_5mm,
            "link_length_choices_1mm_grid": n_length_choices_1mm,
            "axis_choices_15deg_grid": n_axis_choices_15deg,
            "axis_choices_5deg_grid": n_axis_choices_5deg,
            "coupling_mult_choices": n_coupling_mult_choices,
            "coupling_offset_choices": n_coupling_offset_choices,
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(hxl: dict, unblock: dict, bits: dict, generated: dict) -> str:
    lines = []
    lines.append("# E11: support widening for real hands")
    lines.append("")
    lines.append(
        "Cumulative relaxation levels applied to `hand_sampler.grammar.coverage.coverage`'s "
        "`relax` argument. Level definitions:"
    )
    lines.append("")
    for name in LEVEL_ORDER:
        lines.append(f"- **{name}** = {{{', '.join(sorted(LEVELS[name])) or '(none)'}}}")
    lines.append("")

    lines.append("## Hands x levels (in_support, remaining-item count)")
    lines.append("")
    header = "| id | family | split | " + " | ".join(LEVEL_ORDER) + " |"
    sep = "|---|---|---|" + "---|" * len(LEVEL_ORDER)
    lines.append(header)
    lines.append(sep)
    for row in hxl["hands"]:
        if not row["available"]:
            cells = ["unavailable"] * len(LEVEL_ORDER)
        else:
            cells = []
            for level in LEVEL_ORDER:
                lv = row["levels"][level]
                cells.append(("yes" if lv["in_support"] else "no") + f" ({lv['remaining']})")
        lines.append(f"| {row['id']} | {row['family'] or '-'} | {row['split']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Cell format: `in_support (remaining out-of-support + missing-construct item count)`. "
        "`unavailable` hands (see reason in JSON) were never scored."
    )
    lines.append("")

    lines.append("## Single-relaxation unblocking (from L0)")
    lines.append("")
    lines.append(
        f"Of {unblock['n_available_hands']} available hands, {unblock['n_blocked_at_l0']} are out of support "
        "at L0 (no relaxation). For each relaxation applied ALONE (not cumulatively), the count of hands it "
        "alone moves from not-in-support to in-support:"
    )
    lines.append("")
    lines.append("| relaxation | hands unblocked | which hands |")
    lines.append("|---|---|---|")
    for name in unblock["ranked"]:
        ids = ", ".join(unblock["unblocked_hand_ids"][name]) or "-"
        lines.append(f"| {name} | {unblock['counts'][name]} | {ids} |")
    lines.append("")
    lines.append(
        "Every count above is 0 except `rest_bend` itself: every available real hand also fails "
        "`continuation_pose` (mid-digit rpy/lateral offset), so `topology_expressible` stays False -- and "
        "therefore `in_support` stays False -- regardless of what any other single relaxation does to the "
        "parameter-support checks. `rest_bend` is a universal prerequisite, not one relaxation among equals."
    )
    lines.append("")

    lines.append("## Single-relaxation unblocking (from L1, i.e. with rest_bend already applied)")
    lines.append("")
    lines.append(
        f"The more informative version: {unblock['n_blocked_at_l1']} of {unblock['n_available_hands']} hands "
        "are still out of support at L1 ({rest_bend} only). For each OTHER relaxation added alone on top of "
        "rest_bend, the count of hands it unblocks:"
    )
    lines.append("")
    lines.append("| relaxation (+ rest_bend) | hands unblocked | which hands |")
    lines.append("|---|---|---|")
    for name in unblock["ranked_from_l1"]:
        ids = ", ".join(unblock["unblocked_hand_ids_from_l1"][name]) or "-"
        lines.append(f"| {name} | {unblock['counts_from_l1'][name]} | {ids} |")
    lines.append("")

    lines.append("## Bits per joint (design-space-size cost) per level")
    lines.append("")
    lines.append(
        "Choice counts for `DEFAULT_DISTRIBUTION`; `continuous` means the relaxation removes the "
        "grid/choice-set constraint entirely (no finite bit count -- reported as the DOF and range instead)."
    )
    lines.append("")
    lines.append("| level | revolute limits | prismatic limits | link length | axis | coupling (mult, offset) |")
    lines.append("|---|---|---|---|---|---|")
    for row in bits["rows"]:
        lines.append(
            f"| {row['level']} | {row['revolute_limits']} | {row['prismatic_limits']} | "
            f"{row['link_length']} | {row['axis']} | {row['coupling']} |"
        )
    lines.append("")

    lines.append("## Generated designs (200 seeds) stay in support at every level")
    lines.append("")
    lines.append(f"Seeds {generated['seed0']}..{generated['seed0'] + generated['n_seeds'] - 1}, `DEFAULT_DISTRIBUTION`:")
    lines.append("")
    lines.append("| level | all in_support | counterexample seeds |")
    lines.append("|---|---|---|")
    for level in LEVEL_ORDER:
        ok = generated["all_in_support_by_level"][level]
        cex = generated["counterexample_seeds_by_level"][level]
        lines.append(f"| {level} | {'yes' if ok else 'no'} | {', '.join(map(str, cex)) or '-'} |")
    lines.append("")

    lines.append("## Remaining items at L8, hardest hands")
    lines.append("")
    hardest = sorted(
        (r for r in hxl["hands"] if r["available"]),
        key=lambda r: -r["levels"]["L8"]["remaining"],
    )
    for r in hardest:
        l8 = r["levels"]["L8"]
        if l8["remaining"] == 0:
            continue
        lines.append(f"- **{r['id']}** ({r['family']}): in_support={l8['in_support']}")
        if l8["missing_constructs"]:
            lines.append(f"  - missing_constructs: {', '.join(l8['missing_constructs'])}")
        if l8["out_of_support"]:
            lines.append(f"  - out_of_support: {', '.join(l8['out_of_support'])}")
    lines.append("")

    lines.append("## Claims / limitations")
    lines.append("")
    lines.append(
        "- Each relaxation is applied via `coverage.py`'s `relax` argument, never by editing "
        "`rules.py`/`distributions.py`; the grammar's own sampling behaviour is unchanged by this script."
    )
    lines.append(
        "- Relaxations named here do not cover every `missing_constructs`/`out_of_support` category "
        "coverage.py can report (e.g. `coupling_limits_outside_image`, `coupling_scope`, `digit_count_out_of_range` "
        "have no corresponding relax flag in this design) -- a hand blocked by one of those stays "
        "out of support at every level; see the L8 remaining-items list above."
    )
    lines.append("- No universality claim is made; held-out-split hands are reported exactly like dev-split hands.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# runner.py registration entry point
# ---------------------------------------------------------------------------


def run_e11(seed: int, n_generated_seeds: int = N_GENERATED_SEEDS_DEFAULT) -> dict:
    """Per-seed function for ``runner.run_experiment``: ``seed`` is the base
    seed for the 200-derivation generated-design check (the hands x levels /
    unblocking / bits tables are deterministic and independent of ``seed``,
    so they are identical across every seed this is mapped over)."""
    hxl = hands_x_levels()
    unblock = unblocking_table()
    bits = bits_per_joint_table()
    generated = generated_designs_check(seed0=seed * n_generated_seeds, n=n_generated_seeds)
    return {
        "n_available_hands": unblock["n_available_hands"],
        "n_blocked_at_l0": unblock["n_blocked_at_l0"],
        "n_in_support_at_l8": sum(
            1 for r in hxl["hands"] if r["available"] and r["levels"]["L8"]["in_support"]
        ),
        "all_generated_in_support_every_level": all(generated["all_in_support_by_level"].values()),
        "_hxl": hxl,
        "_unblock": unblock,
        "_bits": bits,
        "_generated": generated,
    }


register("e11_support_widening", run_e11)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="project-notes/grammar/experiments/E11_support_widening")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-generated-seeds", type=int, default=N_GENERATED_SEEDS_DEFAULT)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_experiment(
        "e11_support_widening",
        run_e11,
        params={"n_generated_seeds": args.n_generated_seeds},
        seeds=[args.seed],
        out_dir=str(out_dir),
        allow_dirty=args.allow_dirty,
    )
    per_seed = result["per_seed"][0]["result"]
    report_md = render_markdown(per_seed["_hxl"], per_seed["_unblock"], per_seed["_bits"], per_seed["_generated"])
    (out_dir / "e11_report.md").write_text(report_md)
    (out_dir / "e11_report.json").write_text(json.dumps(per_seed, indent=2, default=str))
    print(f"wrote {out_dir / 'e11_report.md'} and {out_dir / 'e11_report.json'} (plus result.json/summary.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
