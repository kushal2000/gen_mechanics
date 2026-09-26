"""E4: redundancy of the grammar's genotype -> phenotype map, for each of
``G_FULL`` and ``G_SERIAL``.

Three measurements, all keyed by seed so they parallelize the same way
every other E0 experiment does (``experiments.runner.run_experiment`` maps
one function over a list of seeds):

1. Genotype-space redundancy: among 10,000 random seeds, what fraction of
   ``phenotype_hash`` values are distinct (a hash collision means two
   different seeds derived to the identical canonical phenotype).
2. Mutational redundancy: for the first 1000 of those seeds (used as
   parents), 10 offspring each (``vary(parent, rng, dist, operators=...)``,
   once for the default operators, once for the small-step operators). I14
   fix 10: two SEPARATE fractions are now reported, previously merged into
   one "null_fraction" (the review found the merged rate was almost
   entirely the ``VariationImpossible`` case, not genuine phenotypic
   inertness):
   - ``null_fraction``: ``vary`` SUCCEEDED and the offspring's hash equals
     its own parent's (a genuinely null, phenotypically-inert mutation).
   - ``impossible_fraction``: every operator in the pool raised
     ``VariationImpossible`` (``vary`` never ran at all).
   Also reported: the fraction of same-parent offspring PAIRS that hash
   identically to each other (an impossible offspring counts, for this pair
   count only, as identical to the parent -- see ``_offspring_stats``).
3. Encoding order: the fraction of models whose canonical joint order
   (``canonical.canonical_form``) differs from the order ``derive()`` itself
   built them in -- i.e. how often the derivation's own step order was
   already the DFS-canonical one vs. needed reordering. Compared by joint
   *key* (type + rounded origin/axis/limits) sequence, not by name, so a
   coincidental rename never counts as "no reordering" when the underlying
   sequence did change, and never counts as "reordered" on its own. I14 fix
   10: this is NOT a redundancy measure (dropped from that framing in the
   summary text below) -- it says nothing about how many genotypes map to
   one phenotype, only about derivation step order vs. a fixed canonical
   order.

seed 0..9999 for (1); the sub-range 0..999 additionally carries (2)'s
offspring for that seed's own parent; (3) is measured on every one of the
10,000 seeds' own model (no extra derivations needed for it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..canonical import canonical_form, phenotype_hash
from ..derive import OPERATORS, SMALL_STEP_OPERATORS, VariationImpossible, derive, generate, vary
from ..kinematics import KinematicModel
from ..variants import G_FULL, G_SERIAL
from .runner import register, run_experiment

VARIANTS: Dict[str, Any] = {"G_FULL": G_FULL, "G_SERIAL": G_SERIAL}
OFFSPRING_OP_GROUPS: Dict[str, Tuple[str, ...]] = {
    "default": tuple(OPERATORS),
    "small": tuple(SMALL_STEP_OPERATORS),
}
N_OFFSPRING = 10
N_OFFSPRING_PARENTS = 1000


def _joint_key(j) -> tuple:
    return (
        j.type,
        tuple(round(float(v), 9) for v in j.origin.xyz),
        tuple(round(float(v), 9) for v in j.origin.rpy),
        tuple(round(float(v), 9) for v in j.axis),
        tuple(round(float(v), 9) for v in j.limits) if j.limits is not None else None,
    )


def _canonical_reorders(model: KinematicModel) -> bool:
    """True iff the sequence of joint keys in ``model.joints`` (the order
    ``derive()`` itself built them in) differs from the sequence in
    ``canonical_form(model).joints`` (the DFS-canonical order) -- i.e.
    ``canonical_form`` actually moved something, as opposed to the
    derivation already having produced joints in canonical order."""
    canon = canonical_form(model)
    orig_keys = [_joint_key(j) for j in model.joints]
    canon_keys = [_joint_key(j) for j in canon.joints]
    return orig_keys != canon_keys


def _offspring_stats(parent_derivation, parent_hash: str, dist, ops: Tuple[str, ...],
                      rng: np.random.Generator) -> Dict[str, Any]:
    """I14 fix 10: ``impossible_fraction`` (every application in ``ops``
    raised ``VariationImpossible``) is now reported SEPARATELY from
    ``null_fraction`` (``vary`` SUCCEEDED and the resulting hash equals the
    parent's) -- previously merged into one ``null_fraction`` (the review
    found the old "null mutation" rate was almost entirely
    ``VariationImpossible``, not a genuine phenotypically-inert mutation)."""
    hashes: List[str] = []
    null_flags: List[float] = []
    impossible_flags: List[float] = []
    for _ in range(N_OFFSPRING):
        try:
            child_derivation = vary(parent_derivation, rng, dist, operators=ops)
            child_hash = phenotype_hash(derive(child_derivation))
            impossible_flags.append(0.0)
            null_flags.append(1.0 if child_hash == parent_hash else 0.0)
        except VariationImpossible:
            # No operator in ``ops`` had any valid application at all; not
            # measured as a null mutation (``vary`` never even ran), only as
            # "impossible". For the pairwise-identical-hash count below, an
            # impossible offspring is conventionally the (unchanged) parent
            # -- consistent with ``e2_drift``'s own convention for a
            # rejected proposal -- since it IS the same phenotype.
            child_hash = parent_hash
            impossible_flags.append(1.0)
        hashes.append(child_hash)
    n_pairs = 0
    n_identical_pairs = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            n_pairs += 1
            if hashes[i] == hashes[j]:
                n_identical_pairs += 1
    return {
        "null_fraction": float(np.mean(null_flags)) if null_flags else None,
        "impossible_fraction": float(np.mean(impossible_flags)) if impossible_flags else None,
        "n_identical_pairs": n_identical_pairs,
        "n_pairs": n_pairs,
    }


def e4_redundancy_seed(seed: int, n_offspring_parents: int = N_OFFSPRING_PARENTS) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for v_idx, (variant_name, dist) in enumerate(VARIANTS.items()):
        derivation, model = generate(seed, dist)
        h = phenotype_hash(model)
        entry: Dict[str, Any] = {
            "hash": h,
            "canonical_reordered": 1.0 if _canonical_reorders(model) else 0.0,
        }
        if seed < n_offspring_parents:
            offspring: Dict[str, Any] = {}
            for g_idx, (group_name, ops) in enumerate(OFFSPRING_OP_GROUPS.items()):
                rng = np.random.default_rng([seed, v_idx, g_idx, 4])
                offspring[group_name] = _offspring_stats(derivation, h, dist, ops, rng)
            entry["offspring"] = offspring
        out[variant_name] = entry
    return out


register("e4_redundancy", e4_redundancy_seed)


def _aggregate(per_seed: List[Dict[str, Any]], n_offspring_parents: int) -> Dict[str, Any]:
    ok = [r["result"] for r in per_seed if r["ok"]]
    agg: Dict[str, Any] = {"n_seeds": len(ok)}
    for variant_name in VARIANTS:
        hashes = [r[variant_name]["hash"] for r in ok]
        reordered = [r[variant_name]["canonical_reordered"] for r in ok]
        distinct_fraction = (len(set(hashes)) / len(hashes)) if hashes else None
        reorder_fraction = float(np.mean(reordered)) if reordered else None
        offspring_agg: Dict[str, Any] = {}
        for group_name in OFFSPRING_OP_GROUPS:
            rows = [r[variant_name]["offspring"][group_name] for r in ok if "offspring" in r[variant_name]]
            null_fracs = [row["null_fraction"] for row in rows if row["null_fraction"] is not None]
            impossible_fracs = [row["impossible_fraction"] for row in rows if row["impossible_fraction"] is not None]
            total_pairs = sum(row["n_pairs"] for row in rows)
            total_identical = sum(row["n_identical_pairs"] for row in rows)
            offspring_agg[group_name] = {
                "offspring_null_fraction": float(np.mean(null_fracs)) if null_fracs else None,
                "offspring_impossible_fraction": float(np.mean(impossible_fracs)) if impossible_fracs else None,
                "pairwise_identical_fraction": (total_identical / total_pairs) if total_pairs else None,
                "n_parents": len(rows),
            }
        agg[variant_name] = {
            "distinct_hash_fraction": distinct_fraction,
            "canonical_reorder_fraction": reorder_fraction,
            "offspring": offspring_agg,
            "n": len(hashes),
        }
    return agg


def _fmt(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _summary_md(params: Dict[str, Any], aggregate: Dict[str, Any], wall_time_s: float) -> str:
    lines = [
        "# Experiment: e4_redundancy", "",
        f"Wall time: {wall_time_s:.3f} s", "",
        "## Params", "", "```json", json.dumps(params, indent=2, sort_keys=True, default=str), "```", "",
        f"n_seeds: {aggregate.get('n_seeds')}", "",
        "## Table: variant x metric", "",
        "| variant | distinct_hash_fraction (10k seeds) | derivation order differs "
        "from canonical order (fraction) | default: null_fraction | default: impossible_fraction | "
        "default: pairwise_identical_fraction | small: null_fraction | small: impossible_fraction | "
        "small: pairwise_identical_fraction |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for variant_name in VARIANTS:
        a = aggregate[variant_name]
        d = a["offspring"]["default"]
        s = a["offspring"]["small"]
        lines.append(
            f"| {variant_name} | {_fmt(a['distinct_hash_fraction'])} | {_fmt(a['canonical_reorder_fraction'])} | "
            f"{_fmt(d['offspring_null_fraction'])} | {_fmt(d['offspring_impossible_fraction'])} | "
            f"{_fmt(d['pairwise_identical_fraction'])} | "
            f"{_fmt(s['offspring_null_fraction'])} | {_fmt(s['offspring_impossible_fraction'])} | "
            f"{_fmt(s['pairwise_identical_fraction'])} |"
        )
    lines += ["", "## Reading", ""]
    for variant_name in VARIANTS:
        a = aggregate[variant_name]
        d = a["offspring"]["default"]
        s = a["offspring"]["small"]
        lines.append(
            f"- {variant_name}: distinct_hash_fraction {_fmt(a['distinct_hash_fraction'])} (n={a['n']}); "
            f"fraction of models whose derivation order differs from canonical order "
            f"{_fmt(a['canonical_reorder_fraction'])} (NOT a redundancy measure -- see e4_redundancy.py's "
            f"module docstring); "
            f"default ops null_fraction {_fmt(d['offspring_null_fraction'])} "
            f"(hash equal AND vary succeeded), impossible_fraction {_fmt(d['offspring_impossible_fraction'])}, "
            f"pairwise_identical_fraction {_fmt(d['pairwise_identical_fraction'])} (n_parents={d['n_parents']}); "
            f"small-step ops null_fraction {_fmt(s['offspring_null_fraction'])}, "
            f"impossible_fraction {_fmt(s['offspring_impossible_fraction'])}, "
            f"pairwise_identical_fraction {_fmt(s['pairwise_identical_fraction'])} (n_parents={s['n_parents']})."
        )
    lines.append("")
    return "\n".join(lines)


def run(out_dir: Optional[str] = None, seeds: Sequence[int] = range(10000),
        n_offspring_parents: int = N_OFFSPRING_PARENTS, processes: int = 24) -> Dict[str, Any]:
    if out_dir is None:
        out_dir = "project-notes/grammar/experiments/E4_redundancy"
    # ``params`` is passed as ``**kwargs`` to ``e4_redundancy_seed`` by the
    # runner, so only its actual keyword arguments belong here;
    # ``n_offspring`` (a fixed module constant, not a per-seed parameter) is
    # recorded into ``result["params"]`` afterward, for documentation only.
    params = {"n_offspring_parents": n_offspring_parents}
    result = run_experiment(
        "e4_redundancy", e4_redundancy_seed, params=params, seeds=seeds, out_dir=out_dir, processes=processes,
    )
    result["params"] = {**params, "n_offspring": N_OFFSPRING}
    aggregate = _aggregate(result["per_seed"], n_offspring_parents)
    result["aggregate"] = aggregate
    out_path = Path(out_dir)
    (out_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    (out_path / "summary.md").write_text(_summary_md(result["params"], aggregate, result["wall_time_s"]))
    return result


if __name__ == "__main__":
    run()
