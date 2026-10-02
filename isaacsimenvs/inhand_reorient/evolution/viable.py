"""Viable-only populations (the driver's ``--viable-only``).

A design enters training only if the grasp search (``grasp_cache_gen``, a
Kit launch) found at least one stable grasp for it. Per generation:

1. Elites are viable by construction and keep their cached grasps. Probes
   are searched once; a non-viable probe is dropped from training and listed.
2. New designs are drawn as before (founders in generation 0; offspring of
   archive elites, or an immigrant when a mutation fails), then pre-filtered
   on the CPU (``prefilter_report``: ``grammar_envelope.viability_report``
   admitted, at least ``min_tip_contacts`` digits, at least one reachable
   fingertip). A design that fails can never pass the search, because one
   finger gives at most one tip contact.
3. Survivors are searched in batches (one Kit launch per batch, all designs
   in parallel) until the target is filled or ``max_batches`` run out. A
   design whose sha256 was searched before is not searched again (``known``,
   kept in state.json). Designs without a stable grasp get fitness 0 and do
   not enter the archive.

The search is a callable, so this module stays CPU-only and testable; the
driver passes ``driver.run_grasp_search``.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hand_sampler.grammar.derive import derivation_from_dict, derive
from hand_sampler.grammar.distributions import Distribution

from ..scene import grammar_envelope as ge
from ..scene import population_file as pf
from . import archive as arch

__all__ = ["prefilter_report", "Proposal", "FillStats", "FillResult", "fill_viable", "build_viable_generation",
           "VIABLE_CSV_COLUMNS"]

VIABLE_CSV_COLUMNS = [
    "n_designs_trained", "short_by", "offspring_viability_rate", "immigrant_viability_rate",
    "founder_viability_rate", "viability_rate", "candidates_drawn", "candidates_searched", "prefilter_rejected",
    "grasp_search_s", "search_batches", "n_non_viable", "n_unused_viable", "probes_dropped",
]
ROLES = ("founder", "offspring", "immigrant", "probe")


def prefilter_report(report: dict, min_tip_contacts: int = 2) -> Tuple[bool, str]:
    """``(ok, reason)`` from a ``viability_report`` dict."""
    if not report.get("admitted"):
        return False, "not_admitted"
    if (report.get("digit_count") or 0) < min_tip_contacts:
        return False, f"digits<{min_tip_contacts}"
    if (report.get("fingertips_reachable") or 0) < 1:
        return False, "reach<1"
    return True, ""


@dataclass
class Proposal:
    entry: Optional["pf.PopulationEntry"]
    meta: object  # driver.DesignMeta
    sha256: str
    prefilter_ok: bool = True
    prefilter_reason: str = ""


@dataclass
class FillStats:
    drawn_by_role: Counter = field(default_factory=Counter)
    prefilter_rejected_by_role: Counter = field(default_factory=Counter)
    searched_by_role: Counter = field(default_factory=Counter)
    viable_by_role: Counter = field(default_factory=Counter)
    known_by_role: Counter = field(default_factory=Counter)
    batches: int = 0
    searched: int = 0
    search_s: float = 0.0
    non_viable: List[dict] = field(default_factory=list)
    unused_viable: List[str] = field(default_factory=list)
    short_by: int = 0

    def viability_rate(self, role: Optional[str] = None) -> Optional[float]:
        """Viable / drawn (pre-filter rejects count as drawn and non-viable);
        None when nothing of that role was drawn."""
        roles = [role] if role else [r for r in ROLES if r != "probe"]
        drawn = sum(self.drawn_by_role[r] for r in roles)
        if drawn == 0:
            return None
        return sum(self.viable_by_role[r] for r in roles) / drawn

    def log_row(self) -> dict:
        new_roles = [r for r in ROLES if r != "probe"]
        return {
            "offspring_viability_rate": self.viability_rate("offspring"),
            "immigrant_viability_rate": self.viability_rate("immigrant"),
            "founder_viability_rate": self.viability_rate("founder"),
            "viability_rate": self.viability_rate(),
            "candidates_drawn": sum(self.drawn_by_role[r] for r in new_roles),
            "candidates_searched": self.searched,
            "prefilter_rejected": sum(self.prefilter_rejected_by_role[r] for r in new_roles),
            "grasp_search_s": round(self.search_s, 1),
            "batches": self.batches,
            "short_by": self.short_by,
            "non_viable": list(self.non_viable),
            "n_non_viable": len(self.non_viable),
            "unused_viable": list(self.unused_viable),
            "n_unused_viable": len(self.unused_viable),
            "by_role": {r: {"drawn": self.drawn_by_role[r], "prefilter_rejected": self.prefilter_rejected_by_role[r],
                            "searched": self.searched_by_role[r], "known": self.known_by_role[r],
                            "viable": self.viable_by_role[r]} for r in ROLES},
        }


@dataclass
class FillResult:
    chosen: List[Proposal]
    forced_viable: List[Proposal]
    forced_non_viable: List[Proposal]
    stats: FillStats


def _record_non_viable(stats: FillStats, p: Proposal, reason: str) -> None:
    stats.non_viable.append({"design_id": p.meta.design_id, "role": p.meta.role, "source": p.meta.source,
                             "sha256": p.sha256, "reason": reason})


def fill_viable(*, target: int, forced: Sequence[Proposal], propose: Callable[[], Proposal],
                search: Callable[[List[Proposal]], Tuple[Dict[str, int], float]], known: Dict[str, int],
                batch_size: int, max_batches: int, max_draws_per_batch: int = 0) -> FillResult:
    """Fill ``target`` viable designs: the viable ``forced`` ones (probes)
    first, then new proposals. ``search(batch) -> ({sha256: n_grasps}, s)``;
    ``known`` (sha256 -> n_grasps) is read and updated."""
    stats = FillStats()
    chosen: List[Proposal] = []
    forced_viable: List[Proposal] = []
    forced_non_viable: List[Proposal] = []
    pending_forced: List[Proposal] = []
    for p in forced:
        stats.drawn_by_role[p.meta.role] += 1
        if p.sha256 in known:
            stats.known_by_role[p.meta.role] += 1
            if known[p.sha256] > 0:
                stats.viable_by_role[p.meta.role] += 1
                forced_viable.append(p)
            else:
                forced_non_viable.append(p)
                _record_non_viable(stats, p, "known_non_viable")
        else:
            pending_forced.append(p)

    def need() -> int:
        return target - len(forced_viable) - len(chosen)

    max_draws = max_draws_per_batch or 50 * max(batch_size, 1)
    while (need() > 0 or pending_forced) and stats.batches < max_batches:
        batch: List[Proposal] = list(pending_forced)
        pending_forced = []
        draws = 0
        n_new = 0
        while n_new < batch_size and draws < max_draws and need() > 0:
            p = propose()
            draws += 1
            stats.drawn_by_role[p.meta.role] += 1
            if p.sha256 in known:
                stats.known_by_role[p.meta.role] += 1
                if known[p.sha256] > 0:
                    stats.viable_by_role[p.meta.role] += 1
                    chosen.append(p)
                else:
                    _record_non_viable(stats, p, "known_non_viable")
                continue
            if not p.prefilter_ok:
                stats.prefilter_rejected_by_role[p.meta.role] += 1
                _record_non_viable(stats, p, f"prefilter:{p.prefilter_reason}")
                continue
            batch.append(p)
            n_new += 1
        if not batch:
            break
        counts, secs = search(batch)
        stats.batches += 1
        stats.search_s += float(secs)
        stats.searched += len(batch)
        for p in batch:
            n = int(counts.get(p.sha256, 0))
            known[p.sha256] = n
            stats.searched_by_role[p.meta.role] += 1
            if n > 0:
                stats.viable_by_role[p.meta.role] += 1
            if p.meta.role == "probe":
                (forced_viable if n > 0 else forced_non_viable).append(p)
                if n == 0:
                    _record_non_viable(stats, p, "no_stable_grasp")
            elif n > 0:
                if need() > 0:
                    chosen.append(p)
                else:
                    stats.unused_viable.append(p.meta.design_id)
            else:
                _record_non_viable(stats, p, "no_stable_grasp")
    stats.short_by = max(0, need())
    return FillResult(chosen=chosen, forced_viable=forced_viable, forced_non_viable=forced_non_viable,
                      stats=stats)


def _proposal_from(entry: "pf.PopulationEntry", meta, model, min_tip_contacts: int, check: bool) -> Proposal:
    ok, why = (True, "")
    if check:
        ok, why = prefilter_report(ge.viability_report(model), min_tip_contacts)
    return Proposal(entry=entry, meta=meta, sha256=entry.sha256, prefilter_ok=ok, prefilter_reason=why)


def build_viable_generation(
    generation: int, n_designs: int, probe_hand_ids: Sequence[str], dist,
    archive: "arch.Archive", driver_rng: np.random.Generator, minter, *, known: Dict[str, int],
    search: Callable[[List[Proposal]], Tuple[Dict[str, int], float]], batch_size: int, max_batches: int,
    max_offspring_retries: int = 16, min_tip_contacts: int = 2,
):
    """The viable-only counterpart of ``driver.build_generation_population``:
    ``(GenerationPlan, report)`` with elites, viable probes and newly drawn
    viable designs; ``report`` is the generation's ``viability`` log entry.
    ``dist`` is a Distribution or a ``{variant name: Distribution}`` dict;
    founders and immigrants cycle through the dict (a mixed population),
    mutation uses its first entry."""
    from . import driver as drv

    named = dict(dist) if isinstance(dist, dict) else {"": dist}
    variant_names, dists = list(named), list(named.values())
    drawn_variants: List[str] = []

    t0 = time.time()
    probes: List[Proposal] = []
    for hand_id in probe_hand_ids:
        entry, status, reason = pf.projected_entry(hand_id)
        if status != "admitted":
            raise RuntimeError(f"probe hand {hand_id!r} is not admitted ({status}): {reason}")
        meta = drv.DesignMeta(design_id=f"probe:{hand_id}", founder_id=f"probe:{hand_id}", parent_id=None,
                              generation_born=0, digit_count=-1, joint_count=-1, role="probe", source=entry.source,
                              hand_id=hand_id)
        probes.append(Proposal(entry=entry, meta=meta, sha256=entry.sha256))

    elites = archive.elites()
    slots = n_designs - len(probe_hand_ids)
    if slots <= 0:
        raise ValueError(f"--designs {n_designs} must exceed the number of probes ({len(probe_hand_ids)})")
    if len(elites) > slots:
        elites = sorted(elites, key=lambda e: e.fitness, reverse=True)[:slots]
    elite_entries, elite_metas = [], []
    for e in elites:
        derivation = derivation_from_dict(e.derivation_dict)
        source = f"arch:{e.design_id}"
        entry = pf.make_entry(source, derivation, derive(derivation))
        elite_entries.append(entry)
        elite_metas.append(drv.DesignMeta(
            design_id=e.design_id, founder_id=e.founder_id, parent_id=e.parent_id,
            generation_born=e.generation_born, digit_count=e.digit_count, joint_count=e.joint_count,
            role="elite", source=source))
        known.setdefault(entry.sha256, 1)  # an elite trained, so it had grasps

    def founder():
        k = len(drawn_variants) % len(dists)
        drawn_variants.append(variant_names[k])
        return drv.sample_new_founder(dists[k], driver_rng)

    def propose() -> Proposal:
        if generation == 0 or not elites:
            derivation, design = founder()
            role, design_id = "founder", minter.mint(generation, "founder")
            founder_id, parent_id = design_id, None
        else:
            parent = archive.sample_parent(driver_rng)
            result = drv.try_offspring(derivation_from_dict(parent.derivation_dict), dists[0], driver_rng,
                                       max_retries=max_offspring_retries)
            if result is not None:
                derivation, design = result
                role, design_id = "offspring", minter.mint(generation, "off")
                founder_id, parent_id = parent.founder_id, parent.design_id
            else:
                derivation, design = founder()
                role, design_id = "immigrant", minter.mint(generation, "imm")
                founder_id, parent_id = design_id, None
        digit_count, joint_count = drv._design_counts(design)
        source = f"arch:{design_id}"
        model = derive(derivation)
        entry = pf.make_entry(source, derivation, model)
        meta = drv.DesignMeta(design_id=design_id, founder_id=founder_id, parent_id=parent_id,
                              generation_born=generation, digit_count=digit_count, joint_count=joint_count,
                              role=role, source=source)
        return _proposal_from(entry, meta, model, min_tip_contacts, check=True)

    target = n_designs - len(elite_entries)
    res = fill_viable(target=target, forced=probes, propose=propose, search=search, known=known,
                      batch_size=batch_size, max_batches=max_batches)
    entries = [p.entry for p in res.forced_viable] + elite_entries + [p.entry for p in res.chosen]
    metas = [p.meta for p in res.forced_viable] + elite_metas + [p.meta for p in res.chosen]
    report = res.stats.log_row()
    report.update({
        "n_designs": len(entries), "n_elites": len(elite_entries), "n_new": len(res.chosen),
        "probes_viable": [p.meta.design_id for p in res.forced_viable],
        "probes_dropped": [p.meta.design_id for p in res.forced_non_viable],
        "assembly_s": round(time.time() - t0, 1),
        "founder_variants": list(drawn_variants),
    })
    return drv.GenerationPlan(entries=entries, metas=metas), report


def main(argv=None) -> int:
    """Build one viable population (no training): founders round-robin over
    ``--variants``, plus probes, grasp-searched in batches until
    ``--designs`` are viable. Writes ``<out-dir>/population.json``, the
    grasp cache ``<out-dir>/grasp_cache.npz`` and ``viability.json``."""
    import argparse
    import json
    from pathlib import Path

    from . import driver as drv

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--variants", required=True, help="comma-separated, e.g. G_V3S,G_V1")
    ap.add_argument("--probes", default="allegro_right,sharpa_left_on_iiwa14,leap_right")
    ap.add_argument("--designs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task-profile", default="hora", choices=("hora", "anyrotate"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--search-envs", type=int, default=4096)
    ap.add_argument("--max-batches", type=int, default=6)
    ap.add_argument("--search-rounds", type=int, default=40)
    ap.add_argument("--grasps-per-design", type=int, default=64)
    ap.add_argument("--search-timeout-s", type=int, default=1500)
    ap.add_argument("--max-joint-speed", type=float, default=5.0, help="grasp_max_joint_speed (rad/s; -1 off)")
    ap.add_argument("--train-python", default=str(drv.REPO_ROOT / ".venv_isaacsim" / "bin" / "python3"))
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        train_python=a.train_python, task_profile=a.task_profile, viable_search_envs=a.search_envs,
        viable_search_timeout_s=a.search_timeout_s, viable_grasps_per_design=a.grasps_per_design,
        viable_search_rounds=a.search_rounds, train_override=[], grasp_cache_path=str(out / "grasp_cache.npz"),
        viable_max_joint_speed=a.max_joint_speed)
    counter = {"i": 0}

    def search(batch):
        counter["i"] += 1
        return drv.run_grasp_search(batch, args=args, run_dir=out, gen_dir=out, index=counter["i"])

    dists = {v: drv.resolve_variant(v) for v in a.variants.split(",")}
    plan, report = build_viable_generation(
        0, a.designs, [h for h in a.probes.split(",") if h], dists, arch.Archive(), np.random.default_rng(a.seed),
        drv.IdMinter(), known={}, search=search, batch_size=a.batch_size, max_batches=a.max_batches)
    doc = pf.write_population(out / "population.json", plan.entries)
    report["population_sha256"] = doc["population_sha256"]
    report["designs"] = [{"source": m.source, "role": m.role, "digit_count": m.digit_count,
                          "joint_count": m.joint_count} for m in plan.metas]
    (out / "viability.json").write_text(json.dumps(report, indent=1))
    print(f"[viable] {len(plan.entries)} viable design(s) -> {out / 'population.json'}; searched "
          f"{report['candidates_searched']} in {report['grasp_search_s']:.0f} s over {report['batches']} launch(es); "
          f"founder viability {report['founder_viability_rate']}; dropped probes {report['probes_dropped']}",
          flush=True)
    return 0 if report["short_by"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
