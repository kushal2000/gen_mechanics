"""CLI: build a grammar population file for `env.assets.hand_population`.

Default invocation (no args) writes the 16-design Kit-smoke test population
the Phase 2 design note asks for: 8 G_SERIAL, 4 "carrier" designs (DEFAULT
variant, admitted WITH a real jointed palm carrier -- i.e. finger slot 3
and/or 4 is real, not a ghost), and the 4 projected commercial hands
allegro_right, dclaw, sharpa_left_on_iiwa14, wuji_right.

Numpy + `hand_sampler` only (no `isaaclab`/`pxr`): runs under plain
`python3`, same as `scene/grammar_envelope.py`/`scene/population_file.py` --
except that importing `isaacsimenvs` at all eagerly imports `gymnasium`
(`isaacsimenvs/__init__.py`), so use the isaacsim venv:

    .venv_isaacsim/bin/python3 -m isaacsimenvs.inhand_reorient.make_grammar_population \
        --out outputs/grammar_populations/test16.json
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Sequence

from hand_sampler.grammar.derive import derive, sample_derivation

from .scene import grammar_envelope as ge
from .scene import population_file as pf

DEFAULT_COMMERCIAL_HANDS: tuple[str, ...] = ("allegro_right", "dclaw", "sharpa_left_on_iiwa14", "wuji_right")


MAX_REST_PENETRATION_M = 0.003
"""Reject a design whose `rest_overlap_pairs` (capsule interpenetration at
q=0) exceeds this. Design note risk 4 ("rest self-penetration: numpy
overlap filter plus the zero-action smoke") flagged this as a risk but this
CLI did not actually filter on it until a Kit smoke traced a violent,
gravity-independent, velocity-capped joint blowup (multiple joints
saturating +/-10 rad/s within one physics step, on every seed tried, always
worst on the pc0/pc1 ghost columns because their limits are tightest) to
exactly this: G_SERIAL seed 7's f0/f1/f2/f4 (all root-mounted, no palm
spread to separate them) overlap by 5.6-9.6 mm at rest -- PhysX's
depenetration impulse at the first step, not a bug in the joint frames or
the authoring (confirmed separately: FK-in-Kit-vs-analytic matches to
~5e-7 m with a pure state readback, no stepping)."""


def _has_real_carrier(model) -> bool:
    """Whether `model` (already admitted) uses a REAL jointed palm carrier
    (finger slot 3 and/or 4 is a real joint, not a ghost) -- the "carrier
    design" selection criterion for the 16-design test population."""
    design = ge.canonicalize(model, source="_probe")
    return bool(design.slot_valid[ge.PC0_SLOT] or design.slot_valid[ge.PC1_SLOT])


def _no_bad_rest_overlap(model, max_penetration_m: float = MAX_REST_PENETRATION_M) -> bool:
    design = ge.canonicalize(model, source="_probe")
    pairs = ge.rest_overlap_pairs(design)
    return all(pen <= max_penetration_m for _a, _b, pen in pairs)


def collect_serial_entries(n: int, seed0: int = 0, max_seeds: int = 5000) -> List[pf.PopulationEntry]:
    out: List[pf.PopulationEntry] = []
    dist = pf._variant_distribution("G_SERIAL")
    for seed in range(seed0, seed0 + max_seeds):
        derivation = sample_derivation(seed, dist)
        model = derive(derivation)
        result = ge.admit(model)
        if result.ok and _no_bad_rest_overlap(model):
            out.append(pf.make_entry(f"sampled:G_SERIAL:{seed}", derivation))
            if len(out) >= n:
                break
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)}/{n} admitted, non-overlapping G_SERIAL designs in {max_seeds} seeds")
    return out


def collect_carrier_entries(n: int, seed0: int = 0, max_seeds: int = 20000) -> List[pf.PopulationEntry]:
    out: List[pf.PopulationEntry] = []
    dist = pf._variant_distribution("DEFAULT")
    for seed in range(seed0, seed0 + max_seeds):
        derivation = sample_derivation(seed, dist)
        model = derive(derivation)
        result = ge.admit(model)
        if not result.ok:
            continue
        if _has_real_carrier(model) and _no_bad_rest_overlap(model):
            out.append(pf.make_entry(f"sampled:DEFAULT_CARRIER:{seed}", derivation))
            if len(out) >= n:
                break
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)}/{n} admitted, non-overlapping carrier designs in {max_seeds} seeds")
    return out


def collect_projected_entries(hand_ids: Sequence[str]) -> List[pf.PopulationEntry]:
    out: List[pf.PopulationEntry] = []
    for hand_id in hand_ids:
        entry, status, reason = pf.projected_entry(hand_id)
        if status != "admitted":
            raise RuntimeError(f"projected hand {hand_id!r} is not admitted: {status} ({reason})")
        out.append(entry)
    return out


def build_test16(seed0: int = 0, commercial_hands: Sequence[str] = DEFAULT_COMMERCIAL_HANDS) -> List[pf.PopulationEntry]:
    entries: List[pf.PopulationEntry] = []
    entries += collect_serial_entries(8, seed0=seed0)
    entries += collect_carrier_entries(4, seed0=seed0)
    entries += collect_projected_entries(commercial_hands)
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/grammar_populations/test16.json",
                     help="output population JSON path (outputs/ is git-ignored)")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--commercial", nargs="*", default=list(DEFAULT_COMMERCIAL_HANDS))
    ap.add_argument("--variant", choices=["test16", "sampled_only"], default="test16",
                     help="'test16': the design note's 16-design Kit-smoke population (default). "
                          "'sampled_only': --n-serial G_SERIAL + --n-default DEFAULT sampled designs, no projections.")
    ap.add_argument("--n-serial", type=int, default=8)
    ap.add_argument("--n-default", type=int, default=8)
    args = ap.parse_args(argv)

    if args.variant == "test16":
        entries = build_test16(seed0=args.seed0, commercial_hands=args.commercial)
    else:
        serial, serial_rej = pf.sampled_entries("G_SERIAL", range(args.seed0, args.seed0 + 5000))
        default, default_rej = pf.sampled_entries("DEFAULT", range(args.seed0, args.seed0 + 5000))
        entries = serial[: args.n_serial] + default[: args.n_default]
        print(f"G_SERIAL: {len(serial)} admitted (used {min(args.n_serial, len(serial))}), "
              f"rejections {serial_rej}", file=sys.stderr)
        print(f"DEFAULT: {len(default)} admitted (used {min(args.n_default, len(default))}), "
              f"rejections {default_rej}", file=sys.stderr)

    doc = pf.write_population(args.out, entries)
    print(f"wrote {len(entries)} designs to {args.out} "
          f"(population_sha256={doc['population_sha256'][:12]}..., git_sha={doc['git_sha'][:12]})")
    for e in entries:
        print(f"  - {e.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
