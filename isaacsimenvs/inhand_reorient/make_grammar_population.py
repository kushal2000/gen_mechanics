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


def _has_real_carrier(model) -> bool:
    """Whether `model` (already admitted) uses a REAL jointed palm carrier
    (finger slot 3 and/or 4 is a real joint, not a ghost) -- the "carrier
    design" selection criterion for the 16-design test population."""
    design = ge.canonicalize(model, source="_probe")
    return bool(design.slot_valid[ge.PC0_SLOT] or design.slot_valid[ge.PC1_SLOT])


def collect_serial_entries(n: int, seed0: int = 0, max_seeds: int = 5000) -> List[pf.PopulationEntry]:
    out: List[pf.PopulationEntry] = []
    dist = pf._variant_distribution("G_SERIAL")
    for seed in range(seed0, seed0 + max_seeds):
        derivation = sample_derivation(seed, dist)
        model = derive(derivation)
        # `admit`'s default args already enforce the rest-overlap AND
        # spawn-height gates (review items 2/4) -- this CLI used to run its
        # own separate, easy-to-bypass `_no_bad_rest_overlap` check on top
        # of a structural-only `admit(model)`; that duplication (and the
        # `--variant sampled_only` path that skipped it entirely) is exactly
        # the review's "not enforced ... sampled_only skips it" gap.
        result = ge.admit(model)
        if result.ok:
            out.append(pf.make_entry(f"sampled:G_SERIAL:{seed}", derivation, model))
            if len(out) >= n:
                break
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)}/{n} admitted G_SERIAL designs in {max_seeds} seeds")
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
        if _has_real_carrier(model):
            out.append(pf.make_entry(f"sampled:DEFAULT_CARRIER:{seed}", derivation, model))
            if len(out) >= n:
                break
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)}/{n} admitted carrier designs in {max_seeds} seeds")
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
