"""CLI: build a grammar population file for `env.assets.hand_population`.

    .venv_isaacsim/bin/python3 -m isaacsimenvs.inhand_reorient.make_grammar_population \
        --out outputs/grammar_populations/test16.json [--sampled 12] [--commercial allegro_right ...]

`test16` (the default): 12 random viable hands (the Feetech rules, C1 and C2
pass) and 4 conformed commercial hands. `validation`: the mixed population
of the Isaac validation (`validation_entries`): five commercial hands
(Allegro, SVH, Wuji v2, Shadow, Inspire with its couplings), a hand with two
fingers beyond a palm cut, a hand with crossed knuckles (0 mm links), a hand
with oblique axes and random hands.

Numpy + `hand_sampler` only (no Kit), but importing `isaacsimenvs` pulls in
`gymnasium`, so run it with the isaacsim venv.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Sequence

import numpy as np

from hand_sampler.grammar import operators as gops
from hand_sampler.grammar.hand import EVOLUTION_RULES, Finger, Hand, Joint, PalmJoint, check

from .scene import grammar_envelope as ge
from .scene import population_file as pf

DEFAULT_COMMERCIAL_HANDS: tuple = ("allegro_right", "dclaw", "sharpa_left_on_iiwa14", "wuji_right")
VALIDATION_COMMERCIAL_HANDS: tuple = ("allegro_right", "svh_right", "wuji2_left", "shadow_right_local",
                                      "inspire_right")


def collect_sampled(n: int, seed0: int = 0, max_seeds: int = 5000) -> List[pf.PopulationEntry]:
    entries, rej = pf.sampled_entries(range(seed0, seed0 + max_seeds), limit=n)
    if len(entries) < n:
        raise RuntimeError(f"only {len(entries)}/{n} viable random hands in {max_seeds} seeds ({rej})")
    return entries[:n]


def collect_commercial(hand_ids: Sequence[str]) -> List[pf.PopulationEntry]:
    out = []
    for hid in hand_ids:
        entry, status, reason = pf.commercial_entry(hid)
        if status != "admitted":
            raise RuntimeError(f"commercial hand {hid!r} not admitted: {status} ({reason})")
        out.append(entry)
    return out


def _finger(y: int, z: int, joints, facing: int = 0, tilt: int = 0) -> Finger:
    return Finger(y=y, z=z, facing=facing, tilt=tilt, joints=tuple(Joint(t, a, L) for t, a, L in joints))


FLEX, ABD, ROLL = (0, 0), (90, 0), (0, 90)
BENDS = (("hinge", FLEX, 50), ("hinge", FLEX, 50), ("hinge", FLEX, 30))
THUMB_221 = (("hinge", ABD, 0), ("hinge", FLEX, 50), ("hinge", ABD, 0), ("hinge", FLEX, 50), ("hinge", FLEX, 30))


def cut_hand() -> Hand:
    """Six fingers under the Feetech rules: a row of three, two fingers
    beyond one palm cut (pointing -y) and a 2-2-1 thumb (spread-bend-spread-
    bend-bend, each spread crossing the next bend)."""
    fingers = [_finger(y, 80, BENDS) for y in (-40, 0, 40)]
    fingers += [_finger(-80, 0, BENDS, facing=270), _finger(-80, -40, BENDS, facing=270)]
    fingers.append(_finger(80, -40, THUMB_221, facing=90))
    return Hand(fingers=tuple(fingers), palm_joints=(PalmJoint(facing=330, offset=30),))


def zero_link_hand() -> Hand:
    """Knuckles with two joints at one point (0 mm links: a spread crossing a
    bend, its motor slid behind the palm), and a 2-2-1 thumb with a second
    crossing in its middle."""
    knuckle = (("hinge", ABD, 0), ("hinge", FLEX, 50), ("hinge", FLEX, 50), ("hinge", FLEX, 30))
    fingers = [_finger(y, 80, knuckle) for y in (-40, 0, 40)]
    fingers.append(_finger(80, -40, THUMB_221, facing=90))
    return Hand(fingers=tuple(fingers))


def oblique_hand() -> Hand:
    """Axes between kinds (30 degree steps), lengths at the servo floors
    their axes need; the fingers sit a grid step apart or more (an oblique
    motor's link is wider than a bending one)."""
    fingers = [gops.fit_lengths(_finger(y, z, (("hinge", (150, 30), 50), ("hinge", (0, 30), 50), ("hinge", (0, -30), 50)),
                                        facing=fa), EVOLUTION_RULES)
               for y, z, fa in ((0, 80, 0), (-80, 0, 270), (80, 0, 90))]
    fingers.append(gops.fit_lengths(_finger(0, -80, (("hinge", (120, 30), 50), ("hinge", (30, 0), 50),
                                                     ("hinge", (0, 30), 50)), facing=180, tilt=30), EVOLUTION_RULES))
    return Hand(fingers=tuple(fingers))


def validation_entries(n_random: int = 3, seed0: int = 0) -> List[pf.PopulationEntry]:
    built = []
    for name, hand in (("cut", cut_hand()), ("zero_links", zero_link_hand()), ("oblique", oblique_hand())):
        problems = check(hand, EVOLUTION_RULES)
        if problems:
            raise RuntimeError(f"{name}: {problems}")
        result = ge.admit(hand)
        if not result.ok:
            raise RuntimeError(f"{name} not admitted: {result.reasons}")
        built.append(pf.make_entry(f"{pf.SAMPLED_PREFIX}{name}", hand))
    return collect_commercial(VALIDATION_COMMERCIAL_HANDS) + built + collect_sampled(n_random, seed0)


def build_test16(seed0: int = 0, commercial_hands: Sequence[str] = DEFAULT_COMMERCIAL_HANDS) -> List[pf.PopulationEntry]:
    return collect_sampled(16 - len(commercial_hands), seed0) + collect_commercial(commercial_hands)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="outputs/grammar_populations/test16.json")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--variant", choices=["test16", "validation", "sampled_only"], default="test16")
    ap.add_argument("--commercial", nargs="*", default=list(DEFAULT_COMMERCIAL_HANDS))
    ap.add_argument("--sampled", type=int, default=12, help="random hands for --variant sampled_only")
    args = ap.parse_args(argv)
    if args.variant == "test16":
        entries = build_test16(args.seed0, args.commercial)
    elif args.variant == "validation":
        entries = validation_entries(seed0=args.seed0)
    else:
        entries, rej = pf.sampled_entries(range(args.seed0, args.seed0 + 5000), limit=args.sampled)
        print(f"rejections: {rej}", file=sys.stderr)
    doc = pf.write_population(args.out, entries)
    print(f"wrote {len(entries)} designs to {args.out} (population_sha256={doc['population_sha256'][:12]}...)")
    for e in entries:
        print(f"  - {e.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
