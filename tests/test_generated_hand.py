#!/usr/bin/env python3
"""Gate the procedural hand generator against the hand it is anchored on.

The generator's design space is centred on SHARPA (docs/proposal_codesign.md,
plan G0-G5), so the first thing it must do is reproduce SHARPA. If a parameter
vector measured from SHARPA's own URDF does not rebuild SHARPA, then nothing
sampled around it is trustworthy either -- the same discipline that caught the
merge_fixed_joints frame error in probe_usd_override.py.

Three checks, none of which need Isaac Sim:

  kinematics  fingertip positions of the SHARPA_LIKE hand, relative to the arm
              flange, against the real SHARPA URDF
  masses      per-tier capsule masses against SHARPA's measured link masses
  ghosting    a 3-finger hand still emits 30 hand joints, and the ghosted ones
              are locked and carry no collision geometry

    .venv_isaacsim/bin/python tests/test_generated_hand.py
    .venv_isaacsim/bin/python tests/test_generated_hand.py --check ghosting --n_fingers 3
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from genmech.robots.generated import params as P                    # noqa: E402
from genmech.robots.generated import sharpa_anchors as A            # noqa: E402
from genmech.tools.build_hand_urdf import (                         # noqa: E402
    build_urdf, joint_name, link_name, write_urdf,
)
from genmech.utils.paths import resolve as resolve_repo_path        # noqa: E402


SHARPA_URDF = (
    "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
)
FLANGE = "iiwa14_link_ee"

# SHARPA fingertip link <-> generated finger slot. Slot order is
# (thumb, index, middle, ring, pinky), matching SHARPA_LIKE.
TIP_PAIRS = (
    ("thumb", "left_thumb_fingertip", 0),
    ("index", "left_index_fingertip", 1),
    ("middle", "left_middle_fingertip", 2),
    ("ring", "left_ring_fingertip", 3),
    ("pinky", "left_pinky_fingertip", 4),
)

KINEMATICS_TOL_MM = 2.0
MASS_TOL_FRAC = 0.05

failures: list[str] = []


def _fail(msg: str) -> None:
    print(f"  FAIL {msg}")
    failures.append(msg)


def _ok(msg: str) -> None:
    print(f"  ok   {msg}")


def check_kinematics() -> None:
    """SHARPA_LIKE must land its fingertips where SHARPA does."""
    import numpy as np
    import yourdfpy

    print("\n[kinematics] SHARPA_LIKE vs the real SHARPA URDF, at zero joints")
    with tempfile.TemporaryDirectory() as td:
        gen_path = write_urdf(P.SHARPA_LIKE, Path(td) / "sharpa_like.urdf")
        gen = yourdfpy.URDF.load(str(gen_path), load_meshes=False,
                                 build_scene_graph=True)
    ref = yourdfpy.URDF.load(str(resolve_repo_path(SHARPA_URDF)),
                             load_meshes=False, build_scene_graph=True)

    for name, ref_link, slot in TIP_PAIRS:
        a = ref.get_transform(ref_link, FLANGE)[:3, 3] * 1000.0
        b = gen.get_transform(link_name(slot, "tip"), FLANGE)[:3, 3] * 1000.0
        err = float(np.linalg.norm(a - b))
        msg = (f"{name:7s} tip ({a[0]:+7.2f},{a[1]:+7.2f},{a[2]:+7.2f}) vs "
               f"({b[0]:+7.2f},{b[1]:+7.2f},{b[2]:+7.2f}) err={err:.3f} mm")
        _ok(msg) if err <= KINEMATICS_TOL_MM else _fail(msg)


def check_masses() -> None:
    """Capsule masses must match the SHARPA links the densities were fitted to.

    This is what stops the capsule model from quietly changing how heavy a hand
    is. Mass is not free: it sets how hard a finger hits the object and what the
    arm has to carry.
    """
    print("\n[masses] generated capsule masses vs SHARPA's measured links")
    root = build_urdf(P.SHARPA_LIKE)
    links = {l.get("name"): l for l in root.findall("link")}

    def mass_of(name: str) -> float:
        el = links[name].find("inertial/mass")
        return float(el.get("value"))

    # Index finger carries the tiers the densities were fitted against.
    for tier, part in (("pp", "PP"), ("mp", "MP"), ("dp", "DP")):
        got = mass_of(link_name(1, part))
        want = A.TIER_MASS_KG[tier]
        rel = abs(got - want) / want
        msg = (f"index {part:2s} {got * 1000:7.3f} g vs SHARPA "
               f"{want * 1000:7.3f} g ({rel * 100:.2f}%)")
        _ok(msg) if rel <= MASS_TOL_FRAC else _fail(msg)

    # Thumb metacarpal, the tier's own calibration sample.
    got, want = mass_of(link_name(0, "MC")), A.TIER_MASS_KG["mc"]
    rel = abs(got - want) / want
    msg = f"thumb MC {got * 1000:7.3f} g vs SHARPA {want * 1000:7.3f} g ({rel * 100:.2f}%)"
    _ok(msg) if rel <= MASS_TOL_FRAC else _fail(msg)

    got, want = mass_of("gen_palm"), A.PALM_MASS_KG
    rel = abs(got - want) / want
    msg = f"palm     {got * 1000:7.3f} g vs SHARPA {want * 1000:7.3f} g ({rel * 100:.2f}%)"
    _ok(msg) if rel <= MASS_TOL_FRAC else _fail(msg)

    # Whole-hand mass. The per-tier fits are exact on index/middle/ring -- the
    # three fingers SHARPA builds identically, and the ones the design space
    # actually varies -- but come out light on the thumb (-15.3 g) and pinky
    # (-80.7 g). Both gaps are metacarpals: the `mc` tier density is fitted to
    # the thumb's long thin metacarpal, and SHARPA's pinky metacarpal is short
    # and fat (9.6 mm long, 17.3 mm radius), whose own fitted density would be
    # 4108 kg/m^3 against the thumb's 1073. One tier cannot span 4x.
    #
    # Accepted for v1: the error is concentrated in structural links buried in
    # the palm, not in the contact surfaces, and closing it means a per-segment
    # density parameter -- more knobs for a part of the hand that never touches
    # the object. The bound below is loose enough to allow this and tight enough
    # to catch a real regression.
    total = sum(mass_of(l.get("name")) for l in root.findall("link")
                if l.get("name").startswith("gen_"))
    SHARPA_HAND_TOTAL_KG = 1.2997
    rel = abs(total - SHARPA_HAND_TOTAL_KG) / SHARPA_HAND_TOTAL_KG
    msg = (f"hand total {total * 1000:7.1f} g vs SHARPA "
           f"{SHARPA_HAND_TOTAL_KG * 1000:7.1f} g ({rel * 100:.1f}%, "
           f"metacarpal model — see comment)")
    _ok(msg) if rel <= 0.10 else _fail(msg)


def check_ghosting(n_fingers: int) -> None:
    """A hand with fewer fingers must still be the same articulation.

    This is the property that lets one Isaac Lab Articulation view hold every
    design (genmech/tools/probe_multi_articulation.py). If the joint count moves
    with finger count, the whole approach collapses back to one view per
    topology.
    """
    import random

    print(f"\n[ghosting] a {n_fingers}-finger hand is still a 30-joint hand")
    hand = P.sample_valid(random.Random(0), name="ghost_probe",
                          n_fingers=n_fingers)
    root = build_urdf(hand)

    hand_joints = [j for j in root.findall("joint")
                   if j.get("name").startswith("gen_f")
                   and j.get("type") == "revolute"]
    want = P.N_FINGER_SLOTS * P.N_JOINT_SLOTS
    msg = f"emitted {len(hand_joints)} hand joints (want {want})"
    _ok(msg) if len(hand_joints) == want else _fail(msg)

    links = {l.get("name"): l for l in root.findall("link")}
    inactive = [i for i, f in enumerate(hand.fingers) if not f.active]
    msg = f"{len(inactive)} ghosted finger slot(s): {inactive}"
    _ok(msg) if len(inactive) == P.N_FINGER_SLOTS - n_fingers else _fail(msg)

    for i in inactive:
        for slot in P.JOINT_SLOTS:
            j = next(x for x in hand_joints if x.get("name") == joint_name(i, slot))
            lim = j.find("limit")
            travel = float(lim.get("upper")) - float(lim.get("lower"))
            if travel > 1e-6:
                _fail(f"ghost joint {joint_name(i, slot)} has {travel:.3g} rad of travel")
        for part, _ in (("CMC_VL", None), ("MC", None), ("PP", None),
                        ("MP", None), ("DP", None)):
            link = links[link_name(i, part)]
            if link.findall("collision"):
                _fail(f"ghost link {link_name(i, part)} still has collision geometry")
            m = float(link.find("inertial/mass").get("value"))
            if m > A.VIRTUAL_LINK_MASS_KG * 10:
                _fail(f"ghost link {link_name(i, part)} has mass {m:.3g} kg")
    _ok("ghosted joints are locked and their links are massless and geometry-free")

    # The active fingers must be untouched by their neighbours' ghosting.
    active_geom = sum(
        len(links[link_name(i, p)].findall("collision"))
        for i, f in enumerate(hand.fingers) if f.active
        for p in ("MC", "PP", "MP", "DP")
    )
    msg = f"active fingers still carry {active_geom} collision shapes"
    _ok(msg) if active_geom > 0 else _fail(msg)


def check_population() -> None:
    """Every sampled hand must emit a well-formed, uniformly shaped robot."""
    print("\n[population] 24 sampled hands build without error")
    hands = P.sample_population(0, 24)
    shapes = set()
    for hand in hands:
        root = build_urdf(hand)
        rev = [j for j in root.findall("joint") if j.get("type") == "revolute"]
        shapes.add(len(rev))
        # Only the generated links are ours to check. The arm is copied verbatim
        # from SHARPA and contains massless frame links (iiwa14_link_ee); giving
        # them an inertial would change the arm, which is precisely what
        # docs/methodology.md §1 forbids.
        for link in root.findall("link"):
            if not link.get("name").startswith("gen_"):
                continue
            if link.find("inertial") is None:
                _fail(f"{hand.name}: link {link.get('name')} has no inertial")
    msg = f"all 24 hands have the same articulation shape: {shapes} revolute joints"
    _ok(msg) if len(shapes) == 1 and shapes == {37} else _fail(msg)

    counts = sorted({h.n_active_fingers for h in hands})
    _ok(f"finger counts spanned: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", default="all",
                        choices=("all", "kinematics", "masses", "ghosting",
                                 "population"))
    parser.add_argument("--n_fingers", type=int, default=3)
    args = parser.parse_args()

    if args.check in ("all", "kinematics"):
        check_kinematics()
    if args.check in ("all", "masses"):
        check_masses()
    if args.check in ("all", "ghosting"):
        check_ghosting(args.n_fingers)
    if args.check in ("all", "population"):
        check_population()

    print()
    if failures:
        print(f"GENERATED HAND CHECKS FAILED ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("GENERATED HAND CHECKS PASSED")


if __name__ == "__main__":
    main()
