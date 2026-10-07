"""SHARPA, as nearly as the grammar can say it.

The one hand with a known-good training curve is SHARPA, and it reaches the
simulator through a URDF -- never through `author_hand`. So nothing has ever
tested the authored path against a reference. This is that reference: SHARPA's
structure, rebuilt from the tokens the sampler and mutator actually produce, so
it can be trained through the same code as every generated design. If it learns
like SHARPA, the authoring is right; if it does not, either the authoring is
off or one of the deviations below is what mattered -- and the list is short
enough to bisect.

Measured in the generated palm frame (x = grasp normal, z = along the palm,
y = across) from the SHARPA URDF at the arm's flange: four fingers on the far
edge, each MCP a coincident flexion+abduction pair, then PIP and DIP at 47 and
31.5 mm, a 20 mm pad; a thumb on the palm surface near the wrist with a 66 mm
proximal; a pinky with an extra roll joint at its base. 22 joints.

Every deviation is forced by a specific rule, which is the point of writing it
down -- it is also the list of things the grammar cannot express.

Its 25 mm palm used to be on this list, against a thickness that was then 50 and
later 30. PALM_THICKNESS is 25 now, a quantum under the capsule diameter, so the
capsule and the grammar agree on it exactly and it is no longer a deviation:

  palm width 100 mm, not 85       four mounts in a row at MIN_MOUNT_SEPARATION
                                  span 105. A palm has no width of its own now
                                  -- it is the hull of where the fingers are --
                                  so this is not a deviation any more either,
                                  only a note on how wide the row has to be
  finger spacing 25 mm, not ~20   MIN_MOUNT_SEPARATION, which is 35 mm now
  MCP as two joints 15 mm apart   MIN_LINK_LENGTH; SHARPA's FE and AA are coincident
  no pinky CMC (21 joints, not 22) a roll about the finger. The grammar HAS a roll
                                  kind now, so this one is no longer forced -- it is
                                  simply what was built before roll existed
  thumb 2.5 mm off the midplane   it was mounted at u = 0.6 on a box face; a
       is lost                     mount is a (y, z) offset in the palm's own
                                  plane now, with no freedom across it, so
                                  every finger starts on that plane
  thumb MCP hinges axis-aligned   the real ones are oblique, and a joint is one of
                                  three kinds, so there is no oblique axis to reach
  lengths on the 5 mm grid        LINK_QUANTUM

This hand also predates two constants it would now break: links of 15 mm against
a 20 mm floor, and mounts off the grid. test_it_is_a_legal_design records that,
and is expected to FAIL.

It also carries assembly offsets -- the thumb's -60 degree abduction rest -- which
a generated hand no longer has at all. They survive here because Joint.offset is
kept for hands that were measured rather than drawn.

What survives: five fingers, the flexion/abduction pattern of every joint, the
proximal/middle/distal proportions, the thumb's opposition via that -60 degree
rest offset, and fingertips within 4-17 mm of SHARPA's on the four fingers and
~33 mm on the thumb.
"""

from __future__ import annotations

import math

from hand_sampler import design_space as ds

NAME = "sharpa_capsule"

FLEXION, ABDUCTION = ds.FLEXION, ds.ABDUCTION


def _joint(kind: int, offset_deg: float = 0.0) -> ds.Joint:
    return ds.Joint(kind=kind,
                    offset=math.radians(offset_deg))


def _polar_mount(radius: float, bearing_deg: float, facing: float) -> "ds.Mount":
    """A mount from the polar pair this hand was MEASURED in.

    A mount is cartesian now, but every number here came off SHARPA as a radius
    and a bearing, and assets/populations/sharpa_capsule.json came from this
    function -- so the measurement is kept in the units it was taken in and
    converted here, rather than rewritten as y and z that match nothing on the
    drawing. Deliberately NOT snapped to PALM_QUANTUM: this hand is already off
    the grammar's grid in several ways (see the xfail in its test) and snapping
    would move its bases as well, for no gain.
    """
    b = math.radians(bearing_deg) % (2.0 * math.pi)
    return ds.Mount.polar(radius, b, facing, snap=False)


def _finger_on_edge(radius: float, bearing_deg: float) -> ds.Finger:
    """Index / middle / ring / pinky: MCP flexion, MCP abduction, PIP, DIP, pad.

    SHARPA's MCP_FE and MCP_AA share one origin; the grammar wants 15 mm between
    joints, so the abduction joint sits one MIN_LINK_LENGTH down the finger and
    the proximal phalanx is shortened to keep PIP where it was (15 + 30 = 45 mm
    against 47). 31.5 -> 30 and the 20 mm pad are on the grid already.
    """
    return ds.Finger(
        mount=_polar_mount(radius, bearing_deg, 0.0),
        segments=(
            ds.Segment(_joint(FLEXION), 0.015),
            ds.Segment(_joint(ABDUCTION), 0.030),
            ds.Segment(_joint(FLEXION), 0.030),
            ds.Segment(_joint(FLEXION), 0.020),
        ),
    )


def _thumb() -> ds.Finger:
    """CMC flexion, CMC abduction, MCP flexion, MCP abduction, IP, pad.

    SHARPA: CMC_FE, CMC_AA 5 mm later, a 66 mm proximal to a coincident MCP
    pair, 39 mm to the IP, 20 mm pad -- 130 mm. Here 15 + 50 + 15 + 25 + 20 =
    125, the two coincident pairs each opened to MIN_LINK_LENGTH and the links
    after them shortened to compensate. The -60 degree rest offset on CMC
    abduction points it out of the -y face and up the palm the way SHARPA's
    thumb opposes the fingers; the mount is at the wrist end of the face
    (v = 0.25, i.e. 21 mm up an 85 mm palm) as SHARPA's is.
    """
    return ds.Finger(
        mount=_polar_mount(0.0543, 247.0, math.radians(270.0)),
        segments=(
            ds.Segment(_joint(FLEXION), 0.015),
            ds.Segment(_joint(ABDUCTION, offset_deg=-60.0), 0.050),
            ds.Segment(_joint(FLEXION), 0.015),
            ds.Segment(_joint(ABDUCTION), 0.025),
            ds.Segment(_joint(FLEXION), 0.020),
        ),
    )


def sharpa_capsule() -> ds.Hand:
    """The hand. Thumb first, then index to pinky across the far edge."""
    palm = ds.Palm(thickness=0.025)
    # Converted from the box this was drawn on -- a 100 x 85 mm palm whose
    # centre sat at z = 42.5 -- so the fingers have not moved. The four sat at
    # y = -37.5 .. +37.5 across its far edge, 25 mm apart.
    return ds.Hand(palm=palm, fingers=(
        _thumb(),
        _finger_on_edge(0.0567, 318.6),
        _finger_on_edge(0.0443, 343.6),
        _finger_on_edge(0.0443, 16.4),
        _finger_on_edge(0.0567, 41.4),
    ))


# SHARPA's fingertips in the same frame, for the tests and for anyone checking
# how far the capsule version sits from the real thing. Millimetres.
SHARPA_TIPS_MM = {
    "thumb": (15.4, -103.8, 90.5),
    "index": (1.0, -30.3, 174.2),
    "middle": (0.0, -10.0, 177.2),
    "ring": (1.5, 10.3, 171.2),
    "pinky": (3.0, 31.1, 165.2),
}


def main() -> None:
    import argparse

    from hand_sampler import population_io, validate_design

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help=f"where to write it (default: assets/populations/{NAME}.json)")
    args = ap.parse_args()

    hand = sharpa_capsule()
    problems = validate_design.check(hand)
    if problems:
        raise SystemExit("not a legal design:\n  " + "\n  ".join(problems))
    out = population_io.save_population(
        [hand], args.out or population_io.default_path(NAME), name=NAME,
        provenance=population_io.provenance(method="sharpa_capsule", source="hand_sampler.sharpa_capsule"))
    print(f"{NAME}: {hand.n_fingers} fingers, {hand.n_joints} joints, "
          f"palm {tuple(round(v * 1000) for v in ds.palm_extents(hand))} mm -> {out}")


if __name__ == "__main__":
    main()
