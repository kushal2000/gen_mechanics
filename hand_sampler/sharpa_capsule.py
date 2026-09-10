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
down -- it is also the list of things the grammar cannot express:

  palm thickness 25 mm, not 50    PALM_THICKNESS_RANGE caps at 40 and the sampler
                                  only seeds 20-25; thickness is never mutated
  palm width 100 mm, not 85       four mounts on one face at MIN_SAME_FACE_SEPARATION
                                  (25 mm) plus two MOUNT_EDGE_MARGINs need 95
  finger spacing 25 mm, not ~20   MIN_SAME_FACE_SEPARATION
  MCP as two joints 15 mm apart   MIN_LINK_LENGTH; SHARPA's FE and AA are coincident
  no pinky CMC (21 joints, not 22) it is a roll about the finger (phi = 0), and the
                                  sampler fixes phi = 90 while mutation moves only theta
  thumb on the -y face            SHARPA mounts it on the palm surface, 24 mm inboard;
                                  the grammar has only the three thin faces
  thumb MCP hinges axis-aligned   the real ones are oblique; theta on the 15-degree
                                  grid cannot tilt a hinge out of the y-z plane
  lengths on the 5 mm grid        LINK_QUANTUM

What survives: five fingers, the flexion/abduction pattern of every joint, the
proximal/middle/distal proportions, the thumb's opposition via a -60 degree rest
offset on its abduction joint, and fingertips within 4-17 mm of SHARPA's on the
four fingers and ~33 mm on the thumb.
"""

from __future__ import annotations

import math

from hand_sampler import design_space as ds

NAME = "sharpa_capsule"

FLEXION, ABDUCTION = 0.0, 90.0


def _joint(theta_deg: float, offset_deg: float = 0.0) -> ds.Joint:
    return ds.Joint(theta=math.radians(theta_deg), phi=math.pi / 2,
                    offset=math.radians(offset_deg))


def _finger_on_edge(v: float) -> ds.Finger:
    """Index / middle / ring / pinky: MCP flexion, MCP abduction, PIP, DIP, pad.

    SHARPA's MCP_FE and MCP_AA share one origin; the grammar wants 15 mm between
    joints, so the abduction joint sits one MIN_LINK_LENGTH down the finger and
    the proximal phalanx is shortened to keep PIP where it was (15 + 30 = 45 mm
    against 47). 31.5 -> 30 and the 20 mm pad are on the grid already.
    """
    return ds.Finger(
        mount=ds.Mount("+z", 0.5, v),
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
        mount=ds.Mount("-y", 0.6, 0.25),
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
    palm = ds.Palm(thickness=0.025, width=0.100, length=0.085)
    # v runs across the width; 0.125 .. 0.875 is -37.5 .. +37.5 mm, 25 mm apart.
    return ds.Hand(palm=palm, fingers=(
        _thumb(),
        _finger_on_edge(0.125),
        _finger_on_edge(0.375),
        _finger_on_edge(0.625),
        _finger_on_edge(0.875),
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
          f"palm {tuple(round(v * 1000) for v in hand.palm.extents)} mm -> {out}")


if __name__ == "__main__":
    main()
