"""Measure the SHARPA hand, so the generated design space is anchored not guessed.

The procedural generator (docs/proposal_codesign.md, plan G0-G5) builds hands from
capsules rather than meshes. Capsules need a radius and a density, and both have
to come from somewhere. Picking them by eye would make every generated hand's mass
an invention, and mass is not a free parameter -- it sets how hard the fingers hit
the object and how much the arm has to carry.

So: read the radius off SHARPA's own collision meshes, then solve each tier's
density so that a capsule at SHARPA's nominal length and radius weighs what SHARPA
weighs. A generated SHARPA-like hand then reproduces SHARPA's masses by
construction, which ``tests/test_generated_hand.py --check kinematics`` asserts.

Two things the measurement turned up that the generator has to handle:

* SHARPA's virtual links (the zero-length bodies between coincident FE/AA joints)
  already carry ``mass = 1e-6 kg``. That is exactly the ghosting convention, so
  ghosted joints are not a foreign device bolted onto this robot -- they are how
  the reference hand already represents a jointless body.
* The pinky metacarpal is 9.6 mm long with a 17.3 mm radius. As a capsule that is
  a sphere with a sliver in the middle: volume is dominated by the end caps and
  barely responds to length, so the fitted density blows up to 4108 kg/m^3 against
  the thumb metacarpal's 1073. The metacarpal tier therefore takes its density
  from the thumb (the well-conditioned sample), and the generator emits no
  geometry at all below MC_MIN_LENGTH.

    .venv_isaacsim/bin/python -m genmech.tools.measure_sharpa_anchors
    .venv_isaacsim/bin/python -m genmech.tools.measure_sharpa_anchors --verify

``--verify`` re-measures and compares against the committed constants in
``genmech/robots/generated/sharpa_anchors.py``, so an asset change cannot silently
invalidate the design space.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from genmech.utils.paths import resolve as resolve_repo_path


SHARPA_URDF = (
    "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
)

# Nominal segment lengths are the *joint spacings* measured from the URDF, not the
# mesh extents. The mesh overhangs its joint (MP's mesh is 36.5 mm long across a
# 31.5 mm joint spacing), but kinematics are what the joint spacing says, so the
# capsule length must match the joint spacing or the generated hand would reach
# further than the numbers claim.
NOMINAL: dict[str, tuple[str, float, tuple[str, ...]]] = {
    # tier: (collision mesh, nominal length [m], links whose mass it must carry)
    "mc": ("left_sharpa_meshes/left_thumb_MC.STL", 0.0661, ("left_thumb_MC",)),
    "pp": ("left_sharpa_meshes/left_PP.STL", 0.0470, ("left_index_PP",)),
    "mp": ("left_sharpa_meshes/left_MP.STL", 0.0315, ("left_index_MP",)),
    # merge_fixed_joints folds the elastomer pad and the fingertip frame into DP,
    # so the capsule has to weigh all three or the fingertip comes out 27% light.
    "dp": (
        "left_sharpa_meshes/left_DP.STL",
        0.0260,
        ("left_index_DP", "left_index_elastomer", "left_index_fingertip"),
    ),
}

PALM_MESH = "left_sharpa_meshes/left_hand_C_MC.STL"
PALM_LINK = "left_hand_C_MC"


def _link_mass(root: ET.Element, name: str) -> float:
    for link in root.findall("link"):
        if link.get("name") != name:
            continue
        inertial = link.find("inertial")
        if inertial is None:
            return 0.0
        return float(inertial.find("mass").get("value"))
    raise KeyError(f"link {name!r} not in URDF")


def cylinder_part(total_length: float, radius: float) -> float:
    """Cylindrical section of a capsule whose TOTAL length is ``total_length``.

    Isaac Lab's converter reads a URDF cylinder's ``length`` as the capsule's
    cylindrical section and adds a hemisphere of ``radius`` at each end. Emitting
    the segment length directly therefore produced a collision shape 2r longer
    than the joint spacing -- 67.3 mm for a 47.0 mm phalanx, +43%. Every capsule
    overhung both its joints, which inflated self-collision by construction and
    made fingers physically longer than their kinematics claimed.
    """
    return max(total_length - 2.0 * radius, 0.0)


def capsule_volume(total_length: float, radius: float) -> float:
    """Volume of a capsule of the given TOTAL length."""
    h = cylinder_part(total_length, radius)
    return math.pi * radius * radius * h + (4.0 / 3.0) * math.pi * radius ** 3


def measure() -> dict:
    import trimesh

    urdf = resolve_repo_path(SHARPA_URDF)
    root = ET.parse(urdf).getroot()
    asset_dir = urdf.parent

    out: dict = {"tiers": {}}
    for tier, (mesh_rel, nominal_len, mass_links) in NOMINAL.items():
        mesh = trimesh.load(asset_dir / mesh_rel, force="mesh")
        extents = mesh.extents
        # Radius from the two SHORT axes: the long axis is the segment direction,
        # so it describes length, not thickness. Half of their mean is the capsule
        # radius that best fills the cross-section.
        short = sorted(extents)[:2]
        radius = float(sum(short) / 4.0)
        mass = sum(_link_mass(root, n) for n in mass_links)
        density = mass / capsule_volume(nominal_len, radius)
        out["tiers"][tier] = {
            "nominal_length_m": nominal_len,
            "radius_m": radius,
            "mass_kg": mass,
            "density_kg_m3": density,
            "mesh_extents_m": tuple(float(e) for e in extents),
        }

    palm_mesh = trimesh.load(asset_dir / PALM_MESH, force="mesh")
    palm_extents = tuple(float(e) for e in palm_mesh.extents)
    palm_mass = _link_mass(root, PALM_LINK)
    palm_box_volume = palm_extents[0] * palm_extents[1] * palm_extents[2]
    out["palm"] = {
        "extents_m": palm_extents,
        "mass_kg": palm_mass,
        "density_kg_m3": palm_mass / palm_box_volume,
        "mesh_fill_fraction": float(palm_mesh.volume) / palm_box_volume,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="compare against the committed constants")
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    m = measure()

    print("=== capsule tiers, calibrated against SHARPA ===")
    for tier, d in m["tiers"].items():
        print(f"  {tier:4s} L={d['nominal_length_m'] * 1000:6.2f}mm "
              f"r={d['radius_m'] * 1000:6.2f}mm "
              f"m={d['mass_kg'] * 1000:7.2f}g "
              f"rho={d['density_kg_m3']:7.1f} kg/m^3")
    p = m["palm"]
    print(f"  palm box=({p['extents_m'][0] * 1000:.1f},{p['extents_m'][1] * 1000:.1f},"
          f"{p['extents_m'][2] * 1000:.1f})mm m={p['mass_kg'] * 1000:.2f}g "
          f"rho={p['density_kg_m3']:.1f} kg/m^3 (mesh fills "
          f"{p['mesh_fill_fraction'] * 100:.0f}% of the box)")

    if not args.verify:
        return

    from genmech.robots.generated import sharpa_anchors as A

    bad = []

    def check(label: str, got: float, want: float) -> None:
        if want == 0.0:
            ok = got == 0.0
        else:
            ok = abs(got - want) <= args.rtol * abs(want)
        print(f"  {'ok ' if ok else 'BAD'} {label:28s} measured={got:12.6g} "
              f"committed={want:12.6g}")
        if not ok:
            bad.append(label)

    print("\n=== verify against genmech/robots/generated/sharpa_anchors.py ===")
    for tier, d in m["tiers"].items():
        check(f"{tier}.radius_m", d["radius_m"], A.TIER_RADIUS_M[tier])
        check(f"{tier}.density", d["density_kg_m3"], A.TIER_DENSITY_KG_M3[tier])
        check(f"{tier}.nominal_length_m", d["nominal_length_m"],
              A.TIER_NOMINAL_LENGTH_M[tier])
    check("palm.density", m["palm"]["density_kg_m3"], A.PALM_DENSITY_KG_M3)
    for i, e in enumerate(m["palm"]["extents_m"]):
        check(f"palm.extent[{i}]", e, A.PALM_EXTENTS_M[i])

    if bad:
        raise SystemExit(f"\nANCHORS DRIFTED: {bad}")
    print("\nANCHORS VERIFIED")


if __name__ == "__main__":
    main()
