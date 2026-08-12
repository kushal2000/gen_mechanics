"""Does a hand collide with itself at its home pose?

Capsules are fatter than the meshes they replace, so a generated hand can
interpenetrate where the reference hand does not -- and a hand that starts every
episode already in self-contact spends the first steps being pushed apart by the
solver, which is both a physics artifact and a silent bias against whichever
design happens to overlap more.

This checks the *collision* geometry (not the visual meshes, which are finer and
irrelevant to physics), at any joint configuration, for every link pair that is
NOT already excluded:

  * directly-jointed parent/child pairs, which PhysX auto-filters;
  * pairs listed in ``spec.adjacent_links``, which the scene explicitly filters
    before baking (``scene_utils._apply_self_collision_filters``).

Anything left is a pair that PhysX will actually resolve as a contact.

Penetration is measured by sampling one link's collision vertices and taking
their signed distance into the other, both directions. That can miss an
edge-through-face overlap with no vertex inside, so it is a lower bound on
penetration -- it under-reports rather than over-reports, which is the safe
direction for a check whose job is to find problems.

    .venv_isaacsim/bin/python -m genmech.tools.check_self_collision \\
        --robot_spec sharpa_iiwa14,gen_sharpa_like

Pure kinematics: no Isaac Sim, no Kit boot.
"""

from __future__ import annotations

import argparse
import itertools
import xml.etree.ElementTree as ET

import numpy as np

from genmech.robots import get_robot_spec
from genmech.utils.paths import resolve as resolve_repo_path


# Below this, a "penetration" is numerical noise on coincident surfaces rather
# than a real overlap.
EPS_M = 1e-5

MAX_VERTS = 400   # subsample dense meshes; the arm's STLs run to tens of thousands


def merge_map(urdf_path) -> dict[str, str]:
    """Link -> the body it becomes after ``merge_fixed_joints=True``.

    The importer collapses every fixed-jointed child into its parent, which is
    why ``spec.adjacent_links`` is written in terms of ``iiwa14_link_7`` rather
    than the URDF's own palm link. Without applying the same collapse here, the
    adjacency map matches nothing around the palm and the check reports overlaps
    that PhysX has already been told to ignore.
    """
    root = ET.parse(urdf_path).getroot()
    fixed_parent: dict[str, str] = {}
    for j in root.findall("joint"):
        if j.get("type") == "fixed":
            fixed_parent[j.find("child").get("link")] = j.find("parent").get("link")

    def resolve(link: str) -> str:
        seen = {link}
        while link in fixed_parent:
            link = fixed_parent[link]
            if link in seen:      # malformed URDF; stop rather than loop
                break
            seen.add(link)
        return link

    return {l.get("name"): resolve(l.get("name")) for l in root.findall("link")}


def jointed_pairs(urdf_path, merged: dict[str, str]) -> set[frozenset[str]]:
    """Parent/child pairs, which PhysX filters automatically.

    Expressed in merged body names, since that is what survives import.
    """
    root = ET.parse(urdf_path).getroot()
    out = set()
    for j in root.findall("joint"):
        a = merged[j.find("parent").get("link")]
        b = merged[j.find("child").get("link")]
        if a != b:
            out.add(frozenset((a, b)))
    return out


def filtered_pairs(spec) -> set[frozenset[str]]:
    """Pairs the scene explicitly filters, from the spec's adjacency map."""
    out: set[frozenset[str]] = set()
    for a, others in spec.adjacent_links.items():
        for b in others:
            out.add(frozenset((a, b)))
    return out


def link_geometry_meshes(urdf, base_dir, merged: dict[str, str],
                         only_prefix: str | None = None,
                         which: str = "collision",
                         hull: bool = True) -> dict[str, "object"]:
    """One mesh per POST-MERGE body, in world coords at the current pose.

    ``which`` selects ``"collision"`` or ``"visual"`` geometry. Physics only ever
    means collision; visual exists so a viewer can show what a render will look
    like, which for generated hands is NOT the same shape -- URDF has no capsule
    primitive, so a capsule is one collision cylinder but a cylinder plus two
    spheres in visual.

    Fixed-jointed links contribute their geometry to whichever body they collapse
    into, so a fingertip pad is checked as part of its distal phalanx rather than
    as a separate object that trivially overlaps it.

    ``only_prefix`` restricts the work to links whose name starts with it, which
    skips loading the arm's STLs entirely -- worth ~seconds per call, and the
    difference between a live design-space viewer and an unusable one.
    """
    import trimesh

    pieces: dict[str, list] = {}
    for name, link in urdf.link_map.items():
        if only_prefix is not None and not name.startswith(only_prefix):
            continue
        body = merged.get(name, name)
        elements = link.collisions if which == "collision" else link.visuals
        for coll in elements:
            mesh = _geometry_to_mesh(coll.geometry, base_dir, hull=hull)
            if mesh is None:
                continue
            mesh = mesh.copy()
            origin = coll.origin if coll.origin is not None else np.eye(4)
            mesh.apply_transform(urdf.get_transform(name) @ origin)
            pieces.setdefault(body, []).append(mesh)

    return {b: trimesh.util.concatenate(p) for b, p in pieces.items() if p}


def link_collision_meshes(urdf, base_dir, merged, only_prefix=None):
    """Collision geometry. Thin alias -- physics is always collision."""
    return link_geometry_meshes(urdf, base_dir, merged, only_prefix,
                                which="collision")


def _geometry_to_mesh(geometry, base_dir, hull: bool = True):
    """URDF collision geometry -> trimesh, for meshes and primitives alike.

    Mesh filenames are relative to the URDF's own directory (the generated hands
    reach the shared arm meshes with ``../kuka_sharpa_description/...``), so they
    resolve against ``base_dir`` rather than the CWD.
    """
    import trimesh

    if geometry.mesh is not None:
        m = trimesh.load(str(base_dir / geometry.mesh.filename), force="mesh")
        if geometry.mesh.scale is not None:
            m = m.copy()
            m.apply_scale(geometry.mesh.scale)
        # PhysX does not simulate the triangle mesh. Isaac Lab's converter
        # stamps approximation="convexHull" on every mesh collider, verified in
        # the baked USD for both SHARPA (34/34) and Allegro (26/26), so what
        # actually resolves contacts is the HULL -- which fills every concavity
        # and strictly contains the mesh. Checking the mesh under-reports
        # overlap, sometimes by a lot, because the detail is not real.
        #
        # Generated hands are unaffected: their capsules and palm box are
        # analytic primitives with approximation "None", i.e. exact.
        #
        # `hull` is on for the CHECK, which must match physics, and off for
        # VIEWERS, which should show the geometry the asset actually declares --
        # a hulled arm renders as blocky lumps that look like a bug.
        return m.convex_hull if hull else m
    if geometry.box is not None:
        return trimesh.creation.box(extents=geometry.box.size)
    if geometry.cylinder is not None:
        # Generated hands emit cylinders that the importer turns into capsules,
        # adding a hemisphere at each end. trimesh's `height` means the same
        # thing, so passing the URDF length through reproduces exactly the shape
        # PhysX ends up with -- including the caps, which a bare cylinder would
        # miss and so under-report overlap.
        return trimesh.creation.capsule(
            height=geometry.cylinder.length, radius=geometry.cylinder.radius
        )
    if geometry.sphere is not None:
        return trimesh.creation.icosphere(radius=geometry.sphere.radius)
    return None


def penetration(a, b) -> float:
    """Lower bound on how deep two meshes overlap, in metres. 0.0 if disjoint."""
    import trimesh

    # Cheap reject on bounding boxes before any proximity query.
    if (a.bounds[0] > b.bounds[1]).any() or (b.bounds[0] > a.bounds[1]).any():
        return 0.0

    worst = 0.0
    for inner, outer in ((a, b), (b, a)):
        v = inner.vertices
        if len(v) > MAX_VERTS:
            idx = np.linspace(0, len(v) - 1, MAX_VERTS).astype(int)
            v = v[idx]
        try:
            sd = trimesh.proximity.signed_distance(outer, v)
        except Exception:
            continue
        if len(sd):
            worst = max(worst, float(np.max(sd)))
    return max(worst, 0.0)


def check(spec_name: str, *, verbose: bool) -> int:
    import yourdfpy

    spec = get_robot_spec(spec_name)
    urdf_path = resolve_repo_path(spec.urdf_path)
    urdf = yourdfpy.URDF.load(
        str(urdf_path),
        load_meshes=False,
        load_collision_meshes=False,
        build_scene_graph=True,
        build_collision_scene_graph=True,
    )

    # Home pose: the configuration every episode actually starts from.
    home = {
        **spec.arm_default_joint_pos_resolved(start_arm_higher=False),
        **spec.hand_default_joint_pos,
    }
    cfg = {n: home.get(n, 0.0) for n in urdf.actuated_joint_names}
    urdf.update_cfg(np.array([cfg[n] for n in urdf.actuated_joint_names]))

    merged = merge_map(urdf_path)
    meshes = link_collision_meshes(urdf, urdf_path.parent, merged)
    skip = jointed_pairs(urdf_path, merged) | filtered_pairs(spec)

    names = sorted(meshes)
    hits: list[tuple[str, str, float]] = []
    checked = 0
    for a, b in itertools.combinations(names, 2):
        if frozenset((a, b)) in skip:
            continue
        checked += 1
        d = penetration(meshes[a], meshes[b])
        if d > EPS_M:
            hits.append((a, b, d))

    print(f"\n=== {spec_name} ===")
    print(f"  {len(names)} links with collision geometry, "
          f"{checked} unfiltered pairs checked "
          f"({len(skip)} pairs skipped: jointed or in adjacent_links)")
    if not hits:
        print(f"  NO self-collision at the home pose")
    else:
        hits.sort(key=lambda h: -h[2])
        print(f"  {len(hits)} OVERLAPPING PAIR(S) at the home pose:")
        for a, b, d in hits[: (len(hits) if verbose else 15)]:
            print(f"    {d * 1000:6.2f} mm   {a}  <->  {b}")
        if not verbose and len(hits) > 15:
            print(f"    ... and {len(hits) - 15} more (--verbose for all)")
    return len(hits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_spec", default="sharpa_iiwa14,gen_sharpa_like")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    total = 0
    for name in (n.strip() for n in args.robot_spec.split(",") if n.strip()):
        total += check(name, verbose=args.verbose)

    print()
    print("SELF-COLLISION CHECK COMPLETE"
          if total == 0 else
          f"SELF-COLLISION CHECK: {total} overlapping pair(s) found")


if __name__ == "__main__":
    main()
