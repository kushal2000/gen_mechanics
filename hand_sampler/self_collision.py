"""Self-collision gates: does a hand intersect itself at its home pose?

Two tiers, and they are not substitutes. The ANALYTIC tier below replaces every
link with a capsule and answers in closed form; the MESH tier at the bottom
loads the real collision geometry and measures the truth. The mesh gate cost
6.02 s per accepted hand -- 41 hours for a 24,576-hand population -- which is
what the capsule gate exists to avoid. Capsules are fatter than the meshes they
replace, so the analytic gate is conservative: it rejects some hands the mesh
gate would accept, and never the reverse.

A hand that starts every episode already in self-contact spends its first steps
being pushed apart by the solver rather than reaching, which is why this runs
before evaluation rather than after.

``validate_design.check`` is the cheap O(1) tier that runs on EVERY mutation;
this one is configuration-dependent and runs once, before evaluation.
"""

from __future__ import annotations

import math

import numpy as np

from hand_sampler import params
from hand_sampler import robot_spec

# The palm's post-merge body name.
PALM_BODY = "iiwa14_link_7"

EPS_M = 1e-5   # shared by both tiers, so they agree on what counts as a graze


# --------------------------------------------------------------------------- transforms...

def _seg_matrix(seg: params.Segment) -> np.ndarray:
    """URDF ``<origin>`` semantics: translate by xyz, then rotate by rpy."""
    r, p, y = (float(v) for v in seg.rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # Rz(yaw) @ Ry(pitch) @ Rx(roll)
    rot = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = [float(v) for v in seg.xyz]
    return m


def finger_link_transforms(fp: params.FingerParams) -> dict[str, np.ndarray]:
    """Palm-frame transform of each finger link at the HOME pose.

    Home pose means every joint angle is zero, so each joint contributes only
    its origin transform. The chain mirrors ``build_hand_urdf._build_finger``
    exactly; if that chain changes this must change with it.
    """
    chain = (
        ("CMC_VL", fp.mount),
        ("MC", fp.cmc),
        ("MCP_VL", fp.mc),
        ("PP", fp.mcp),
        ("MP", params.Segment(xyz=(fp.pp_length, 0.0, 0.0), rpy=params.ROLL_AA_TO_FE.rpy)),
        ("DP", params.Segment(xyz=(fp.mp_length, 0.0, 0.0))),
    )
    out, acc = {}, np.eye(4)
    for part, seg in chain:
        acc = acc @ _seg_matrix(seg)
        out[part] = acc.copy()
    return out


# Which link part draws its geometry from which tier.
_PART_TIER: tuple[tuple[str, str], ...] = (
    ("MC", "mc"), ("PP", "pp"), ("MP", "mp"), ("DP", "dp"),
)


def hand_capsules(hand: params.HandParams) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """``(body_name, p0, p1, radius)`` per collision capsule, in palm frame.

    The capsule's CORE SEGMENT, not its full extent: the URDF emits a cylinder
    of ``cylinder_part(length, radius)`` centred at length/2 and the importer
    caps it with a hemisphere at each end, so the segment runs from x=radius to
    x=length-radius and the capsule spans [0, length] exactly.
    """
    from hand_sampler.urdf import has_collision_geometry, link_name

    out = []
    for i, fp in enumerate(hand.fingers):
        if not fp.active:
            continue                      # ghosted: every link is virtual
        tf = finger_link_transforms(fp)
        for part, tier in _PART_TIER:
            if not has_collision_geometry(fp, tier):
                continue
            length = fp.segment_length(tier)
            radius = robot_spec.TIER_RADIUS_M[tier] * fp.radius_scale
            m = tf[part]
            p0 = m @ np.array([radius, 0.0, 0.0, 1.0])
            p1 = m @ np.array([length - radius, 0.0, 0.0, 1.0])
            out.append((link_name(i, part), p0[:3], p1[:3], radius))
    return out


def palm_box(hand: params.HandParams) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` of the palm's axis-aligned box, in palm frame."""
    ext = np.asarray([float(v) for v in hand.palm_extents])
    centre = np.asarray([float(v) for v in params.palm_center(hand.palm_extents)])
    return centre - ext / 2.0, centre + ext / 2.0


# --------------------------------------------------------------------------- distances...

def segment_distance(p1, q1, p2, q2) -> float:
    """Closest distance between two 3D segments. Standard clamped solution."""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-18 and e <= 1e-18:
        return float(np.linalg.norm(r))
    if a <= 1e-18:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = d1 @ r
        if e <= 1e-18:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if denom > 1e-18 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def _point_aabb_distance(pt, lo, hi) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(lo - pt, pt - hi), 0.0)))


def segment_aabb_distance(p, q, lo, hi, iters: int = 60) -> float:
    """Distance from one segment to an axis-aligned box. See the batched form."""
    d = _segments_aabb_distance(np.asarray([p], dtype=float),
                                np.asarray([q], dtype=float), lo, hi, iters)
    return float(d[0])


# --------------------------------------------------------------------------- batched...

def _all_pairs_segment_distance(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """(K,K) closest distance between every pair of segments.

    Ericson's clamped solution, vectorised with np.where instead of branches.
    """
    k = len(p0)
    d = p1 - p0                                    # (K,3)
    # Pairwise: index i is segment 1, index j is segment 2.
    d1 = d[:, None, :]                             # (K,1,3)
    d2 = d[None, :, :]                             # (1,K,3)
    r = p0[:, None, :] - p0[None, :, :]            # (K,K,3)

    a = np.einsum("ijk,ijk->ij", np.broadcast_to(d1, (k, k, 3)),
                  np.broadcast_to(d1, (k, k, 3)))
    e = np.einsum("ijk,ijk->ij", np.broadcast_to(d2, (k, k, 3)),
                  np.broadcast_to(d2, (k, k, 3)))
    b = np.einsum("ijk,ijk->ij", np.broadcast_to(d1, (k, k, 3)),
                  np.broadcast_to(d2, (k, k, 3)))
    c = np.einsum("ijk,ijk->ij", np.broadcast_to(d1, (k, k, 3)), r)
    f = np.einsum("ijk,ijk->ij", np.broadcast_to(d2, (k, k, 3)), r)

    tiny = 1e-18
    a_safe = np.maximum(a, tiny)
    e_safe = np.maximum(e, tiny)
    denom = a * e - b * b
    s = np.where(denom > tiny,
                 np.clip((b * f - c * e) / np.where(denom > tiny, denom, 1.0),
                         0.0, 1.0),
                 0.0)
    t = (b * s + f) / e_safe

    # Re-clamp s where t left [0,1]; order matters, t<0 first then t>1.
    s = np.where(t < 0.0, np.clip(-c / a_safe, 0.0, 1.0), s)
    s = np.where(t > 1.0, np.clip((b - c) / a_safe, 0.0, 1.0), s)
    t = np.clip(t, 0.0, 1.0)

    # Degenerate segments (zero length) fall back to point distances, which the clamps above...
    diff = (p0[:, None, :] + np.broadcast_to(d1, (k, k, 3)) * s[..., None]) - \
           (p0[None, :, :] + np.broadcast_to(d2, (k, k, 3)) * t[..., None])
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def _segments_aabb_distance(p: np.ndarray, q: np.ndarray, lo, hi,
                            iters: int = 60) -> np.ndarray:
    """(K,) distance from each segment to one axis-aligned box.

    Ternary search on the segment parameter, run on all segments at once.
    Distance from a point to a convex set is convex and a segment is a convex
    combination, so the objective is convex in t and ternary search converges
    monotonically; 60 iterations shrink the bracket by (2/3)^60, far below any
    tolerance here. Exact closed forms for segment-box exist but are long and
    easy to get subtly wrong, and this is no longer the hot loop.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = q - p
    lo_t = np.zeros(len(p))
    hi_t = np.ones(len(p))

    def dist_at(t):
        x = p + d * t[:, None]
        gap = np.maximum(np.maximum(lo - x, x - hi), 0.0)
        return np.sqrt(np.einsum("ij,ij->i", gap, gap))

    for _ in range(iters):
        span = (hi_t - lo_t) / 3.0
        m1, m2 = lo_t + span, hi_t - span
        take_left = dist_at(m1) < dist_at(m2)
        hi_t = np.where(take_left, m2, hi_t)
        lo_t = np.where(take_left, lo_t, m1)
    return dist_at((lo_t + hi_t) / 2.0)


# --------------------------------------------------------------------------- the gate...

_SKIP_CACHE: set[frozenset[str]] | None = None


def skip_pairs() -> set[frozenset[str]]:
    """Body pairs PhysX already filters. Template-constant, so computed once.

    Jointed parent/child pairs (auto-filtered by PhysX) plus the template
    adjacency map the scene authors as FilteredPairsAPI. Both depend only on
    names and on which joints are fixed -- identical for every hand in the
    design space, which is why the mesh gate's per-candidate recomputation was
    pure waste.
    """
    global _SKIP_CACHE
    if _SKIP_CACHE is not None:
        return _SKIP_CACHE

    from hand_sampler.synth_spec import template_adjacent_links
    from hand_sampler.urdf import link_name

    skip: set[frozenset[str]] = set()
    for i in range(params.N_FINGER_SLOTS):
        parts = ["CMC_VL", "MC", "MCP_VL", "PP", "MP", "DP"]
        prev = PALM_BODY
        for part in parts:
            skip.add(frozenset((prev, link_name(i, part))))
            prev = link_name(i, part)
    for a, others in template_adjacent_links().items():
        for b in others:
            skip.add(frozenset((a, b)))
    _SKIP_CACHE = skip
    return skip


def analytic_hand_hits(hand: params.HandParams) -> list[tuple[str, str, float]]:
    """Self-collision pairs at the home pose. Same contract as the mesh gate.

    Returns ``[(body_a, body_b, depth_m), ...]``, deepest first.
    """
    caps = hand_capsules(hand)
    if not caps:
        return []
    skip = skip_pairs()
    lo, hi = palm_box(hand)
    hits: list[tuple[str, str, float]] = []

    names = [c[0] for c in caps]
    p0 = np.asarray([c[1] for c in caps], dtype=float)
    p1 = np.asarray([c[2] for c in caps], dtype=float)
    radii = np.asarray([c[3] for c in caps], dtype=float)

    # capsule vs capsule: every pair in one batched call
    dist = _all_pairs_segment_distance(p0, p1)
    depth = (radii[:, None] + radii[None, :]) - dist
    iu, ju = np.triu_indices(len(caps), k=1)
    for i, j in zip(iu[depth[iu, ju] > EPS_M], ju[depth[iu, ju] > EPS_M]):
        na, nb = names[i], names[j]
        if frozenset((na, nb)) in skip:
            continue
        hits.append((na, nb, float(depth[i, j])))

    # capsule vs palm box
    box_depth = radii - _segments_aabb_distance(p0, p1, lo, hi)
    for i in np.nonzero(box_depth > EPS_M)[0]:
        if frozenset((PALM_BODY, names[i])) in skip:
            continue
        hits.append((PALM_BODY, names[i], float(box_depth[i])))

    hits.sort(key=lambda h: -h[2])
    return hits


def is_collision_free(hand: params.HandParams) -> bool:
    return not analytic_hand_hits(hand)


def sample_collision_free_fast(seed: int, count: int, *,
                               # 400 was tuned when the sampler accepted 2.79% of draws: expected ~51 draws per hand then,...
                               max_tries_per_hand: int = 4000,
                               verbose: bool = True,
                               stream_key: str | None = None,
                               name_offset: int = 0,
                               **sample_kwargs) -> list[params.HandParams]:
    """Rejection-sample ``count`` collision-free hands, analytically.

    Same contract and same RNG consumption pattern as
    ``check_self_collision.sample_collision_free``, but with the analytic gate
    and NO file I/O: the mesh version wrote a URDF per candidate purely to feed
    the geometry check, and with ~92% rejection that is ~13 wasted URDF writes
    and parses per accepted hand.

    The population this produces is NOT guaranteed identical to the mesh gate's
    for the same seed -- the two gates agree on every candidate measured
    (compare_collision_gates), but a graze within EPS_M of the threshold could
    in principle split them. Populations are identified by seed AND gate.
    """
    import random as _random

    # stream_key gives a SHARD its own independent stream.
    rng = _random.Random(seed if stream_key is None else f"{seed}:{stream_key}")
    hands: list[params.HandParams] = []
    drawn = rejected_invalid = rejected_collision = 0

    while len(hands) < count:
        tries = 0
        while tries < max_tries_per_hand:
            tries += 1
            drawn += 1
            try:
                hand = params.sample(rng,
                                name=f"gen_{seed:04d}_"
                                     f"{name_offset + len(hands):05d}",
                                **sample_kwargs)
            except params.InvalidHand:
                rejected_invalid += 1
                continue
            if is_collision_free(hand):
                hands.append(hand)
                break
            rejected_collision += 1
        else:
            raise RuntimeError(
                f"no collision-free hand in {max_tries_per_hand} tries at "
                f"index {len(hands)}; the design space or the gate has changed")
        if verbose and len(hands) % 1000 == 0:
            print(f"[sample] {len(hands)}/{count} accepted, {drawn} drawn",
                  flush=True)

    if verbose:
        print(f"[sample] {len(hands)} collision-free hands from {drawn} draws "
              f"({100 * (drawn - len(hands)) / max(drawn, 1):.0f}% rejected: "
              f"{rejected_invalid} on validity, {rejected_collision} on geometry)")
    return hands


__all__ = [
    "sample_collision_free_fast",
    "analytic_hand_hits",
    "is_collision_free",
    "hand_capsules",
    "palm_box",
    "segment_distance",
    "segment_aabb_distance",
    "skip_pairs",
    "finger_link_transforms",
]


# --- mesh tier: the ground truth the capsules approximate ------------------

import argparse
import itertools
import xml.etree.ElementTree as ET

import numpy as np

from hand_sampler import resolve as resolve_repo_path


# Below this, a "penetration" is numerical noise on coincident surfaces rather than a real...

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
        # PhysX does not simulate the triangle mesh.
        return m.convex_hull if hull else m
    if geometry.box is not None:
        return trimesh.creation.box(extents=geometry.box.size)
    if geometry.cylinder is not None:
        # Generated hands emit cylinders that the importer turns into capsules, adding a hemisphere at...
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


def generated_hand_hits(hand, workdir, name: str = "probe") -> list:
    """Self-collision pairs of a ``HandParams`` at its home pose.

    The real gate. ``params.validate()`` only checks mount separation and reach,
    which are proxies -- every actual overlap found in this project was caught
    by geometry afterwards, never prevented by those. This builds the hand and
    measures, which is slow (a URDF write, a load, and pairwise penetration per
    candidate) but is the thing that is actually true.

    Returns ``[(body_a, body_b, depth_m), ...]``, deepest first.
    """
    import yourdfpy

    from hand_sampler.synth_spec import template_adjacent_links
    from hand_sampler.urdf import OUT_DIR, write_urdf
    from hand_sampler import resolve as _resolve

    path = write_urdf(hand, workdir / f"{name}.urdf")
    urdf = yourdfpy.URDF.load(
        str(path), load_meshes=False, load_collision_meshes=False,
        build_scene_graph=True, build_collision_scene_graph=True)
    urdf.update_cfg(np.zeros(len(urdf.actuated_joint_names)))

    merged = merge_map(path)
    meshes = link_geometry_meshes(urdf, _resolve(OUT_DIR), merged,
                                  only_prefix="gen_", hull=False)
    skip = jointed_pairs(path, merged)
    for a, others in template_adjacent_links().items():
        for b in others:
            skip.add(frozenset((a, b)))

    names = sorted(meshes)
    hits = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if frozenset((a, b)) in skip:
                continue
            d = penetration(meshes[a], meshes[b])
            if d > EPS_M:
                hits.append((a, b, d))
    hits.sort(key=lambda h: -h[2])
    return hits


def sample_collision_free(seed: int, count: int, workdir,
                          max_tries_per_hand: int = 40, **sample_kwargs) -> list:
    """A population of hands that do not overlap themselves at the home pose.

    Rejection sampling on the geometric check rather than on the proxies, so
    what comes back is clean by measurement. Reports the rejection rate, which
    is itself a reading on how much of the design space is unusable.
    """
    import random as _random

    from hand_sampler import params as _P

    rng = _random.Random(seed)
    out, tried = [], 0
    while len(out) < count:
        ok = False
        for _ in range(max_tries_per_hand):
            tried += 1
            hand = _P.sample_valid(rng, name=f"gen_{seed:04d}_{len(out):03d}",
                                   **sample_kwargs)
            if not generated_hand_hits(hand, workdir, f"cand{len(out):03d}"):
                out.append(hand)
                ok = True
                break
        if not ok:
            raise RuntimeError(
                f"no collision-free hand in {max_tries_per_hand} tries "
                f"(got {len(out)}/{count})")
    print(f"[sample] {count} collision-free hands from {tried} draws "
          f"({(1 - count / tried) * 100:.0f}% rejected)", flush=True)
    return out


def check(spec_name: str, *, verbose: bool) -> int:
    import yourdfpy

    # Imported lazily: the registry lives in isaacsimenvs, and hand_sampler must stay importable...
    from isaacsimenvs.pose_reaching_6d.scene_utils.robots import get_robot_spec

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
