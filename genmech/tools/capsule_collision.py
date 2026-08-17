"""Analytic self-collision gate for generated hands.

The mesh gate (``check_self_collision.generated_hand_hits``) measures the truth,
and measured 6.02 s per ACCEPTED hand -- 41 hours for a 24,576-hand population.
That number is absurd on its face, and the reason is that it answers a question
about capsules using mesh machinery. Per candidate it:

  * re-parses the entire SHARPA URDF (build_urdf does this every call),
  * writes a URDF to disk,
  * re-parses it with yourdfpy and builds two scene graphs,
  * tessellates every link into a trimesh capsule,
  * runs ``trimesh.proximity.signed_distance`` BVH queries over ~400 link pairs.

But a generated hand's collision geometry is exactly: one box (the palm) and up
to 20 capsules. Two capsules overlap iff the distance between their core
segments is less than the sum of their radii -- a closed form. So this computes
the same predicate directly, with no file, no parse, and no mesh.

WHAT IS SHARED WITH THE MESH GATE, so the two cannot drift:

  * ``A.cylinder_part`` for the capsule's cylindrical section, so the shape
    assumed here is the shape the URDF emits and PhysX simulates;
  * ``build_hand_urdf.has_collision_geometry`` for which segments carry a shape
    at all -- a segment below MIN_SEGMENT_M is a bare virtual link, and one
    shorter than its own diameter keeps its mass but drops its geometry;
  * ``template_adjacent_links()`` plus the jointed pairs for what to skip.

THE SKIP SET IS COMPUTED ONCE. It depends only on link and joint NAMES and on
which joints are fixed, all of which are template constants -- the same for
every hand in the design space. The mesh gate recomputed it per candidate by
parsing a freshly written URDF, which is most of what made it slow.

This gate is only trustworthy insofar as it agrees with the mesh gate. Check
that with ``genmech.tools.compare_collision_gates`` before trusting a population
built on it; a faster gate that accepts hands the real check would reject is not
an optimisation, it is a corrupted population.
"""

from __future__ import annotations

import math

import numpy as np

from genmech.robots.generated import params as P
from genmech.robots.generated import sharpa_anchors as A

# The palm's post-merge body name. gen_palm is fixed-jointed to the flange, so
# merge_fixed_joints folds it into the arm's last link, and the adjacency map is
# written in those terms.
PALM_BODY = "iiwa14_link_7"

EPS_M = 1e-5   # matches check_self_collision.EPS_M


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

def _seg_matrix(seg: P.Segment) -> np.ndarray:
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


def finger_link_transforms(fp: P.FingerParams) -> dict[str, np.ndarray]:
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
        ("MP", P.Segment(xyz=(fp.pp_length, 0.0, 0.0), rpy=P.ROLL_AA_TO_FE.rpy)),
        ("DP", P.Segment(xyz=(fp.mp_length, 0.0, 0.0))),
    )
    out, acc = {}, np.eye(4)
    for part, seg in chain:
        acc = acc @ _seg_matrix(seg)
        out[part] = acc.copy()
    return out


# Which link part draws its geometry from which tier. Mirrors
# build_hand_urdf.LINK_PARTS; virtual parts carry no shape.
_PART_TIER: tuple[tuple[str, str], ...] = (
    ("MC", "mc"), ("PP", "pp"), ("MP", "mp"), ("DP", "dp"),
)


def hand_capsules(hand: P.HandParams) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """``(body_name, p0, p1, radius)`` per collision capsule, in palm frame.

    The capsule's CORE SEGMENT, not its full extent: the URDF emits a cylinder
    of ``cylinder_part(length, radius)`` centred at length/2 and the importer
    caps it with a hemisphere at each end, so the segment runs from x=radius to
    x=length-radius and the capsule spans [0, length] exactly.
    """
    from genmech.tools.build_hand_urdf import has_collision_geometry, link_name

    out = []
    for i, fp in enumerate(hand.fingers):
        if not fp.active:
            continue                      # ghosted: every link is virtual
        tf = finger_link_transforms(fp)
        for part, tier in _PART_TIER:
            if not has_collision_geometry(fp, tier):
                continue
            length = fp.segment_length(tier)
            radius = A.TIER_RADIUS_M[tier] * fp.radius_scale
            m = tf[part]
            p0 = m @ np.array([radius, 0.0, 0.0, 1.0])
            p1 = m @ np.array([length - radius, 0.0, 0.0, 1.0])
            out.append((link_name(i, part), p0[:3], p1[:3], radius))
    return out


def palm_box(hand: P.HandParams) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` of the palm's axis-aligned box, in palm frame."""
    ext = np.asarray([float(v) for v in hand.palm_extents])
    centre = np.asarray([float(v) for v in P.palm_center(hand.palm_extents)])
    return centre - ext / 2.0, centre + ext / 2.0


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# batched distances
#
# The scalar forms above are correct and were the first implementation, and they
# cost 13.3 ms per candidate hand -- for roughly 200 closed-form distances that
# should take microseconds. The time was not in the mathematics but in numpy
# call overhead: ~2 us of dispatch per operation on a 3-element array, paid
# ~1600 times. Batching every pair into one array turns that into a fixed number
# of numpy calls on (K,) and (K,K) arrays, and the overhead disappears.
# ---------------------------------------------------------------------------

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

    # Degenerate segments (zero length) fall back to point distances, which the
    # clamps above already produce: s and t collapse to 0.
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


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

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

    from genmech.robots.generated.synth_spec import template_adjacent_links
    from genmech.tools.build_hand_urdf import link_name

    skip: set[frozenset[str]] = set()
    for i in range(P.N_FINGER_SLOTS):
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


def analytic_hand_hits(hand: P.HandParams) -> list[tuple[str, str, float]]:
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


def is_collision_free(hand: P.HandParams) -> bool:
    return not analytic_hand_hits(hand)


def sample_collision_free_fast(seed: int, count: int, *,
                               # 400 was tuned when the sampler accepted 2.79%
                               # of draws: expected ~51 draws per hand then, and
                               # (1-0.0279)^400 gave ~0.3 expected failures over
                               # 24,576 designs -- it squeaked through. Excluding
                               # the degenerate-capsule band dropped acceptance to
                               # 1.97%, which makes it ~9 expected failures, and
                               # the build died at index 14,740. 4000 puts the
                               # expectation at e^-79, i.e. never, and costs
                               # nothing: the budget is only consumed on the rare
                               # hand that needs it.
                               max_tries_per_hand: int = 4000,
                               verbose: bool = True,
                               stream_key: str | None = None,
                               name_offset: int = 0,
                               **sample_kwargs) -> list[P.HandParams]:
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

    # stream_key gives a SHARD its own independent stream. The default stream is
    # seeded by `seed` alone and consumed sequentially, so shard N cannot know
    # where shard N-1 stopped -- there is no way to split one stream across
    # processes. Independent streams per shard sample the same distribution
    # through the same gate, so the population is equally valid; it is simply
    # not the same draw a single-process build would make. Populations are
    # identified by seed AND gate AND shard count.
    rng = _random.Random(seed if stream_key is None else f"{seed}:{stream_key}")
    hands: list[P.HandParams] = []
    drawn = rejected_invalid = rejected_collision = 0

    while len(hands) < count:
        tries = 0
        while tries < max_tries_per_hand:
            tries += 1
            drawn += 1
            try:
                hand = P.sample(rng,
                                name=f"gen_{seed:04d}_"
                                     f"{name_offset + len(hands):05d}",
                                **sample_kwargs)
            except P.InvalidHand:
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
