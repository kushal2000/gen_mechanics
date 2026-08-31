"""Mutation operators for evolving a cached hand population.

One operator per call, one child per parent, INDEX ALIGNED: child ``i`` of every
operator manifest is the mutant of parent ``i``. That alignment is the whole
point. Env ``i`` holds design ``i`` and object ``i % pool_size``, so evaluating a
mutant manifest puts each child on the SAME OBJECT its parent held, and

    delta_i = child_i - parent_i

is paired within-parent and within-object. The object is the largest confound in
the population study (docs/analysis.md 5); this cancels it outright rather than
averaging over it, and it costs nothing but keeping the order.

WHAT THE GENOTYPE ACTUALLY IS. Not the dataclass fields. ``FingerParams.mount``
is DERIVED -- ``mount_on_face(face, u_frac, v_frac, roll, tilt, azimuth,
palm_extents)`` -- and cached beside its provenance in ``mount_params``. Every
operator here edits ``mount_params`` and recomputes; editing the Segment
directly would desync the two and lose the face decomposition that ``mounting``
exists to explore.

WHY NO OPERATOR MUTATES ROLL. ``population.build_population`` runs
``align_flexion_downward`` on every accepted hand, which sets
``new_roll = roll + optimal_roll_offset(...)`` -- driving roll to the value that
curls the finger toward the workspace. So roll is not a free parameter in the
parent population; it is a function of the rest of the geometry. Mutating it and
re-aligning would erase the mutation, and mutating it without re-aligning would
just mis-point the finger. Instead every operator perturbs the geometry and then
RE-ALIGNS, which is also what keeps mutants on the same manifold the parents
live on: change a palm dimension or a segment length and the optimal roll moves,
so a mutant that skipped alignment would differ from its parent in two ways
rather than one.

Operators:

    palm            palm_extents; every mount recomputed on the new faces
    scale           segment lengths + radius_scale, multiplicative
    mounting        one finger's position and lean (NOT roll -- see above)
    num_joints_up   one finger's ladder position +1
    num_joints_down one finger's ladder position -1

MEASURED ON 60 SEED-3 PARENTS AT alpha = 0.1, because the numbers are not what
the parameter names suggest:

    operator          fail    median |d reach|  median |d separation|
    palm               0%          2.5 mm             3.2 mm
    scale              0%          3.2 mm             0.0 mm
    mounting           0%          0.4 mm             1.2 mm
    num_joints_up     18%          0.0 mm             0.0 mm
    num_joints_down   13%          0.0 mm             0.0 mm

`mounting` moves the MOUNT 5.6 mm -- exactly its 5 mm/axis sigma -- but moves
separation by only 1.2 mm, because the two fingers usually sit on DIFFERENT
faces (-y 45, +z 42, +y 41 over 60 parents) and an in-face step is largely
perpendicular to the inter-mount vector. It is a where-on-the-face-and-lean
operator; `palm` is the separation lever, because it scales every face at once.
37% of mutated mounts also land exactly on the wrist keep-out plane, where part
of the step is absorbed -- hence `keepout_clamped` on every entry.

`num_joints_up` fails only when every active finger is already at 6 joints,
which is deterministic. `num_joints_down` fails by SELF-COLLISION (340
rejections over 60 parents): 5->4 zeroes the metacarpal, pulling the proximal
capsule back against the palm and its neighbour, and MIN_MOUNT_SEPARATION is
only 15 mm. The metacarpal is part of what holds fingers apart.

``finger_count`` is deliberately absent, and its absence is permanent for any
population descended from these: none of the five changes how many fingers are
active, so a topology class that dies in one generation cannot come back.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from dataclasses import replace
from pathlib import Path

from genmech.robots.generated import params as P
from genmech.robots.generated import sharpa_anchors as anchors
# Private by convention, not by intent: this module is part of the same package
# and must reproduce sample()'s construction rules exactly. Re-deriving them
# here is how a mutant silently stops being drawn from the parents' space.
from genmech.robots.generated.params import _LIM_FINGER, _LIM_THUMB, _limits
from genmech.robots.generated.population import (
    _hits, _roundtrip_ok, hand_from_json, hand_to_json,
)
from genmech.utils.paths import resolve as resolve_repo_path

OPERATORS = ("palm", "scale", "mounting", "num_joints_up", "num_joints_down")

# `mounting` steps in METRES on the palm face, not in u_frac/v_frac. The faces
# differ 2-4x in span (+z spans palm_x by palm_y; +-y span palm_x by palm_z) and
# the spans shrink with the palm, so a fixed fractional step is a different
# physical move on every hand -- while the axis that matters, mount separation,
# is measured in metres with an optimum at 4-5 cm.
MOUNT_POS_REF_M = 0.050

# Tilt is jittered in the tangent plane, so this is the reference for the
# finger's POINTING DIRECTION, not for the tilt scalar. (tilt, azimuth) are polar
# coordinates: azimuth does nothing at tilt = 0 and a fixed azimuth step is a
# vanishing angular move near it, so perturbing them independently gives a step
# size that depends on where the finger already points.
MOUNT_TILT_REF_RAD = P.MOUNT_TILT_RANGE[1] - P.MOUNT_TILT_RANGE[0]

_PALM_RANGES = (P.PALM_X_RANGE, P.PALM_Y_RANGE, P.PALM_Z_RANGE)
_TIER_RANGE = {"mc": P.MC_LENGTH_RANGE, "pp": P.PP_LENGTH_RANGE,
               "mp": P.MP_LENGTH_RANGE, "dp": P.DP_LENGTH_RANGE}


class MutationImpossible(ValueError):
    """This operator cannot act on this hand (e.g. every finger is at the ladder end)."""


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------

def _reflect(x: float, lo: float, hi: float) -> float:
    """Fold x back into [lo, hi]. Reflection rather than clipping, because
    clipping piles probability mass on the boundary -- the same reason
    params.sample rejects instead of clamping."""
    if hi <= lo:
        return lo
    span = hi - lo
    y = (x - lo) % (2.0 * span)
    return lo + (y if y <= span else 2.0 * span - y)


def _wrap(x: float, period: float = 2.0 * math.pi) -> float:
    return x % period


def _reproject_length(new: float, old: float, tier: str, radius_scale: float,
                      ) -> float:
    """Keep a segment out of the band where a capsule cannot be built.

    build_hand_urdf._is_degenerate drops the collider for any segment in
    [MIN_SEGMENT_M, 2*r), where r = TIER_RADIUS_M[tier] * radius_scale: the link
    keeps its mass and still moves, but nothing can touch it. 1.7% of the seed-3
    population carries such a segment and design 5120 has NO fingertip collider
    on any active finger -- a hand that cannot grasp.

    Two regions are legitimate and must both stay reachable:
    [lo, MIN_SEGMENT_M) is a virtual link ("this finger has no metacarpal"), and
    [2r, hi] is a buildable capsule. A segment that was virtual stays virtual; a
    segment that was real is pushed UP to the first buildable length rather than
    collapsed into a virtual one, because deleting a link is a far larger change
    than the mutation intended.

    Note radius_scale MOVES this band, so this must run over every segment after
    a radius change -- including segments the operator did not touch.
    """
    from genmech.tools.build_hand_urdf import MIN_SEGMENT_M

    lo, hi = _TIER_RANGE[tier]
    min_len = 2.0 * anchors.TIER_RADIUS_M[tier] * radius_scale
    new = _reflect(new, lo, hi)
    if old < MIN_SEGMENT_M:                      # was virtual -> stays virtual
        return min(new, max(lo, MIN_SEGMENT_M * 0.5))
    if new < MIN_SEGMENT_M:                      # was real -> do not delete it
        return min(max(min_len, lo), hi)
    if new < min_len:                            # landed in the degenerate band
        return min(max(min_len, lo), hi)
    return new


def _finger_n_joints(f: P.FingerParams) -> int:
    return sum(f.enabled)


def _active_indices(hand: P.HandParams) -> list[int]:
    return [i for i, f in enumerate(hand.fingers) if f.active]


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

def op_palm(hand: P.HandParams, alpha: float, rng: random.Random) -> P.HandParams:
    """Resize the palm, then put every finger back on the new faces.

    Recomputing the mounts is not optional. face_frame reads the hand's own
    extents, so a mount left alone would float off a bigger palm or sink into a
    smaller one. Fingers keep their FRACTIONAL position on the face, so absolute
    mount separation scales with the palm -- which is what makes this the
    coherent-separation operator rather than a per-finger nudge.
    """
    extents = tuple(
        _reflect(e + rng.gauss(0.0, alpha * (hi - lo)), lo, hi)
        for e, (lo, hi) in zip(hand.palm_extents, _PALM_RANGES)
    )
    fingers = [
        replace(f, mount=P.mount_on_face(*f.mount_params, extents=extents))
        if (f.active and f.mount_params) else f
        for f in hand.fingers
    ]
    return replace(hand, fingers=tuple(fingers), palm_extents=extents)


def op_scale(hand: P.HandParams, alpha: float, rng: random.Random) -> P.HandParams:
    """Perturb every active finger's segment lengths and radius, multiplicatively.

    MULTIPLICATIVE, not a fraction of each tier's range. The ranges are wildly
    uneven -- mc spans 0-110 mm against dp's 16-36 mm -- so a range-fraction step
    puts ~83% of its variance into the metacarpal alone, and then behaves
    completely differently on fingers with fewer than MIN_JOINTS_FOR_METACARPAL
    joints, where mc is pinned at 0. Scaling by current value gives every tier a
    step proportional to its own size and makes the operator behave the same way
    on every finger.

    radius_scale is drawn FIRST because it defines the degenerate band that every
    length is then re-projected out of.
    """
    fingers = list(hand.fingers)
    for i in _active_indices(hand):
        f = fingers[i]
        rs = _reflect(f.radius_scale * (1.0 + rng.gauss(0.0, alpha)),
                      *P.RADIUS_SCALE_RANGE)

        def jitter(v: float, tier: str) -> float:
            return _reproject_length(v * (1.0 + rng.gauss(0.0, alpha)), v, tier, rs)

        # A metacarpal only exists once a CMC joint can move it. If this finger
        # has none, mc stays exactly 0 -- a nonzero length there duplicates the
        # mount and makes the two non-identifiable (MIN_JOINTS_FOR_METACARPAL).
        mc_len = f.mc.length
        if mc_len > 0.0:
            mc_len = jitter(mc_len, "mc")

        fingers[i] = replace(
            f,
            mc=P.Segment(xyz=(mc_len, 0.0, 0.0), rpy=f.mc.rpy),
            pp_length=jitter(f.pp_length, "pp"),
            mp_length=jitter(f.mp_length, "mp"),
            dp_length=jitter(f.dp_length, "dp"),
            radius_scale=rs,
        )
        # The radius moved, so lengths this operator did NOT resample may now sit
        # in the new degenerate band. Re-project them against the final radius.
        g = fingers[i]
        fingers[i] = replace(
            g,
            pp_length=_reproject_length(g.pp_length, f.pp_length, "pp", rs),
            mp_length=_reproject_length(g.mp_length, f.mp_length, "mp", rs),
            dp_length=_reproject_length(g.dp_length, f.dp_length, "dp", rs),
        )
    return replace(hand, fingers=tuple(fingers))


def op_mounting(hand: P.HandParams, alpha: float, rng: random.Random) -> P.HandParams:
    """Shift ONE finger's base position and lean on its own face.

    The face is held fixed -- this is a local move, not a relocation -- and roll
    is left alone because align_flexion_downward overwrites it (see module
    docstring).

    Position steps in metres and converts through the face's usable span, so the
    same alpha is the same physical shift on every face and every palm size.
    Lean is perturbed in the TANGENT PLANE: (tilt, azimuth) are polar, so
    Cartesian jitter gives a uniform angular step and removes the degeneracy at
    tilt = 0, where azimuth means nothing.
    """
    cand = [i for i in _active_indices(hand) if hand.fingers[i].mount_params]
    if not cand:
        raise MutationImpossible("no face-mounted active finger")
    i = rng.choice(cand)
    f = hand.fingers[i]
    face, u_frac, v_frac, roll, tilt, azimuth = f.mount_params

    _, _, _, _, (span_u, span_v) = P.face_frame(face, hand.palm_extents)
    usable_u = max(span_u - 2 * P.FACE_MARGIN, 0.0) / 2.0
    usable_v = max(span_v - 2 * P.FACE_MARGIN, 0.0) / 2.0
    sigma_m = alpha * MOUNT_POS_REF_M
    if usable_u > 1e-9:
        u_frac = _reflect(u_frac + rng.gauss(0.0, sigma_m) / usable_u, -1.0, 1.0)
    if usable_v > 1e-9:
        v_frac = _reflect(v_frac + rng.gauss(0.0, sigma_m) / usable_v, -1.0, 1.0)

    sigma_t = alpha * MOUNT_TILT_REF_RAD
    tx = tilt * math.cos(azimuth) + rng.gauss(0.0, sigma_t)
    ty = tilt * math.sin(azimuth) + rng.gauss(0.0, sigma_t)
    tilt = min(math.hypot(tx, ty), P.MOUNT_TILT_RANGE[1])
    azimuth = _wrap(math.atan2(ty, tx))

    mp = (face, u_frac, v_frac, _wrap(roll), tilt, azimuth)
    fingers = list(hand.fingers)
    fingers[i] = replace(
        f, mount=P.mount_on_face(*mp, extents=hand.palm_extents), mount_params=mp)
    return replace(hand, fingers=tuple(fingers))


def _op_num_joints(hand: P.HandParams, delta: int, rng: random.Random,
                   ) -> P.HandParams:
    """Move ONE finger up or down the ACTIVATION_ORDER ladder by one rung.

    Two things ride along and are silent if missed:

    NEWLY LIVE SLOTS NEED REAL TRAVEL. A slot SHARPA ghosts carries
    _GHOST_LIMIT = (0, 1e-8), so promoting it without fixing its limits yields a
    joint that exists, consumes an action dimension, and cannot move. sample()
    does this; so must this.

    THE 4<->5 BOUNDARY IS DISCONTINUOUS. MIN_JOINTS_FOR_METACARPAL = 5, so 4->5
    turns on CMC_FE *and* unlocks a metacarpal that has to be drawn, while 5->4
    forces it back to 0. Either direction moves reach by up to 110 mm alongside
    the joint change.
    """
    lo_j, hi_j = P.JOINTS_PER_FINGER_RANGE
    cand = [i for i in _active_indices(hand)
            if lo_j <= _finger_n_joints(hand.fingers[i]) + delta <= hi_j]
    if not cand:
        raise MutationImpossible(f"no active finger can move {delta:+d} on the ladder")
    i = rng.choice(cand)
    f = hand.fingers[i]
    n_new = _finger_n_joints(f) + delta
    enabled = P.enabled_for(n_new)

    limits = dict(zip(P.JOINT_SLOTS, f.limits))
    for slot, live in zip(P.JOINT_SLOTS, enabled):
        if live and (limits[slot][1] - limits[slot][0]) < 1e-6:
            limits[slot] = _LIM_THUMB[slot] if slot.startswith("CMC") else _LIM_FINGER[slot]

    mc_len = f.mc.length
    if n_new >= P.MIN_JOINTS_FOR_METACARPAL and mc_len <= 0.0:
        mc_len = P._u_segment(rng, P.MC_LENGTH_RANGE, "mc", f.radius_scale)
    elif n_new < P.MIN_JOINTS_FOR_METACARPAL:
        mc_len = 0.0

    fingers = list(hand.fingers)
    fingers[i] = replace(f, enabled=enabled, limits=_limits(limits),
                         mc=P.Segment(xyz=(mc_len, 0.0, 0.0), rpy=f.mc.rpy))
    return replace(hand, fingers=tuple(fingers))


def op_num_joints_up(hand, alpha, rng):    return _op_num_joints(hand, +1, rng)
def op_num_joints_down(hand, alpha, rng):  return _op_num_joints(hand, -1, rng)


_OPS = {"palm": op_palm, "scale": op_scale, "mounting": op_mounting,
        "num_joints_up": op_num_joints_up, "num_joints_down": op_num_joints_down}


# ---------------------------------------------------------------------------
# phenotype, for comparing operators in a common currency
# ---------------------------------------------------------------------------

def keepout_clamped(hand: P.HandParams) -> int:
    """Active face-mounted fingers sitting exactly on the wrist keep-out plane.

    mount_on_face lifts any side-face mount below WRIST_KEEPOUT_M up to it, so in
    that band different v_frac values map to the SAME mount and part of a
    mounting step is silently absorbed. Measured at 22/60 on seed-3 parents, so
    it is common enough that the analysis has to be able to condition on it.
    """
    n = 0
    for f in hand.fingers:
        if not (f.active and f.mount_params):
            continue
        if f.mount_params[0][1] != "z" and abs(f.mount.xyz[2] - P.WRIST_KEEPOUT_M) < 1e-9:
            n += 1
    return n


def phenotype(hand: P.HandParams) -> dict:
    """The quantities the population study found predictive, plus palm volume.

    alpha is normalised in PARAMETER space, so it is not comparable across
    operators: the same alpha moves reach by one amount and mount separation by
    another. Recording realised displacement puts every operator on one axis --
    fitness change per millimetre of phenotype moved.
    """
    act = hand.active_fingers
    # FingerParams.reach() sums mount.length + cmc + mc + pp + mp + dp, so it
    # conflates two mechanically distinct things: WHERE the finger attaches on
    # the palm, and HOW LONG the finger is. They are also cleanly separated by
    # operator -- `palm` moves the mount by ~3 mm and the finger by exactly 0,
    # `scale` the reverse -- so a single "reach" number attributes palm mutations
    # to a finger-length change that never happened. Measured separately here;
    # `finger_length` (extension beyond the palm) is the one that carries the
    # measured fitness effect.
    reaches = [f.reach() for f in act]
    mounts_len = [f.mount.length for f in act]
    finger_len = [f.reach() - f.mount.length for f in act]
    mounts = [f.mount.xyz for f in act]
    seps = [math.dist(a, b) for i, a in enumerate(mounts) for b in mounts[i + 1:]]
    return {
        "mean_reach": sum(reaches) / len(reaches) if reaches else 0.0,
        "mean_finger_length": sum(finger_len) / len(finger_len) if finger_len else 0.0,
        "mean_mount_offset": sum(mounts_len) / len(mounts_len) if mounts_len else 0.0,
        "min_separation": min(seps) if seps else 0.0,
        "mean_separation": sum(seps) / len(seps) if seps else 0.0,
        "palm_volume": hand.palm_extents[0] * hand.palm_extents[1] * hand.palm_extents[2],
        "n_active_joints": hand.n_active_joints,
        "n_active_fingers": hand.n_active_fingers,
    }


def _displacement(parent: P.HandParams, child: P.HandParams) -> dict:
    a, b = phenotype(parent), phenotype(child)
    return {f"d_{k}": b[k] - a[k] for k in a}


# ---------------------------------------------------------------------------
# one child
# ---------------------------------------------------------------------------

def mutate_one(parent: P.HandParams, operator: str, alpha: float,
               rng: random.Random, *, name: str, gate: str = "analytic",
               align: bool = True, max_tries: int = 40,
               tmpdir: Path | None = None) -> tuple[P.HandParams, bool, int]:
    """Return ``(child, mutation_failed, attempts)``.

    On exhaustion the PARENT is returned with ``mutation_failed=True`` rather
    than a hole, because child i must stay the mutant of parent i for the paired
    delta to hold. Those children must then be excluded from the operator's
    statistics -- counting them would drag every low-acceptance operator toward
    its parents' scores and make it look harmless rather than infeasible.

    The accept pipeline mirrors build_population exactly: validate, self-collision
    gate, then flexion alignment with a re-check that reverts the alignment if the
    new roll made the hand overlap itself.
    """
    from genmech.robots.generated.flexion import align_flexion_downward
    from genmech.tools.build_hand_urdf import write_urdf

    fn = _OPS[operator]
    attempt = 0
    for attempt in range(1, max_tries + 1):
        try:
            child = replace(fn(parent, alpha, rng), name=name)
            P.validate(child)
        except MutationImpossible:
            # Deterministic: no finger CAN move this way (every one is already at
            # a ladder end). Retrying redraws nothing that would change that, so
            # 40 attempts would just burn the gate 40 times.
            break
        except (P.InvalidHand, ValueError):
            continue
        if _hits(child, gate):
            continue
        if align:
            # curl_directions parses a URDF off disk. Write it to a temp file:
            # the canonical path is a flat directory already holding ~50k files,
            # and nothing downstream reads a mutant's URDF anyway.
            td = tmpdir or Path(tempfile.gettempdir())
            up = td / f"{name}.urdf"
            up.parent.mkdir(parents=True, exist_ok=True)
            write_urdf(child, up)
            aligned = align_flexion_downward(child, urdf_path=up)
            if aligned is not child and not _hits(aligned, gate):
                child = aligned
            up.unlink(missing_ok=True)
        if not _roundtrip_ok(child):
            raise RuntimeError(f"{name}: parameters do not survive a JSON round-trip")
        return child, False, attempt
    # The REAL attempt count, not max_tries: a MutationImpossible break costs one
    # gate call and a collision-bound operator costs forty, and telling them
    # apart is the point of recording it.
    return replace(parent, name=name), True, attempt


# ---------------------------------------------------------------------------
# a whole operator manifest
# ---------------------------------------------------------------------------

def mutate_population(parents: list[P.HandParams], operator: str, alpha: float,
                      *, seed: int, prefix: str, gate: str = "analytic",
                      align: bool = True, max_tries: int = 40,
                      lo: int = 0, hi: int | None = None,
                      progress_every: int = 500) -> dict:
    """Mutate ``parents[lo:hi]`` and return a manifest dict.

    Each child's RNG is seeded from (seed, operator, parent index), NOT from a
    stream walked in order, so a shard produces byte-identical children whatever
    slice it is given and shards can be merged without the caveat that dogs the
    sharded population build (see 00_build_population_sharded.sub).
    """
    hi = len(parents) if hi is None else hi
    entries, failed, attempts_total = [], 0, 0
    with tempfile.TemporaryDirectory(prefix="genmech_mutate_") as td:
        for i in range(lo, hi):
            parent = parents[i]
            rng = random.Random(f"{seed}:{operator}:{i}")
            name = f"{prefix}_{i:05d}"
            child, bad, tries = mutate_one(
                parent, operator, alpha, rng, name=name, gate=gate,
                align=align, max_tries=max_tries, tmpdir=Path(td))
            failed += bad
            attempts_total += tries
            entries.append({
                "name": name,
                "params": hand_to_json(child),
                "parent": parent.name,
                "parent_index": i,
                "operator": operator,
                "alpha": alpha,
                "mutation_failed": bad,
                "attempts": tries,
                "displacement": _displacement(parent, child),
                "keepout_clamped": keepout_clamped(child),
            })
            if progress_every and (i - lo + 1) % progress_every == 0:
                print(f"[mutate] {operator} {i - lo + 1}/{hi - lo}  "
                      f"failed {failed}  mean tries {attempts_total / (i - lo + 1):.2f}",
                      flush=True)
    return {
        "version": 1,
        "kind": "mutant_population",
        "operator": operator,
        "alpha": alpha,
        "seed": seed,
        "gate": gate,
        "align_flexion": align,
        "max_tries": max_tries,
        "count": len(entries),
        "lo": lo, "hi": hi,
        "n_mutation_failed": failed,
        "mean_attempts": attempts_total / max(1, len(entries)),
        "hands": entries,
    }


def merge_shards(out_dir: Path, num_shards: int) -> dict:
    """Concatenate shard files in index order into manifest.json.

    Strict about order for the same reason merge_population_shards is: env i
    holds design i, so a mis-ordered merge would break the pairing this whole
    layout exists to provide.
    """
    entries, meta = [], None
    for s in range(num_shards):
        f = out_dir / f"shard_{s:03d}.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing; refusing to merge a population with holes")
        d = json.loads(f.read_text(encoding="utf-8"))
        meta = meta or d
        entries.extend(d["hands"])
    idx = [e["parent_index"] for e in entries]
    if idx != list(range(len(entries))):
        raise RuntimeError("shards do not concatenate to a contiguous parent index range")
    out = {**{k: v for k, v in meta.items() if k != "hands"},
           "count": len(entries), "lo": 0, "hi": len(entries),
           "n_mutation_failed": sum(e["mutation_failed"] for e in entries),
           "mean_attempts": sum(e["attempts"] for e in entries) / max(1, len(entries)),
           "num_shards": num_shards, "hands": entries}
    (out_dir / "manifest.json").write_text(json.dumps(out), encoding="utf-8")
    return out


def load_parents(manifest: Path) -> list[P.HandParams]:
    d = json.loads(Path(manifest).read_text(encoding="utf-8"))
    return [hand_from_json(e["params"]) for e in d["hands"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--parents", required=True, help="Parent manifest.json")
    p.add_argument("--operator", required=True, choices=OPERATORS + ("merge",))
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--out", required=True, help="Output directory for this operator")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--prefix", default=None, help="Child name prefix; default mut_<operator>")
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--no_align", action="store_true")
    p.add_argument("--max_tries", type=int, default=40)
    a = p.parse_args()

    out_dir = resolve_repo_path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.operator == "merge":
        m = merge_shards(out_dir, a.num_shards)
        print(f"[mutate] merged {m['count']} children; "
              f"{m['n_mutation_failed']} mutation_failed; "
              f"mean attempts {m['mean_attempts']:.2f}")
        return

    parents = load_parents(resolve_repo_path(a.parents))
    n = len(parents)
    lo, hi = 0, n
    if a.shard is not None:
        per = (n + a.num_shards - 1) // a.num_shards
        lo, hi = a.shard * per, min(n, (a.shard + 1) * per)
    prefix = a.prefix or f"mut_{a.operator}"
    print(f"[mutate] {a.operator} alpha={a.alpha} parents[{lo}:{hi}] of {n}", flush=True)

    man = mutate_population(parents, a.operator, a.alpha, seed=a.seed,
                            prefix=prefix, align=not a.no_align,
                            max_tries=a.max_tries, lo=lo, hi=hi)
    name = "manifest.json" if a.shard is None else f"shard_{a.shard:03d}.json"
    (out_dir / name).write_text(json.dumps(man), encoding="utf-8")
    print(f"[mutate] wrote {out_dir / name}: {man['count']} children, "
          f"{man['n_mutation_failed']} failed, "
          f"mean attempts {man['mean_attempts']:.2f}")


if __name__ == "__main__":
    main()
