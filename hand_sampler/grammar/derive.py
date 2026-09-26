"""Sampling and (pure, deterministic) derivation for the hand-kinematics
grammar (iteration 3).

``sample_derivation`` is the only place randomness happens; it produces a
``Derivation`` -- a flat, ordered trace of every production applied and
every parameter value sampled. ``derive`` is pure (no RNG) and rebuilds the
identical ``KinematicModel`` from that trace, so ``replay`` (calling
``derive`` twice on the same derivation, or deriving two derivations
sampled from the same seed) is exact.

No production here ever accepts a raw model or URDF: models are built only
through ``derive`` walking productions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .coords import CONTINUOUS_SAMPLE_RANGE
from .distributions import (
    ANGLE_STEP_DEG,
    DEFAULT_DISTRIBUTION,
    DEG,
    Distribution,
    N_ANGLE_STEPS,
    N_ELEVATION_STEPS,
    sample_axis,
    sample_bend,
    sample_capsule_radius_m,
    sample_grid_angle_rad,
    sample_grid_length_m,
    sample_module,
    sample_palm_joint_limits_rad,
    sample_revolute_limits_rad,
)
from .fk import forward_kinematics
from .kinematics import (
    AffineCoupling,
    Body,
    Frame,
    Joint,
    KinematicModel,
    ModelError,
    Pose,
    validate,
)
from .rules import GRAMMAR_VERSION

DERIVATION_SCHEMA = "hand_grammar_derivation/0.1"


class VariationImpossible(Exception):
    """Raised by ``vary`` when the requested operator has no valid
    application after 32 attempts (e.g. ``remove_digit`` at 1 digit)."""


class DerivationError(ModelError):
    """Raised by ``derive`` when ``validate_derivation`` finds the
    *derivation itself* malformed -- before any attempt is made to build a
    ``KinematicModel`` from it. A ``ModelError`` subclass (same ``issues``
    list interface), so every existing ``except ModelError`` catch site
    (e.g. ``vary``'s own retry loop) already handles it."""


@dataclass(frozen=True)
class DerivationStep:
    path: str
    production: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class Derivation:
    seed: int
    grammar_version: str
    steps: Tuple[DerivationStep, ...]
    # Provenance of a ``vary``-produced derivation: an ordered tuple of
    # (operator, parent_seed) entries, one per ``vary`` application, oldest
    # first. Empty for a freshly sampled derivation. ``seed`` itself is
    # always the *founder* seed (the seed originally passed to
    # ``sample_derivation``) and is never changed by ``vary`` -- a varied
    # derivation is replayed by re-running its own stored ``steps`` through
    # ``derive``, never by resampling from ``seed`` again, so ``lineage`` is
    # the only record of which operators were applied and in what order;
    # ``seed`` alone does not recover it.
    lineage: Tuple[Tuple[str, int], ...] = ()


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def _coerce_rng(rng_or_seed):
    if isinstance(rng_or_seed, np.random.Generator):
        return -1, rng_or_seed
    seed = int(rng_or_seed)
    return seed, np.random.default_rng(seed)


def _mount_bodies_from_steps(steps) -> List[str]:
    hand = next(s for s in steps if s.path == "hand")
    n = hand.params["palm_body_count"]
    return ["root"] + [f"palm{i}" for i in range(n)]


def _step_digit_id(st: "DerivationStep") -> Optional[str]:
    if st.production in ("Digit", "Phalanx"):
        return st.params["digit_id"]
    return None


def _is_descendant_digit(host_id: str, candidate_id: Optional[str]) -> bool:
    """True if ``candidate_id`` is ``host_id`` itself or a branch digit nested
    (at any depth) under one of ``host_id``'s phalanges. Branch ids are always
    ``f"{parent_id}p{phalanx}b{slot}"``, so a ``host_id + "p"`` prefix match is
    exact: top-level ids are plain integers and never contain ``"p"``, so they
    can never collide with this pattern."""
    if candidate_id is None:
        return False
    return candidate_id == host_id or candidate_id.startswith(host_id + "p")


def _revolute_source_indices(steps, digit_id: str, upto: int) -> Tuple[int, ...]:
    """0-based indices < ``upto`` of ``digit_id``'s own Phalanx steps (already
    present in ``steps``) whose module is revolute ("R") -- the only valid
    ``Coupled`` source per the coupling-source rule (see
    ``distributions.sample_module``)."""
    return tuple(
        sorted(
            s.params["p"] for s in steps
            if s.production == "Phalanx" and s.params["digit_id"] == digit_id
            and s.params["p"] < upto and s.params["module"]["kind"] == "R"
        )
    )


def _max_uid(steps: Sequence[DerivationStep]) -> int:
    """Highest ``uid`` (I15 fix 1) already stamped on any step's ``params``
    (-1 if none). ``uid`` lives INSIDE ``params`` (not as a separate
    ``DerivationStep`` field) so every existing in-place-edit operator
    (``p = dict(s.params); p[...] = ...; DerivationStep(..., params=p)``)
    already preserves it for free via the ``dict(s.params)`` copy -- only
    the handful of sites that create a genuinely NEW step need to consult
    this to allocate a fresh one."""
    best = -1
    for s in steps:
        u = s.params.get("uid")
        if isinstance(u, int) and u > best:
            best = u
    return best


def _sample_phalanx(rng, dist: Distribution, steps: List[DerivationStep], digit_id: str, p: int,
                     depth: int, is_last: bool, next_uid: List[int]) -> None:
    module = sample_module(rng, dist, p, _revolute_source_indices(steps, digit_id, p))
    length = sample_grid_length_m(rng, dist.link_length_range_m, dist.link_length_grid_m)
    bend_rpy, bend_offset = sample_bend(rng, dist)
    branch_digit_count = 0
    if depth < dist.max_branch_depth and float(rng.random()) < dist.branch_probability:
        # A phalanx's body must end up with >= 2 child joints for this to be
        # real branching (measured structurally, see the support-audit /
        # structural-validity assertions). A non-last phalanx already gets a
        # "next phalanx" child, so 1 branch digit suffices; the *last*
        # phalanx in a digit has no next-phalanx child, so it needs >= 2
        # branch digits on its own (skipped entirely if the distribution's
        # ``max_branch_digits`` can't reach 2).
        min_branches = 2 if is_last else 1
        if min_branches <= dist.max_branch_digits:
            branch_digit_count = int(rng.integers(min_branches, dist.max_branch_digits + 1))
    uid = next_uid[0]
    next_uid[0] += 1
    steps.append(DerivationStep(path=f"digit/{digit_id}/phalanx/{p}", production="Phalanx", params={
        "digit_id": digit_id, "p": p, "module": module, "length": length,
        "branch_digit_count": branch_digit_count, "uid": uid,
        "bend_rpy": bend_rpy, "bend_offset": bend_offset,
    }))
    if branch_digit_count:
        # The branch mounts on THIS phalanx's own body (distal to its
        # joint), never on a palm body, so that body genuinely gets >= 2
        # child joints (its own next phalanx plus one per branch digit).
        phalanx_body = f"d{digit_id}p{p + 1}"
        for b in range(branch_digit_count):
            sub_id = f"{digit_id}p{p + 1}b{b}"
            _emit_digit(rng, dist, steps, sub_id, [phalanx_body], top_level=False, depth=depth + 1,
                        next_uid=next_uid)


def _emit_digit(rng, dist: Distribution, steps: List[DerivationStep], digit_id: str,
                 mount_bodies: List[str], top_level: bool, depth: int, next_uid: List[int]) -> None:
    mount = mount_bodies[int(rng.integers(0, len(mount_bodies)))]
    mount_frac = float(dist.mount_frac_choices[int(rng.integers(0, len(dist.mount_frac_choices)))])
    mount_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
    phalanx_count = int(rng.integers(dist.phalanx_count_range[0], dist.phalanx_count_range[1] + 1))
    uid = next_uid[0]
    next_uid[0] += 1
    steps.append(DerivationStep(path=f"digit/{digit_id}", production="Digit", params={
        "digit_id": digit_id, "mount": mount, "mount_frac": mount_frac, "mount_rpy": mount_rpy,
        "phalanx_count": phalanx_count, "top_level": top_level, "depth": depth, "uid": uid,
    }))
    for p in range(phalanx_count):
        _sample_phalanx(rng, dist, steps, digit_id, p, depth, is_last=(p == phalanx_count - 1), next_uid=next_uid)


def _sample_digit(rng, dist: Distribution, steps: List[DerivationStep], next_id: List[int],
                   mount_bodies: List[str], next_uid: List[int]) -> None:
    """Sample a fresh *top-level* digit (id is the next 1-based integer)."""
    digit_id = str(next_id[0])
    next_id[0] += 1
    _emit_digit(rng, dist, steps, digit_id, mount_bodies, top_level=True, depth=0, next_uid=next_uid)


def sample_derivation(rng_or_seed, dist: Distribution = DEFAULT_DISTRIBUTION) -> Derivation:
    seed, rng = _coerce_rng(rng_or_seed)
    steps: List[DerivationStep] = []

    digit_count = int(rng.integers(dist.digit_count_range[0], dist.digit_count_range[1] + 1))
    # Additional palm bodies beyond the root -- the root is always a palm
    # body with a real segment of its own (root_length below), so a hand
    # never lacks a palm even when palm_body_count == 0.
    palm_body_count = int(rng.integers(dist.palm_body_count_range[0], dist.palm_body_count_range[1] + 1))
    root_length = sample_grid_length_m(rng, dist.palm_length_range_m, dist.link_length_grid_m)
    # One capsule radius per hand (iteration 7 / M2, geometry overlay -- see
    # rules.py's module note): stamped onto every Body.radius by ``derive``.
    capsule_radius_m = sample_capsule_radius_m(rng, dist)
    steps.append(DerivationStep(path="hand", production="Hand", params={
        "digit_count": digit_count, "palm_body_count": palm_body_count, "root_length": root_length,
        "capsule_radius_m": capsule_radius_m,
    }))

    # I15 fix 1: a stable ``uid`` counter, shared across every PalmBody/
    # Digit/Phalanx step this derivation creates (never resets, never
    # reused) -- see ``_max_uid``'s docstring and ``joint_identity`` below.
    next_uid: List[int] = [0]

    palm_names: List[str] = []
    for i in range(palm_body_count):
        name = f"palm{i}"
        parent_choices = ["root"] + palm_names
        parent = parent_choices[int(rng.integers(0, len(parent_choices)))]
        mount_frac = float(dist.mount_frac_choices[int(rng.integers(0, len(dist.mount_frac_choices)))])
        length = sample_grid_length_m(rng, dist.palm_length_range_m, dist.link_length_grid_m)
        direction_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
        has_joint = bool(float(rng.random()) < dist.palm_joint_probability)
        axis = sample_axis(rng)
        limits: Optional[Tuple[float, float]] = None
        if has_joint:
            limits = sample_palm_joint_limits_rad(rng, dist)
        uid = next_uid[0]
        next_uid[0] += 1
        steps.append(DerivationStep(path=f"palm/{i}", production="PalmBody", params={
            "name": name, "parent": parent, "mount_frac": mount_frac, "length": length,
            "direction_rpy": direction_rpy, "has_joint": has_joint, "axis": axis, "limits": limits,
            "uid": uid,
        }))
        palm_names.append(name)

    mount_bodies = ["root"] + palm_names
    next_id = [1]
    for _ in range(digit_count):
        _sample_digit(rng, dist, steps, next_id, mount_bodies, next_uid)

    return Derivation(seed=seed, grammar_version=GRAMMAR_VERSION, steps=tuple(steps))


# --------------------------------------------------------------------------
# derive: pure, deterministic reconstruction
# --------------------------------------------------------------------------


def validate_derivation(derivation: Derivation) -> List[str]:
    """Structural validation of a ``Derivation`` *as a trace* -- independent
    of, and prior to, building any ``KinematicModel`` from it. Returns a list
    of issue strings (empty if clean). Checks:

    - ``grammar_version`` matches the current ``GRAMMAR_VERSION``.
    - every ``Phalanx`` step's ``digit_id`` names a ``Digit`` step present in
      the same derivation.
    - the ``hand`` step's ``digit_count`` equals the number of top-level
      ``Digit`` steps.
    - each ``Digit`` step's ``Phalanx`` steps have exactly the indices
      ``0..phalanx_count - 1`` (contiguous, no gaps or duplicates).
    - every ``Digit`` step's ``mount`` names a body that another step in the
      derivation actually creates (``"root"``, a ``PalmBody`` step's
      ``name``, or some digit's own phalanx body ``f"d{digit_id}p{p+1}"``).
    """
    issues: List[str] = []

    if derivation.grammar_version != GRAMMAR_VERSION:
        issues.append(
            f"grammar_version mismatch: derivation has {derivation.grammar_version!r}, "
            f"expected {GRAMMAR_VERSION!r}"
        )

    hand_steps = [s for s in derivation.steps if s.path == "hand"]
    hand = hand_steps[0].params if len(hand_steps) == 1 else None
    if len(hand_steps) != 1:
        issues.append(f"expected exactly one 'hand' step, found {len(hand_steps)}")

    digit_steps: Dict[str, DerivationStep] = {}
    for s in derivation.steps:
        if s.production == "Digit":
            digit_id = s.params["digit_id"]
            if digit_id in digit_steps:
                issues.append(f"duplicate Digit step for digit id {digit_id!r}")
            digit_steps[digit_id] = s

    phalanx_by_digit: Dict[str, Dict[int, DerivationStep]] = {}
    for s in derivation.steps:
        if s.production != "Phalanx":
            continue
        digit_id = s.params["digit_id"]
        if digit_id not in digit_steps:
            issues.append(f"Phalanx step {s.path!r} references unknown digit id {digit_id!r}")
            continue
        by_p = phalanx_by_digit.setdefault(digit_id, {})
        p = s.params["p"]
        if p in by_p:
            issues.append(f"duplicate Phalanx step for digit {digit_id!r} index {p}")
        by_p[p] = s

    if hand is not None:
        n_top_level = sum(1 for s in digit_steps.values() if s.params.get("top_level"))
        if n_top_level != hand["digit_count"]:
            issues.append(
                f"hand digit_count={hand['digit_count']} disagrees with "
                f"{n_top_level} top-level Digit step(s)"
            )

    for digit_id, dstep in digit_steps.items():
        expected = list(range(dstep.params["phalanx_count"]))
        got = sorted(phalanx_by_digit.get(digit_id, {}))
        if got != expected:
            issues.append(
                f"digit {digit_id!r} phalanx indices {got} are not contiguous 0..{dstep.params['phalanx_count'] - 1}"
            )

    known_bodies = {"root"}
    for s in derivation.steps:
        if s.production == "PalmBody":
            known_bodies.add(s.params["name"])
    for digit_id, by_p in phalanx_by_digit.items():
        for p in by_p:
            known_bodies.add(f"d{digit_id}p{p + 1}")

    for digit_id, dstep in digit_steps.items():
        mount = dstep.params["mount"]
        if mount not in known_bodies:
            issues.append(f"digit {digit_id!r} mount {mount!r} is not a body any step creates")

    return issues


def _compose_bend_rpy(existing_rpy: Tuple[float, float, float],
                       bend_rpy: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Grammar 0.5's rest-bend primitive (I16 priority 1): combine a
    phalanx joint's existing origin orientation (the digit's own sampled
    ``mount_rpy`` for its first phalanx, or the identity ``(0,0,0)`` for a
    mid-digit continuation joint -- see rules.py's convention note) with the
    small additional ``bend_rpy`` perturbation.

    Composition order/method: plain COMPONENTWISE Euler-angle addition
    (roll+roll, pitch+pitch, yaw+yaw) -- deliberately NOT a rotation-matrix
    product (which would require decomposing the product back into a single
    fixed-axis-XYZ triple, a lossy, gimbal-lock-prone operation for no
    benefit here). This choice is exact and FK-correct regardless: fk.py's
    ``rpy_to_matrix`` (and, identically, the URDF/Pinocchio convention
    ``to_urdf`` exports to) only ever consumes ONE fixed-axis-XYZ triple per
    joint origin -- the componentwise sum below IS that triple, so whatever
    rotation it represents is exactly what gets built and exactly what any
    FK oracle sees, with no separate decomposition step to get wrong. For a
    continuation joint (``existing_rpy == (0,0,0)``) this reduces to
    ``bend_rpy`` exactly, which is what ``coverage.py``'s bend-grid check
    judges against ``Distribution.bend_rpy_choices_rad``."""
    return tuple(float(a) + float(b) for a, b in zip(existing_rpy, bend_rpy))


def derive(derivation: Derivation) -> KinematicModel:
    issues = validate_derivation(derivation)
    if issues:
        raise DerivationError(issues)

    steps_by_path = {s.path: s for s in derivation.steps}
    hand = steps_by_path["hand"].params
    palm_body_count = hand["palm_body_count"]
    root_length = hand["root_length"]
    # One scalar capsule radius for the whole hand (see rules.py's module
    # note / distributions.py's capsule_radius_choices_m); stamped onto every
    # Body below -- geometry itself is never derived here (see geometry.py),
    # only this one per-hand parameter that geometry.py later reads off
    # Body.radius.
    capsule_radius_m = hand["capsule_radius_m"]

    # The root always owns a real segment (see rules.py's RootProduction /
    # convention-change note): a hand always has a palm, so body_length for
    # "root" is never 0, and the root gets its own "<body>_tip" frame just
    # like every other segment-owning body.
    bodies: List[Body] = [Body(name="root", palm=True, radius=capsule_radius_m)]
    joints: List[Joint] = []
    frames: List[Frame] = [Frame(name="root_tip", body="root", pose=Pose(xyz=(0.0, 0.0, root_length)))]
    couplings: List[AffineCoupling] = []
    body_length: Dict[str, float] = {"root": root_length}
    joints_by_name: Dict[str, Joint] = {}

    for i in range(palm_body_count):
        p = steps_by_path[f"palm/{i}"].params
        name = p["name"]
        parent = p["parent"]
        length = p["length"]
        rpy = tuple(p["direction_rpy"])
        # Mount point ON the parent's own segment: T = Trans(0,0,frac*L) *
        # Rot(rpy) -- xyz is the (unrotated) translation along the parent's
        # own z-axis, rpy is the child frame's orientation relative to the
        # parent, exactly as a URDF joint origin means (see rules.py's
        # convention-change note / fk.py's docstring). Never place the
        # child a further ``length`` out from that mount point -- that was
        # the old bug that left the parent's segment with an unowned
        # stretch between the mount and the parent's own tip.
        mount_len = body_length[parent]
        base_xyz = (0.0, 0.0, p["mount_frac"] * mount_len)
        jtype = "revolute" if p["has_joint"] else "fixed"
        axis = tuple(p["axis"]) if p["has_joint"] else (1.0, 0.0, 0.0)
        limits = tuple(p["limits"]) if p["has_joint"] else None
        j = Joint(
            name=f"{name}_j", type=jtype, parent=parent, child=name,
            origin=Pose(xyz=base_xyz, rpy=rpy),
            axis=axis, limits=limits,
        )
        joints.append(j)
        joints_by_name[j.name] = j
        bodies.append(Body(name=name, palm=True, radius=capsule_radius_m))
        frames.append(Frame(name=f"{name}_tip", body=name, pose=Pose(xyz=(0.0, 0.0, length))))
        body_length[name] = length

    def _process_digit(p: Dict[str, Any]) -> None:
        digit_id = p["digit_id"]
        mount = p["mount"]
        # Mount point ON the host segment (see the palm-mount comment above
        # -- the same convention, applied to a digit/branch mounting on a
        # palm or phalanx body): xyz is the unrotated translation along the
        # host's own z-axis, mount_rpy is the child frame's orientation.
        mount_len = body_length.get(mount, 0.0)
        base_xyz = (0.0, 0.0, p["mount_frac"] * mount_len)
        base_rpy = tuple(p["mount_rpy"])

        prev_body = mount
        prev_len = 0.0
        phalanx_joint_names: Dict[int, str] = {}
        for pi in range(p["phalanx_count"]):
            pp = steps_by_path[f"digit/{digit_id}/phalanx/{pi}"].params
            body_name = f"d{digit_id}p{pi + 1}"
            joint_name = f"{body_name}_j"
            # Grammar 0.5 rest-bend primitive (I16 priority 1): ``bend_rpy``/
            # ``bend_offset`` default to (0,0,0)/(0,0) via ``.get`` for
            # backward compatibility with any hand-authored Phalanx params
            # dict predating this field (see distributions.Distribution
            # .bend_probability's docstring). xyz becomes
            # (bend_offset_x, bend_offset_y, t) -- t as today (the mount
            # fraction along the host segment for pi==0, or the previous
            # phalanx's own length for a continuation) -- since the existing
            # convention's own x/y are always exactly 0 in both cases.
            # rpy is composed with the existing convention's own rpy (the
            # digit's sampled ``mount_rpy`` for pi==0, or identity for a
            # continuation) by plain componentwise Euler-angle addition, NOT
            # a rotation-matrix product -- see ``_compose_bend_rpy``'s own
            # docstring for why this is FK-exact regardless.
            bend_rpy = tuple(pp.get("bend_rpy", (0.0, 0.0, 0.0)))
            bend_offset = tuple(pp.get("bend_offset", (0.0, 0.0)))
            if pi == 0:
                origin_xyz = (bend_offset[0], bend_offset[1], base_xyz[2])
                origin_rpy = _compose_bend_rpy(base_rpy, bend_rpy)
            else:
                origin_xyz = (bend_offset[0], bend_offset[1], prev_len)
                origin_rpy = _compose_bend_rpy((0.0, 0.0, 0.0), bend_rpy)
            origin = Pose(xyz=origin_xyz, rpy=origin_rpy)

            mod = pp["module"]
            axis = tuple(mod["axis"])
            coupling = None
            if mod["kind"] == "R":
                jtype = "revolute"
                limits = tuple(mod["limits"])
            elif mod["kind"] == "C":
                jtype = "continuous"
                limits = None
            elif mod["kind"] == "P":
                jtype = "prismatic"
                limits = tuple(mod["limits"])
            elif mod["kind"] == "Coupled":
                jtype = "revolute"
                src_name = phalanx_joint_names[mod["source_p"]]
                src_joint = joints_by_name[src_name]
                if src_joint.type == "continuous":
                    slo, shi = CONTINUOUS_SAMPLE_RANGE
                else:
                    slo, shi = src_joint.limits
                mult = mod["multiplier"]
                off_c = mod["offset"]
                lo, hi = mult * slo + off_c, mult * shi + off_c
                if lo > hi:
                    lo, hi = hi, lo
                limits = (lo, hi)
                coupling = AffineCoupling(dependent=joint_name, source=src_name, multiplier=mult, offset=off_c)
            else:
                raise ModelError([f"unknown module kind {mod['kind']!r}"])

            j = Joint(name=joint_name, type=jtype, parent=prev_body, child=body_name,
                      origin=origin, axis=axis, limits=limits)
            joints.append(j)
            joints_by_name[joint_name] = j
            if coupling is not None:
                couplings.append(coupling)
            bodies.append(Body(name=body_name, radius=capsule_radius_m))
            length = pp["length"]
            frames.append(Frame(name=f"{body_name}_tip", body=body_name, pose=Pose(xyz=(0.0, 0.0, length))))
            body_length[body_name] = length
            phalanx_joint_names[pi] = joint_name
            prev_body = body_name
            prev_len = length

    # Digits form a DAG by mount (top-level digits mount on root/palm bodies,
    # always available; branch digits mount on a phalanx body created by
    # processing their host digit) -- process in that dependency order (a
    # simple fixed-point/BFS-by-depth over ``derivation.steps``' own order)
    # rather than assuming any particular position in the flat step list, so
    # ``vary``'s list-splicing operators can never silently reorder a branch
    # ahead of the phalanx body it mounts on.
    pending = [s.params for s in derivation.steps if s.production == "Digit"]
    while pending:
        ready = [p for p in pending if p["mount"] in body_length]
        if not ready:
            raise ModelError([
                f"digit {p['digit_id']!r} mount {p['mount']!r} is never created"
                for p in pending
            ])
        still_pending = [p for p in pending if p["mount"] not in body_length]
        for p in ready:
            _process_digit(p)
        pending = still_pending

    model = KinematicModel(
        name="grammar_hand", root="root", bodies=tuple(bodies), joints=tuple(joints),
        frames=tuple(frames), couplings=tuple(couplings),
    )
    validate(model)
    return model


def joint_identity(derivation: Derivation) -> Dict[str, int]:
    """I15 fix 1: ``{joint_name: uid}`` for every joint ``derive(derivation)``
    would produce, where ``uid`` is the stable identifier of the
    ``Phalanx``/``PalmBody`` step that CREATED that joint (see
    ``_max_uid``'s docstring). ``derive`` itself keeps joint NAMES
    positional/renumbering-sensitive (so replay hashes and URDF names are
    unchanged) -- this function is the alignment key callers (``phenodist.
    phenotype_distance``, ``e1_locality``) should use instead of a bare name
    match whenever they have a parent/child pair descended from a common
    ancestor derivation, since insert/delete_phalanx, remove_digit and
    remove_palm_body can renumber a joint's NAME (``d{digit_id}p{p+1}_j``)
    without that joint being a new piece of structure at all. A joint whose
    creating step predates this fix (no ``"uid"`` key in its params, e.g. a
    derivation produced by an older ``sample_derivation``) maps to ``-1``."""
    out: Dict[str, int] = {}
    for s in derivation.steps:
        if s.production == "PalmBody":
            out[f"{s.params['name']}_j"] = int(s.params.get("uid", -1))
        elif s.production == "Phalanx":
            out[f"d{s.params['digit_id']}p{s.params['p'] + 1}_j"] = int(s.params.get("uid", -1))
    return out


def segment(model: KinematicModel, body: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return ``(start, end)``: the endpoints, in the root frame at q=0, of
    ``body``'s own geometric segment. ``start`` is ``body``'s own origin;
    ``end`` is its ``"<body>_tip"`` frame (every body the grammar produces
    carries one -- see rules.py). This is a thin q=0 forward-kinematics
    query; it lives here (rather than in coverage.py) because it is about
    the derived model's geometry itself, not about judging that geometry
    against a ``Distribution``."""
    transforms = forward_kinematics(model, {})
    start = tuple(float(v) for v in transforms[body][:3, 3])
    end = tuple(float(v) for v in transforms[f"{body}_tip"][:3, 3])
    return start, end


def generate(seed, dist: Distribution = DEFAULT_DISTRIBUTION) -> Tuple[Derivation, KinematicModel]:
    derivation = sample_derivation(seed, dist)
    model = derive(derivation)
    return derivation, model


# --------------------------------------------------------------------------
# JSON round trip for Derivation
# --------------------------------------------------------------------------


def _encode(v: Any) -> Any:
    if isinstance(v, tuple):
        return {"__tuple__": [_encode(x) for x in v]}
    if isinstance(v, dict):
        return {k: _encode(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_encode(x) for x in v]
    return v


def _decode(v: Any) -> Any:
    if isinstance(v, dict):
        if set(v.keys()) == {"__tuple__"}:
            return tuple(_decode(x) for x in v["__tuple__"])
        return {k: _decode(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_decode(x) for x in v]
    return v


def _step_to_dict(s: DerivationStep) -> Dict[str, Any]:
    return {"path": s.path, "production": s.production, "params": _encode(s.params)}


def _step_from_dict(d: Dict[str, Any]) -> DerivationStep:
    return DerivationStep(path=d["path"], production=d["production"], params=_decode(d["params"]))


def derivation_to_dict(d: Derivation) -> Dict[str, Any]:
    return {
        "schema": DERIVATION_SCHEMA,
        "seed": d.seed,
        "grammar_version": d.grammar_version,
        "steps": [_step_to_dict(s) for s in d.steps],
        "lineage": _encode(d.lineage),
    }


def derivation_from_dict(d: Dict[str, Any]) -> Derivation:
    if d.get("schema") != DERIVATION_SCHEMA:
        raise ValueError(f"unsupported schema {d.get('schema')!r}, expected {DERIVATION_SCHEMA!r}")
    return Derivation(
        seed=d["seed"],
        grammar_version=d["grammar_version"],
        steps=tuple(_step_from_dict(s) for s in d["steps"]),
        lineage=_decode(d["lineage"]) if "lineage" in d else (),
    )


def derivation_to_json(d: Derivation) -> str:
    return json.dumps(derivation_to_dict(d), allow_nan=False)


def derivation_from_json(text: str) -> Derivation:
    return derivation_from_dict(json.loads(text))


# --------------------------------------------------------------------------
# vary: structural / parametric mutation operators
# --------------------------------------------------------------------------

OPERATORS: Tuple[str, ...] = (
    "resample_parameter",
    "perturb_parameter",
    "regrow_subtree",
    "insert_phalanx",
    "delete_phalanx",
    "add_digit",
    "remove_digit",
)


def _growth_dist(dist: Distribution) -> Distribution:
    """I14 fix 5: the ``Distribution`` a GROWTH operator (``add_digit``,
    ``add_palm_body``, ``regrow_subtree``, ``add_minimal_digit``) should
    draw its brand-new material from -- ``dist.insertion`` if set, else
    ``dist`` itself (current behaviour, unchanged). Caller-level caps
    (``digit_count_range``, ``palm_body_count_range``) are never read from
    this: they bound the whole hand, so callers must keep reading those off
    the outer ``dist``, not this function's return value."""
    return dist.insertion if dist.insertion is not None else dist


def _op_resample_parameter(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    idx = int(rng.integers(0, len(steps)))
    s = steps[idx]
    if s.production == "PalmBody":
        p = dict(s.params)
        length = sample_grid_length_m(rng, dist.palm_length_range_m, dist.link_length_grid_m)
        direction_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
        has_joint = bool(float(rng.random()) < dist.palm_joint_probability)
        axis = sample_axis(rng)
        limits = sample_palm_joint_limits_rad(rng, dist) if has_joint else None
        p.update({"length": length, "direction_rpy": direction_rpy, "has_joint": has_joint,
                  "axis": axis, "limits": limits})
        steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
        return steps
    if s.production == "Phalanx":
        p = dict(s.params)
        module = sample_module(rng, dist, p["p"], _revolute_source_indices(steps, p["digit_id"], p["p"]))
        length = sample_grid_length_m(rng, dist.link_length_range_m, dist.link_length_grid_m)
        p.update({"module": module, "length": length})
        steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
        return steps
    if s.production == "Digit":
        p = dict(s.params)
        if p["top_level"]:
            mount_bodies = _mount_bodies_from_steps(steps)
            p["mount"] = mount_bodies[int(rng.integers(0, len(mount_bodies)))]
        # A branch digit's mount is fixed to its host phalanx's body (it is
        # not a free choice); only its pose within that body is resampled.
        p["mount_frac"] = float(dist.mount_frac_choices[int(rng.integers(0, len(dist.mount_frac_choices)))])
        p["mount_rpy"] = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
        steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
        return steps
    return None


def _op_perturb_parameter(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    candidates = [i for i, s in enumerate(steps) if s.production in ("PalmBody", "Phalanx")]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    length_range = dist.palm_length_range_m if s.production == "PalmBody" else dist.link_length_range_m
    grid = dist.link_length_grid_m
    lo, hi = length_range
    direction = 1.0 if float(rng.random()) < 0.5 else -1.0
    new_len = p["length"] + direction * grid
    if new_len > hi:
        new_len = hi - (new_len - hi)
    if new_len < lo:
        new_len = lo + (lo - new_len)
    new_len = min(max(new_len, lo), hi)
    p["length"] = round(new_len, 10)
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


def _op_regrow_subtree(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    digit_idxs = [i for i, s in enumerate(steps) if s.production == "Digit"]
    if not digit_idxs:
        return None
    idx = digit_idxs[int(rng.integers(0, len(digit_idxs)))]
    s = steps[idx]
    digit_id = s.params["digit_id"]
    top_level = s.params["top_level"]
    depth = s.params["depth"]
    mount = s.params["mount"]
    # Top-level digits may re-pick which palm body they mount on; a branch
    # digit's mount is fixed to its host phalanx's body.
    mount_bodies = _mount_bodies_from_steps(steps) if top_level else [mount]

    kept = [st for st in steps if not _is_descendant_digit(digit_id, _step_digit_id(st))]

    new_steps: List[DerivationStep] = []
    next_uid = [_max_uid(steps) + 1]
    _emit_digit(rng, _growth_dist(dist), new_steps, digit_id, mount_bodies, top_level, depth, next_uid)
    return kept + new_steps


def _rename_branch_mounts(steps: List[DerivationStep], host_digit_id: str,
                           renumber: Dict[int, int]) -> List[DerivationStep]:
    """After insert/delete-phalanx renumbers ``host_digit_id``'s local
    phalanx indices, update the ``mount`` field of any branch ``Digit`` step
    hosted on one of those phalanges so it still names the (renumbered) body
    it is actually mounted on. The branch digit's own id/body names are
    never renamed (see ``rules.py``'s module docstring): they stay valid,
    globally unique identifiers even if they no longer literally encode the
    host's *current* phalanx index."""
    rename = {
        f"d{host_digit_id}p{old + 1}": f"d{host_digit_id}p{new + 1}"
        for old, new in renumber.items() if old != new
    }
    if not rename:
        return steps
    out = []
    for st in steps:
        if (st.production == "Digit" and not st.params.get("top_level", True)
                and st.params.get("mount") in rename):
            out.append(DerivationStep(path=st.path, production="Digit",
                                       params={**st.params, "mount": rename[st.params["mount"]]}))
        else:
            out.append(st)
    return out


def _op_insert_phalanx(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    digit_idxs = [i for i, s in enumerate(steps)
                  if s.production == "Digit" and s.params["phalanx_count"] < dist.phalanx_count_range[1]]
    if not digit_idxs:
        return None
    idx = digit_idxs[int(rng.integers(0, len(digit_idxs)))]
    dstep = steps[idx]
    digit_id = dstep.params["digit_id"]
    old_count = dstep.params["phalanx_count"]
    ins_p = int(rng.integers(0, old_count + 1))

    others = [st for st in steps if st is not dstep]
    phalanx_steps = {st.params["p"]: st for st in others
                      if st.production == "Phalanx" and st.params["digit_id"] == digit_id}
    non_digit_others = [st for st in others
                         if not (st.production == "Phalanx" and st.params["digit_id"] == digit_id)]

    renumber: Dict[int, int] = {}
    new_phalanx_list: List[DerivationStep] = []
    for old_p in range(old_count):
        new_p = old_p if old_p < ins_p else old_p + 1
        renumber[old_p] = new_p
        st = phalanx_steps[old_p]
        params = dict(st.params)
        mod = dict(params["module"])
        if mod.get("kind") == "Coupled":
            sp = mod["source_p"]
            mod["source_p"] = sp if sp < ins_p else sp + 1
        params["module"] = mod
        params["p"] = new_p
        new_phalanx_list.append(
            DerivationStep(path=f"digit/{digit_id}/phalanx/{new_p}", production="Phalanx", params=params))

    revolute_source_indices = tuple(
        old_p for old_p in range(ins_p) if phalanx_steps[old_p].params["module"]["kind"] == "R"
    )
    module = sample_module(rng, dist, ins_p, revolute_source_indices)
    length = sample_grid_length_m(rng, dist.link_length_range_m, dist.link_length_grid_m)
    bend_rpy, bend_offset = sample_bend(rng, dist)
    new_uid = _max_uid(steps) + 1
    new_phalanx_list.append(DerivationStep(path=f"digit/{digit_id}/phalanx/{ins_p}", production="Phalanx", params={
        "digit_id": digit_id, "p": ins_p, "module": module, "length": length, "branch_digit_count": 0,
        "uid": new_uid, "bend_rpy": bend_rpy, "bend_offset": bend_offset,
    }))

    new_dstep = DerivationStep(path=dstep.path, production="Digit",
                                params={**dstep.params, "phalanx_count": old_count + 1})
    result = non_digit_others + [new_dstep] + new_phalanx_list
    return _rename_branch_mounts(result, digit_id, renumber)


def _op_delete_phalanx(rng, dist: Distribution, derivation: Derivation,
                        target: Optional[int] = None) -> Optional[List[DerivationStep]]:
    """``target`` (grammar 0.5, reversibility): when given, the uid of the
    ``Phalanx`` step to delete -- restricts the candidate pool to exactly
    that phalanx (still subject to every safety check below) instead of
    drawing uniformly, so a caller (the reversibility test) can undo a
    specific ``insert_phalanx`` application precisely.

    I15 fix 4 (I18 fix 1 corrects the ``del_p == 0`` case below): a phalanx
    that hosts a branch MAY now be deleted -- its branch sub-digit(s) are
    re-attached (their ``Digit`` step's ``mount`` field updated) rather than
    orphaned: to the PROXIMAL neighbour phalanx's body (index ``del_p - 1``,
    whose body name is never itself renumbered by this deletion, since only
    phalanges with an index ABOVE ``del_p`` shift down) if ``del_p > 0``, or
    to the digit's NEW phalanx 0 (the phalanx that was index 1, before it is
    itself renumbered down to index 0) if ``del_p == 0`` (the first
    phalanx) -- NEVER to the digit's own ``mount`` (I18 fix 1: that body can
    be a palm body when this digit is top-level, and a branch digit must
    never mount on a palm body; the old code reattached there, which broke
    that invariant). A digit with only 1 phalanx is never a delete_phalanx
    candidate at all (see the ``phalanx_count <= 1: continue`` guard below),
    so this case never has to fall back to the digit's own mount: there is
    always a new phalanx 0 to reattach to once ``del_p == 0`` is reached.
    Deleting any phalanx except the digit's *current last one* never
    changes which phalanx is last, so it is always safe. Deleting the last
    one promotes the previous phalanx to "last" -- still refused when that
    PREVIOUS phalanx (not ``del_p`` itself) hosts exactly 1 branch digit of
    its OWN (valid only for a non-last phalanx, which already has a
    next-phalanx child; as the new last phalanx it would drop to a single
    child) -- this pre-existing safety check is unrelated to ``del_p``'s own
    branches, which are simply relocated, never left dangling."""
    steps = list(derivation.steps)
    candidates: List[Tuple[int, List[int]]] = []
    for i, s in enumerate(steps):
        if s.production != "Digit" or s.params["phalanx_count"] <= 1:
            continue
        digit_id = s.params["digit_id"]
        phalanx_count = s.params["phalanx_count"]
        last_idx = phalanx_count - 1
        phalanx_by_p = {
            pp.params["p"]: pp for pp in steps
            if pp.production == "Phalanx" and pp.params["digit_id"] == digit_id
        }
        second_last = phalanx_by_p.get(last_idx - 1)
        last_deletion_safe = not (second_last is not None and second_last.params["branch_digit_count"] == 1)
        deletable = [
            p_idx for p_idx in phalanx_by_p
            if (p_idx != last_idx or last_deletion_safe)
            and (target is None or phalanx_by_p[p_idx].params.get("uid") == target)
        ]
        if deletable:
            candidates.append((i, deletable))
    if not candidates:
        return None
    idx, deletable = candidates[int(rng.integers(0, len(candidates)))]
    dstep = steps[idx]
    digit_id = dstep.params["digit_id"]
    old_count = dstep.params["phalanx_count"]
    del_p = deletable[int(rng.integers(0, len(deletable)))]

    others = [st for st in steps if st is not dstep]
    phalanx_steps = {st.params["p"]: st for st in others
                      if st.production == "Phalanx" and st.params["digit_id"] == digit_id}
    non_digit_others = [st for st in others
                         if not (st.production == "Phalanx" and st.params["digit_id"] == digit_id)]

    # Re-attach ``del_p``'s own branch sub-digits (if any) BEFORE the
    # renumbering pass below, since the deleted phalanx's body never
    # survives renumbering at all (it is not merely renamed, it is gone).
    deleted_body = f"d{digit_id}p{del_p + 1}"
    if del_p > 0:
        # Proximal neighbour's body name is unaffected by this deletion
        # (only phalanges with an index ABOVE del_p shift down).
        reattach_mount = f"d{digit_id}p{del_p}"
    else:
        # I18 fix 1: reattach to the phalanx that BECOMES the new phalanx 0
        # (old index 1), using ITS pre-renumbering name -- the renumbering
        # pass + ``_rename_branch_mounts`` call below then retargets this
        # same mount (along with every other branch already hosted there)
        # from "d{digit_id}p2" to "d{digit_id}p1", so the reattached branch
        # ends up on the correct final phalanx-0 body. Never the digit's
        # own ``mount``: that can be a palm body when this digit is
        # top-level, and a branch digit must never mount on a palm body.
        reattach_mount = f"d{digit_id}p{del_p + 2}"
    non_digit_others = [
        DerivationStep(path=st.path, production="Digit", params={**st.params, "mount": reattach_mount})
        if (st.production == "Digit" and not st.params.get("top_level", True)
            and st.params.get("mount") == deleted_body)
        else st
        for st in non_digit_others
    ]

    renumber: Dict[int, int] = {}
    new_phalanx_list: List[DerivationStep] = []
    for old_p in range(old_count):
        if old_p == del_p:
            continue
        new_p = old_p if old_p < del_p else old_p - 1
        renumber[old_p] = new_p
        st = phalanx_steps[old_p]
        params = dict(st.params)
        mod = dict(params["module"])
        if mod.get("kind") == "Coupled":
            sp = mod["source_p"]
            if sp == del_p:
                # Source phalanx removed: fall back to a fresh, non-Coupled module.
                mod = sample_module(rng, dist, 0)
            elif sp > del_p:
                mod["source_p"] = sp - 1
        params["module"] = mod
        params["p"] = new_p
        new_phalanx_list.append(
            DerivationStep(path=f"digit/{digit_id}/phalanx/{new_p}", production="Phalanx", params=params))

    new_dstep = DerivationStep(path=dstep.path, production="Digit",
                                params={**dstep.params, "phalanx_count": old_count - 1})
    result = non_digit_others + [new_dstep] + new_phalanx_list
    return _rename_branch_mounts(result, digit_id, renumber)


def _op_add_digit(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    if hand_params["digit_count"] >= dist.digit_count_range[1]:
        return None
    mount_bodies = _mount_bodies_from_steps(steps)
    top_ids = [int(s.params["digit_id"]) for s in steps if s.production == "Digit" and s.params.get("top_level")]
    next_id = [max(top_ids, default=0) + 1]
    new_steps: List[DerivationStep] = []
    next_uid = [_max_uid(steps) + 1]
    _sample_digit(rng, _growth_dist(dist), new_steps, next_id, mount_bodies, next_uid)
    new_hand = DerivationStep(path="hand", production="Hand",
                               params={**hand_params, "digit_count": hand_params["digit_count"] + 1})
    rest = [s for i, s in enumerate(steps) if i != hand_idx]
    return rest + [new_hand] + new_steps


def _op_remove_digit(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    if hand_params["digit_count"] <= 1:
        return None
    top_ids = [s.params["digit_id"] for s in steps if s.production == "Digit" and s.params.get("top_level")]
    if not top_ids:
        return None
    digit_id = top_ids[int(rng.integers(0, len(top_ids)))]
    kept = [s for s in steps if not _is_descendant_digit(digit_id, _step_digit_id(s))]
    new_hand = DerivationStep(path="hand", production="Hand",
                               params={**hand_params, "digit_count": hand_params["digit_count"] - 1})
    return [s if s.path != "hand" else new_hand for s in kept]


# --------------------------------------------------------------------------
# Small-step operators (opt-in only -- see ``vary``'s ``operators`` argument
# and ``SMALL_STEP_OPERATORS`` below). Each edits EXACTLY one field of one
# existing step (never adds/removes a step), by moving that field to a
# grid-neighbouring choice rather than resampling it fresh -- unlike
# ``resample_parameter``/``perturb_parameter`` above, which redraw a field
# from its whole distribution. Adding these to ``_OPERATOR_FNS`` does not
# change ``OPERATORS`` or default ``vary`` behaviour: a caller only reaches
# them by naming them explicitly (``operator=...``) or opting into
# ``operators=SMALL_STEP_OPERATORS``.
# --------------------------------------------------------------------------

SMALL_STEP_OPERATORS: Tuple[str, ...] = (
    "step_axis",
    "step_limits",
    "step_mount",
    "step_coupling",
    # I14 fix 5: ``step_length`` was an exact alias of ``perturb_parameter``
    # (same function, see ``_OPERATOR_FNS`` below) -- included here it gave
    # any pool containing both double weight on the same effect. The
    # function itself (and the ``"step_length"`` key in ``_OPERATOR_FNS``,
    # reachable via ``operator="step_length"``) stays, for compatibility;
    # only the default small-step POOL no longer draws it.
    "step_root_length",
    "step_radius",
    # Grammar 0.5 (I16 priority 1): one grid step on one component of a
    # phalanx's own bend_rpy/bend_offset (see _op_step_bend_rpy/_offset
    # below).
    "step_bend_rpy",
    "step_bend_offset",
)


def _axis_grid_indices(axis: Tuple[float, float, float]) -> Tuple[int, int]:
    """Invert ``distributions.sample_axis``'s grid: find the (elevation,
    azimuth) step indices whose axis matches ``axis`` exactly (every axis
    stored in a derivation was produced by that same formula, so this is an
    exact float match, not a nearest-neighbour search)."""
    x, y, z = axis
    for el_k in range(N_ELEVATION_STEPS):
        el = el_k * ANGLE_STEP_DEG * DEG
        for az_k in range(N_ANGLE_STEPS):
            az = (az_k * ANGLE_STEP_DEG - 180.0) * DEG
            xx, yy, zz = math.sin(el) * math.cos(az), math.sin(el) * math.sin(az), math.cos(el)
            if abs(xx - x) < 1e-9 and abs(yy - y) < 1e-9 and abs(zz - z) < 1e-9:
                return el_k, az_k
    raise ValueError(f"axis {axis!r} is not on the sampling grid")


def _step_axis_value(rng, axis: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """One grid step in elevation (clamped to [0, N_ELEVATION_STEPS-1]) or
    azimuth (wrapped, since azimuth is circular), chosen at random."""
    el_k, az_k = _axis_grid_indices(axis)
    step_elevation = bool(rng.integers(0, 2))
    direction = 1 if bool(rng.integers(0, 2)) else -1
    if step_elevation:
        el_k = max(0, min(N_ELEVATION_STEPS - 1, el_k + direction))
    else:
        az_k = (az_k + direction) % N_ANGLE_STEPS
    el = el_k * ANGLE_STEP_DEG * DEG
    az = (az_k * ANGLE_STEP_DEG - 180.0) * DEG
    return (float(math.sin(el) * math.cos(az)), float(math.sin(el) * math.sin(az)), float(math.cos(el)))


def _op_step_axis(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    candidates = [
        i for i, s in enumerate(steps)
        if (s.production == "PalmBody" and s.params.get("has_joint"))
        or s.production == "Phalanx"
    ]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    try:
        if s.production == "PalmBody":
            p["axis"] = _step_axis_value(rng, p["axis"])
        else:
            mod = dict(p["module"])
            mod["axis"] = _step_axis_value(rng, mod["axis"])
            p["module"] = mod
    except ValueError:
        return None
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


def _find_choice_index(value: Tuple[float, float], choices: Sequence[Tuple[float, float]],
                        scale: float) -> Optional[int]:
    for i, (lo, hi) in enumerate(choices):
        if abs(lo * scale - value[0]) < 1e-9 and abs(hi * scale - value[1]) < 1e-9:
            return i
    return None


def _step_choice_index(rng, idx: int, n: int) -> int:
    if n <= 1:
        return idx
    direction = 1 if bool(rng.integers(0, 2)) else -1
    return max(0, min(n - 1, idx + direction))


def _op_step_limits(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    """I14 fix: ``dist.*_limit_choices_*`` are sorted HERE (a local copy, on
    every call) before finding/stepping the current choice's index, so a
    "neighbouring" choice is numerically adjacent (e.g. never jumps a
    revolute joint's limits from ``(0, 110)`` straight to ``(-30, 60)``, nor
    flips a multiplier's sign) -- WITHOUT reordering the ``Distribution``
    field itself, since ``resample_parameter``/``sample_module`` pick a
    choice by RAW index into that same (unsorted) tuple, and reordering it
    would silently change every existing seed's default-sampling replay."""
    steps = list(derivation.steps)
    candidates = [
        i for i, s in enumerate(steps)
        if (s.production == "PalmBody" and s.params.get("has_joint"))
        or (s.production == "Phalanx" and s.params["module"]["kind"] in ("R", "P"))
    ]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    if s.production == "PalmBody":
        choices = sorted(dist.palm_joint_limit_choices_deg)
        ci = _find_choice_index(p["limits"], choices, scale=DEG)
        if ci is None:
            return None
        new_ci = _step_choice_index(rng, ci, len(choices))
        if new_ci == ci:
            return None
        lo_deg, hi_deg = choices[new_ci]
        p["limits"] = (lo_deg * DEG, hi_deg * DEG)
        steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
        return steps
    mod = dict(p["module"])
    if mod["kind"] == "R" and dist.limits_continuous:
        # Grammar 0.5 (I16 priority 2): move ONE bound by one
        # ``dist.limit_step_deg`` grid step, clamped within
        # ``dist.revolute_limit_range_deg`` and away from the other bound
        # (never crossing it, so ``lo < hi`` always still holds).
        lo, hi = mod["limits"]
        range_lo_deg, range_hi_deg = dist.revolute_limit_range_deg
        range_lo, range_hi = range_lo_deg * DEG, range_hi_deg * DEG
        step = dist.limit_step_deg * DEG
        direction = 1.0 if bool(rng.integers(0, 2)) else -1.0
        if bool(rng.integers(0, 2)):
            new_lo = max(range_lo, min(lo + direction * step, min(range_hi, hi - 1e-9)))
            if abs(new_lo - lo) < 1e-12:
                return None
            mod["limits"] = (new_lo, hi)
        else:
            new_hi = min(range_hi, max(hi + direction * step, max(range_lo, lo + 1e-9)))
            if abs(new_hi - hi) < 1e-12:
                return None
            mod["limits"] = (lo, new_hi)
        p["module"] = mod
        steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
        return steps
    if mod["kind"] == "R":
        choices, scale = sorted(dist.revolute_limit_choices_deg), DEG
    else:
        choices, scale = sorted(dist.prismatic_limit_choices_m), 1.0
    ci = _find_choice_index(mod["limits"], choices, scale=scale)
    if ci is None:
        return None
    new_ci = _step_choice_index(rng, ci, len(choices))
    if new_ci == ci:
        return None
    lo, hi = choices[new_ci]
    mod["limits"] = (lo * scale, hi * scale)
    p["module"] = mod
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


def _step_one_grid_angle(rng, current: float) -> float:
    k = int(round((current / DEG + 180.0) / ANGLE_STEP_DEG)) % N_ANGLE_STEPS
    direction = 1 if bool(rng.integers(0, 2)) else -1
    new_k = (k + direction) % N_ANGLE_STEPS
    return (new_k * ANGLE_STEP_DEG - 180.0) * DEG


def _op_step_mount(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    candidates = [i for i, s in enumerate(steps) if s.production in ("Digit", "PalmBody")]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    # A ``PalmBody`` step has no ``mount_rpy`` field of its own; its
    # ``direction_rpy`` (the segment's own orientation) plays the analogous
    # role and is stepped the same way.
    rpy_field = "mount_rpy" if s.production == "Digit" else "direction_rpy"
    step_frac = bool(rng.integers(0, 2))
    if step_frac:
        choices = list(dist.mount_frac_choices)
        if p["mount_frac"] not in choices or len(choices) <= 1:
            return None
        ci = choices.index(p["mount_frac"])
        new_ci = _step_choice_index(rng, ci, len(choices))
        if new_ci == ci:
            return None
        p["mount_frac"] = choices[new_ci]
    else:
        rpy = list(p[rpy_field])
        axis_i = int(rng.integers(0, 3))
        rpy[axis_i] = _step_one_grid_angle(rng, rpy[axis_i])
        p[rpy_field] = tuple(rpy)
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


def _op_step_coupling(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    candidates = [
        i for i, s in enumerate(steps)
        if s.production == "Phalanx" and s.params["module"]["kind"] == "Coupled"
    ]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    mod = dict(p["module"])
    field = "multiplier" if bool(rng.integers(0, 2)) else "offset"
    # I14 fix: sorted locally (see ``_op_step_limits``'s docstring) so a
    # neighbour is numerically adjacent, e.g. never flips a multiplier's
    # sign in one step.
    choices = sorted(dist.coupling_multiplier_choices if field == "multiplier" else dist.coupling_offset_choices_rad)
    cur = mod[field]
    if cur not in choices or len(choices) <= 1:
        return None
    ci = choices.index(cur)
    new_ci = _step_choice_index(rng, ci, len(choices))
    if new_ci == ci:
        return None
    mod[field] = choices[new_ci]
    p["module"] = mod
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


def _op_step_root_length(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    """I14 fix 5: the ``hand`` step's own ``root_length`` (the root/palm
    segment's length), previously never mutated by ANY operator, stepped by
    one ``dist.link_length_grid_m`` grid increment (up or down, clamped
    within ``dist.palm_length_range_m`` -- the same range/grid
    ``root_length`` is originally sampled from)."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand = steps[hand_idx]
    lo, hi = dist.palm_length_range_m
    grid = dist.link_length_grid_m
    n = int(round((hi - lo) / grid))
    cur = hand.params["root_length"]
    ci = int(round((cur - lo) / grid))
    ci = max(0, min(n, ci))
    new_ci = _step_choice_index(rng, ci, n + 1)
    if new_ci == ci:
        return None
    p = dict(hand.params)
    p["root_length"] = round(lo + new_ci * grid, 10)
    steps[hand_idx] = DerivationStep(path="hand", production="Hand", params=p)
    return steps


def _op_step_radius(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    """I14 fix 5: the ``hand`` step's own ``capsule_radius_m`` (one scalar
    stamped onto every ``Body.radius`` -- previously never mutated by any
    operator), stepped to a numerically-neighbouring choice in
    ``dist.capsule_radius_choices_m`` (sorted locally, same rationale as
    ``_op_step_limits``)."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand = steps[hand_idx]
    choices = sorted(dist.capsule_radius_choices_m)
    cur = hand.params["capsule_radius_m"]
    if cur not in choices or len(choices) <= 1:
        return None
    ci = choices.index(cur)
    new_ci = _step_choice_index(rng, ci, len(choices))
    if new_ci == ci:
        return None
    p = dict(hand.params)
    p["capsule_radius_m"] = choices[new_ci]
    steps[hand_idx] = DerivationStep(path="hand", production="Hand", params=p)
    return steps


def _bend_component_grid(choices, axis_index: int) -> List[float]:
    """Sorted, de-duplicated set of the ``axis_index``-th component across
    every whole triple/pair in ``choices`` (``dist.bend_rpy_choices_rad`` --
    3 components -- or ``dist.bend_offset_choices_m`` -- 2 components).
    ``variants.G_BEND`` builds each of these choice sets as the full
    Cartesian product of one small per-component grid, so this recovers
    that per-component grid for the small-step operators below (which move
    ONE component by one grid step, unlike ``step_limits``/``step_coupling``,
    which step the whole tuple to its neighbour in a flat choice list)."""
    return sorted({c[axis_index] for c in choices})


def _op_step_bend_rpy(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    """Grammar 0.5 (I16 priority 1): move ONE component (roll, pitch, or
    yaw) of one Phalanx step's own ``bend_rpy`` to its numerically
    neighbouring value on ``dist.bend_rpy_choices_rad``'s own per-component
    grid (see ``_bend_component_grid``). A no-op (``None``) whenever that
    component's grid has only one value (true of every default
    ``Distribution``, whose ``bend_rpy_choices_rad`` is the single value
    ``(0.0, 0.0, 0.0)``) or the current value is not itself on that grid."""
    steps = list(derivation.steps)
    candidates = [i for i, s in enumerate(steps) if s.production == "Phalanx"]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    bend_rpy = list(p.get("bend_rpy", (0.0, 0.0, 0.0)))
    comp = int(rng.integers(0, 3))
    grid = _bend_component_grid(dist.bend_rpy_choices_rad, comp)
    if len(grid) <= 1 or bend_rpy[comp] not in grid:
        return None
    ci = grid.index(bend_rpy[comp])
    new_ci = _step_choice_index(rng, ci, len(grid))
    if new_ci == ci:
        return None
    bend_rpy[comp] = grid[new_ci]
    p["bend_rpy"] = tuple(bend_rpy)
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


def _op_step_bend_offset(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    """Grammar 0.5 (I16 priority 1): the ``bend_offset`` counterpart of
    ``_op_step_bend_rpy`` -- moves ONE component (x or y) of one Phalanx
    step's own ``bend_offset`` to its numerically neighbouring value on
    ``dist.bend_offset_choices_m``'s own per-component grid."""
    steps = list(derivation.steps)
    candidates = [i for i, s in enumerate(steps) if s.production == "Phalanx"]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    bend_offset = list(p.get("bend_offset", (0.0, 0.0)))
    comp = int(rng.integers(0, 2))
    grid = _bend_component_grid(dist.bend_offset_choices_m, comp)
    if len(grid) <= 1 or bend_offset[comp] not in grid:
        return None
    ci = grid.index(bend_offset[comp])
    new_ci = _step_choice_index(rng, ci, len(grid))
    if new_ci == ci:
        return None
    bend_offset[comp] = grid[new_ci]
    p["bend_offset"] = tuple(bend_offset)
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


# --------------------------------------------------------------------------
# Minimal structural operators (opt-in only -- see ``MINIMAL_STRUCTURAL_OPERATORS``
# below). Motivation (E1): ``add_digit``/``remove_digit`` change ~7 joints in
# one application because a new digit is sampled at full size, and no
# default operator changes palm structure at all. Each operator here edits
# structure by the smallest possible increment (one single-phalanx digit, one
# palm body, one palm joint's presence). Adding these to ``_OPERATOR_FNS``
# does not change ``OPERATORS`` or default ``vary`` behaviour: a caller only
# reaches them via ``operator=...`` or ``operators=MINIMAL_STRUCTURAL_OPERATORS``.
# --------------------------------------------------------------------------

MINIMAL_STRUCTURAL_OPERATORS: Tuple[str, ...] = (
    "add_minimal_digit",
    "add_palm_body",
    "remove_palm_body",
    "toggle_palm_joint",
    "remove_digit_minimal",
)


def _op_add_minimal_digit(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    """Add a new top-level digit with exactly one phalanx (module kind
    always ``"R"`` -- a single-phalanx digit has no earlier phalanx to
    couple to, so ``"Coupled"`` is never valid there anyway; forcing ``"R"``
    keeps this operator's effect exactly one movable revolute joint, never a
    continuous/prismatic one). Mounts on a uniformly sampled existing body
    (root or a palm body), like a fresh top-level digit."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    if hand_params["digit_count"] >= dist.digit_count_range[1]:
        return None
    gdist = _growth_dist(dist)
    mount_bodies = _mount_bodies_from_steps(steps)
    top_ids = [int(s.params["digit_id"]) for s in steps if s.production == "Digit" and s.params.get("top_level")]
    digit_id = str(max(top_ids, default=0) + 1)
    mount = mount_bodies[int(rng.integers(0, len(mount_bodies)))]
    mount_frac = float(gdist.mount_frac_choices[int(rng.integers(0, len(gdist.mount_frac_choices)))])
    mount_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
    axis = sample_axis(rng)
    limits = sample_revolute_limits_rad(rng, gdist)
    length = sample_grid_length_m(rng, gdist.link_length_range_m, gdist.link_length_grid_m)
    bend_rpy, bend_offset = sample_bend(rng, gdist)
    uid_base = _max_uid(steps) + 1
    digit_step = DerivationStep(path=f"digit/{digit_id}", production="Digit", params={
        "digit_id": digit_id, "mount": mount, "mount_frac": mount_frac, "mount_rpy": mount_rpy,
        "phalanx_count": 1, "top_level": True, "depth": 0, "uid": uid_base,
    })
    phalanx_step = DerivationStep(path=f"digit/{digit_id}/phalanx/0", production="Phalanx", params={
        "digit_id": digit_id, "p": 0, "module": {"kind": "R", "axis": axis, "limits": limits},
        "length": length, "branch_digit_count": 0, "uid": uid_base + 1,
        "bend_rpy": bend_rpy, "bend_offset": bend_offset,
    })
    new_hand = DerivationStep(path="hand", production="Hand",
                               params={**hand_params, "digit_count": hand_params["digit_count"] + 1})
    rest = [s for i, s in enumerate(steps) if i != hand_idx]
    return rest + [new_hand, digit_step, phalanx_step]


def _op_remove_digit_minimal(rng, dist: Distribution, derivation: Derivation,
                              target: Optional[int] = None) -> Optional[List[DerivationStep]]:
    """Reversible counterpart of ``add_minimal_digit``: remove a top-level
    digit with 1 or 2 phalanges (I15 fix 4, widened from exactly 1 --
    ``add_minimal_digit`` itself only ever ADDS a 1-phalanx digit, but
    ``insert_phalanx``/other operators applied afterward can grow that
    digit to 2 phalanges without this operator's own ratchet ever letting
    it back down again), none of which host a branch (removing the whole
    digit here -- via ``_is_descendant_digit`` -- also removes any branch
    nested under it, which would make this "minimal" operator silently
    remove a much larger subtree; ``delete_phalanx`` is the operator that
    handles branch re-attachment).

    ``target`` (grammar 0.5, reversibility): when given, the uid of the
    ``Digit`` step to remove -- restricts the candidate pool to exactly that
    digit (still subject to every safety check above) instead of drawing
    uniformly, so a caller (the reversibility test) can undo a specific
    ``add_minimal_digit`` application precisely."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    if hand_params["digit_count"] <= 1:
        return None
    phalanx_by_digit: Dict[str, List[DerivationStep]] = {}
    for s in steps:
        if s.production == "Phalanx":
            phalanx_by_digit.setdefault(s.params["digit_id"], []).append(s)
    candidates = [
        s.params["digit_id"] for s in steps
        if s.production == "Digit" and s.params.get("top_level") and s.params["phalanx_count"] in (1, 2)
        and len(phalanx_by_digit.get(s.params["digit_id"], [])) == s.params["phalanx_count"]
        and all(ph.params["branch_digit_count"] == 0 for ph in phalanx_by_digit.get(s.params["digit_id"], []))
        and (target is None or s.params.get("uid") == target)
    ]
    if not candidates:
        return None
    digit_id = candidates[int(rng.integers(0, len(candidates)))]
    kept = [s for s in steps if not _is_descendant_digit(digit_id, _step_digit_id(s))]
    new_hand = DerivationStep(path="hand", production="Hand",
                               params={**hand_params, "digit_count": hand_params["digit_count"] - 1})
    return [s if s.path != "hand" else new_hand for s in kept]


def _op_add_palm_body(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    """Add one new palm body (index ``palm_body_count``), sampled exactly
    like ``sample_derivation``'s own palm-body loop: a sampled parent among
    root and every existing palm body, a sampled mount fraction/length/
    direction, and a palm joint present with probability
    ``dist.palm_joint_probability``. Always appended at the end, so no
    existing palm/digit step's name or path needs renumbering."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    palm_body_count = hand_params["palm_body_count"]
    if palm_body_count >= dist.palm_body_count_range[1]:
        return None
    gdist = _growth_dist(dist)
    palm_names = [f"palm{i}" for i in range(palm_body_count)]
    parent_choices = ["root"] + palm_names
    parent = parent_choices[int(rng.integers(0, len(parent_choices)))]
    mount_frac = float(gdist.mount_frac_choices[int(rng.integers(0, len(gdist.mount_frac_choices)))])
    length = sample_grid_length_m(rng, gdist.palm_length_range_m, gdist.link_length_grid_m)
    direction_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
    has_joint = bool(float(rng.random()) < gdist.palm_joint_probability)
    axis = sample_axis(rng)
    limits = sample_palm_joint_limits_rad(rng, gdist) if has_joint else None
    name = f"palm{palm_body_count}"
    new_step = DerivationStep(path=f"palm/{palm_body_count}", production="PalmBody", params={
        "name": name, "parent": parent, "mount_frac": mount_frac, "length": length,
        "direction_rpy": direction_rpy, "has_joint": has_joint, "axis": axis, "limits": limits,
        "uid": _max_uid(steps) + 1,
    })
    new_hand = DerivationStep(path="hand", production="Hand",
                               params={**hand_params, "palm_body_count": palm_body_count + 1})
    rest = [s for i, s in enumerate(steps) if i != hand_idx]
    return rest + [new_hand, new_step]


def _op_remove_palm_body(rng, dist: Distribution, derivation: Derivation,
                          target: Optional[int] = None) -> Optional[List[DerivationStep]]:
    """``target`` (grammar 0.5, reversibility): when given, the uid of the
    ``PalmBody`` step to remove -- restricts the candidate pool to exactly
    that body instead of drawing uniformly, so a caller (the reversibility
    test) can undo a specific ``add_palm_body`` application precisely.

    Remove ANY existing palm body (I14 fix 5 -- previously restricted to
    a leaf palm body with no digit/palm children, which made this operator
    far less often applicable than ``add_palm_body``, an asymmetry the
    review flagged). If the removed body ``X`` hosts palm children
    (``PalmBody`` steps whose ``parent == X``) or digit mounts (``Digit``
    steps whose ``mount == X``), those are RE-ATTACHED to ``X``'s own
    parent, at ``X``'s own ``mount_frac`` on that parent (i.e. wherever
    ``X`` itself used to attach) -- an approximation (the reattached
    child's position along ``X`` itself, and ``X``'s own segment length,
    are both dropped), not an exact geometric inverse of ``add_palm_body``,
    but one that makes add/remove close to symmetric: applying
    ``add_palm_body`` then ``remove_palm_body`` on the body it just added
    always succeeds and returns to a body count matching the start (the
    grammar's own `parent index < child index` invariant guarantees ``X``'s
    parent is never itself renumbered by this removal, since only bodies
    with an index ABOVE ``X`` shift down).

    Every palm body with a higher index than the removed one is renumbered
    down by one (name and ``path``)."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    palm_body_count = hand_params["palm_body_count"]
    if palm_body_count == 0:
        return None
    palm_steps = {s.params["name"]: s for s in steps if s.production == "PalmBody"}
    if target is None:
        candidates = list(palm_steps.keys())
    else:
        candidates = [name for name, s in palm_steps.items() if s.params.get("uid") == target]
    if not candidates:
        return None
    remove_name = candidates[int(rng.integers(0, len(candidates)))]
    remove_step = palm_steps[remove_name]
    remove_idx = int(remove_name[len("palm"):])
    remove_parent = remove_step.params["parent"]
    remove_mount_frac = remove_step.params["mount_frac"]

    rename: Dict[str, str] = {
        f"palm{i}": f"palm{i - 1}" for i in range(remove_idx + 1, palm_body_count)
    }

    new_steps: List[DerivationStep] = []
    for s in steps:
        if s.path == f"palm/{remove_idx}":
            continue
        if s.production == "PalmBody":
            old_i = int(s.params["name"][len("palm"):])
            new_i = old_i if old_i < remove_idx else old_i - 1
            p = dict(s.params)
            p["name"] = f"palm{new_i}"
            if p["parent"] == remove_name:
                p["parent"] = remove_parent
                p["mount_frac"] = remove_mount_frac
            elif p["parent"] in rename:
                p["parent"] = rename[p["parent"]]
            new_steps.append(DerivationStep(path=f"palm/{new_i}", production="PalmBody", params=p))
        elif s.production == "Digit" and s.params.get("mount") == remove_name:
            new_steps.append(DerivationStep(path=s.path, production="Digit",
                                             params={**s.params, "mount": remove_parent,
                                                     "mount_frac": remove_mount_frac}))
        elif s.production == "Digit" and s.params.get("mount") in rename:
            new_steps.append(DerivationStep(path=s.path, production="Digit",
                                             params={**s.params, "mount": rename[s.params["mount"]]}))
        elif s.path == "hand":
            new_steps.append(DerivationStep(path="hand", production="Hand",
                                             params={**hand_params, "palm_body_count": palm_body_count - 1}))
        else:
            new_steps.append(s)
    return new_steps


def _op_remove_palm_body_empty(rng, dist: Distribution, derivation: Derivation,
                                target: Optional[int] = None) -> Optional[List[DerivationStep]]:
    """Exact-inverse counterpart of ``add_palm_body`` (grammar 0.5, I18 fix
    2): remove a LEAF palm body -- one with no ``PalmBody`` child (no other
    palm body's ``parent`` names it) and no ``Digit`` mounted on it -- i.e.
    exactly the shape ``add_palm_body`` always produces (it is always
    appended at the end, with nothing yet attached to it). Unlike the
    general ``remove_palm_body`` (which accepts ANY existing body and
    approximately re-attaches its children to its parent), this never
    re-attaches anything, since a leaf by construction has nothing to
    re-attach; it is used as the shrink half of the ``add_palm_body`` pair
    in ``EVOLUTION_PAIRS``/``EVOLUTION_OPERATORS`` so that pair's neutral
    walk is a genuine exact inverse rather than the approximate,
    asymmetric general removal. ``remove_palm_body`` itself stays available
    outside the evolution pool as a larger, non-exact-inverse move.

    ``target``: uid of the ``PalmBody`` step to remove (restricts the
    candidate pool to exactly that body instead of drawing uniformly, so
    the reversibility test can undo a specific ``add_palm_body``
    application precisely)."""
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    palm_body_count = hand_params["palm_body_count"]
    if palm_body_count == 0:
        return None
    palm_steps = {s.params["name"]: s for s in steps if s.production == "PalmBody"}
    parent_names = {s.params["parent"] for s in steps if s.production == "PalmBody"}
    mount_names = {s.params["mount"] for s in steps if s.production == "Digit"}
    leaf_names = [name for name in palm_steps if name not in parent_names and name not in mount_names]
    if target is not None:
        leaf_names = [name for name in leaf_names if palm_steps[name].params.get("uid") == target]
    if not leaf_names:
        return None
    remove_name = leaf_names[int(rng.integers(0, len(leaf_names)))]
    remove_idx = int(remove_name[len("palm"):])

    rename: Dict[str, str] = {
        f"palm{i}": f"palm{i - 1}" for i in range(remove_idx + 1, palm_body_count)
    }
    new_steps: List[DerivationStep] = []
    for s in steps:
        if s.path == f"palm/{remove_idx}":
            continue
        if s.production == "PalmBody":
            old_i = int(s.params["name"][len("palm"):])
            new_i = old_i if old_i < remove_idx else old_i - 1
            p = dict(s.params)
            p["name"] = f"palm{new_i}"
            if p["parent"] in rename:
                p["parent"] = rename[p["parent"]]
            new_steps.append(DerivationStep(path=f"palm/{new_i}", production="PalmBody", params=p))
        elif s.production == "Digit" and s.params.get("mount") in rename:
            new_steps.append(DerivationStep(path=s.path, production="Digit",
                                             params={**s.params, "mount": rename[s.params["mount"]]}))
        elif s.path == "hand":
            new_steps.append(DerivationStep(path="hand", production="Hand",
                                             params={**hand_params, "palm_body_count": palm_body_count - 1}))
        else:
            new_steps.append(s)
    return new_steps


def _op_toggle_palm_joint(rng, dist: Distribution, derivation: Derivation,
                           target: Optional[int] = None) -> Optional[List[DerivationStep]]:
    """On one existing (necessarily non-root -- the root has no ``PalmBody``
    step of its own) palm body, add a palm joint (sampled axis/limits) if
    absent, or remove it (``has_joint=False``, ``limits=None``) if present.

    Self-inverse (grammar 0.5, reversibility): toggling twice on the SAME
    body restores it exactly (on/off keeps the stored axis only if the
    limits were also byte-identical -- toggling off then on again resamples
    axis/limits, so use ``target`` below rather than relying on a second
    random toggle happening to land on the same body/axis). ``target``, when
    given, is the uid of the ``PalmBody`` step to toggle -- restricts the
    candidate pool to exactly that body instead of drawing uniformly."""
    steps = list(derivation.steps)
    if target is None:
        candidates = [i for i, s in enumerate(steps) if s.production == "PalmBody"]
    else:
        candidates = [i for i, s in enumerate(steps)
                      if s.production == "PalmBody" and s.params.get("uid") == target]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    s = steps[idx]
    p = dict(s.params)
    if p["has_joint"]:
        p["has_joint"] = False
        p["limits"] = None
    else:
        p["has_joint"] = True
        p["axis"] = sample_axis(rng)
        p["limits"] = sample_palm_joint_limits_rad(rng, dist)
    steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
    return steps


# --------------------------------------------------------------------------
# Branch-digit operators (grammar 0.5, iteration B): the exact-inverse pair
# for "branch" in ``EVOLUTION_OPERATORS``' operator table (add a
# single-phalanx branch digit on a sampled existing phalanx body / remove a
# branch digit with exactly 1 phalanx and no sub-branches of its own). Each
# mirrors ``add_minimal_digit``/``remove_digit_minimal`` but for a BRANCH
# digit (mounted on a phalanx body, ``top_level=False``) rather than a
# top-level one -- unlike a fresh top-level digit, a branch digit's HOST
# phalanx step tracks how many branches it hosts (``branch_digit_count``,
# see ``rules.PhalanxProduction``'s docstring), which both operators keep
# accurate (read by ``_op_delete_phalanx``'s own safety check and by
# ``_op_remove_digit_minimal``'s "no branch" gate).
# --------------------------------------------------------------------------


def _op_add_branch_digit(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    """Add one new branch digit (1 phalanx, mirrors ``add_minimal_digit``)
    on a uniformly sampled existing ``Phalanx`` step's own body, subject to
    ``dist.max_branch_depth`` (the host digit's own depth must still be
    below it) and ``dist.max_branch_digits`` (the host phalanx must not
    already host the max). New material (module/length/bend/mount pose) is
    drawn from ``_growth_dist(dist)``, like every other growth operator."""
    steps = list(derivation.steps)
    digit_depth = {s.params["digit_id"]: s.params["depth"] for s in steps if s.production == "Digit"}
    candidates = [
        i for i, s in enumerate(steps)
        if s.production == "Phalanx"
        and digit_depth.get(s.params["digit_id"], 0) < dist.max_branch_depth
        and s.params["branch_digit_count"] < dist.max_branch_digits
    ]
    if not candidates:
        return None
    idx = candidates[int(rng.integers(0, len(candidates)))]
    host = steps[idx]
    host_digit_id = host.params["digit_id"]
    host_p = host.params["p"]
    host_body = f"d{host_digit_id}p{host_p + 1}"
    slot = host.params["branch_digit_count"]
    sub_id = f"{host_digit_id}p{host_p + 1}b{slot}"
    depth = digit_depth[host_digit_id] + 1

    gdist = _growth_dist(dist)
    mount_frac = float(gdist.mount_frac_choices[int(rng.integers(0, len(gdist.mount_frac_choices)))])
    mount_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
    axis = sample_axis(rng)
    limits = sample_revolute_limits_rad(rng, gdist)
    length = sample_grid_length_m(rng, gdist.link_length_range_m, gdist.link_length_grid_m)
    bend_rpy, bend_offset = sample_bend(rng, gdist)
    uid_base = _max_uid(steps) + 1

    digit_step = DerivationStep(path=f"digit/{sub_id}", production="Digit", params={
        "digit_id": sub_id, "mount": host_body, "mount_frac": mount_frac, "mount_rpy": mount_rpy,
        "phalanx_count": 1, "top_level": False, "depth": depth, "uid": uid_base,
    })
    phalanx_step = DerivationStep(path=f"digit/{sub_id}/phalanx/0", production="Phalanx", params={
        "digit_id": sub_id, "p": 0, "module": {"kind": "R", "axis": axis, "limits": limits},
        "length": length, "branch_digit_count": 0, "uid": uid_base + 1,
        "bend_rpy": bend_rpy, "bend_offset": bend_offset,
    })
    new_host = DerivationStep(path=host.path, production="Phalanx",
                               params={**host.params, "branch_digit_count": slot + 1})
    rest = [s if i != idx else new_host for i, s in enumerate(steps)]
    return rest + [digit_step, phalanx_step]


def _op_remove_branch_digit(rng, dist: Distribution, derivation: Derivation,
                             target: Optional[int] = None) -> Optional[List[DerivationStep]]:
    """Exact-inverse counterpart of ``add_branch_digit``: remove a branch
    digit (``top_level=False``) with exactly 1 phalanx and no sub-branches
    of its own, decrementing its HOST phalanx's ``branch_digit_count``.
    ``target``: uid of the branch ``Digit`` step to remove (restricts the
    candidate pool instead of drawing uniformly, so a caller -- the
    reversibility test -- can undo a specific ``add_branch_digit``
    application precisely)."""
    steps = list(derivation.steps)
    phalanx_by_digit: Dict[str, DerivationStep] = {
        s.params["digit_id"]: s for s in steps if s.production == "Phalanx"
    }
    candidates = [
        s for s in steps
        if s.production == "Digit" and not s.params.get("top_level", True)
        and s.params["phalanx_count"] == 1
        and phalanx_by_digit.get(s.params["digit_id"]) is not None
        and phalanx_by_digit[s.params["digit_id"]].params["branch_digit_count"] == 0
        and (target is None or s.params.get("uid") == target)
    ]
    if not candidates:
        return None
    branch_step = candidates[int(rng.integers(0, len(candidates)))]
    branch_digit_id = branch_step.params["digit_id"]
    host_body = branch_step.params["mount"]

    kept = [s for s in steps if not _is_descendant_digit(branch_digit_id, _step_digit_id(s))]

    def _decrement_host(s: DerivationStep) -> DerivationStep:
        if s.production != "Phalanx":
            return s
        body_name = f"d{s.params['digit_id']}p{s.params['p'] + 1}"
        if body_name != host_body:
            return s
        return DerivationStep(path=s.path, production="Phalanx",
                               params={**s.params, "branch_digit_count": s.params["branch_digit_count"] - 1})

    return [_decrement_host(s) for s in kept]


_OPERATOR_FNS = {
    "resample_parameter": _op_resample_parameter,
    "perturb_parameter": _op_perturb_parameter,
    "regrow_subtree": _op_regrow_subtree,
    "insert_phalanx": _op_insert_phalanx,
    "delete_phalanx": _op_delete_phalanx,
    "add_digit": _op_add_digit,
    "remove_digit": _op_remove_digit,
    "step_axis": _op_step_axis,
    "step_limits": _op_step_limits,
    "step_mount": _op_step_mount,
    "step_length": _op_perturb_parameter,
    "step_coupling": _op_step_coupling,
    "step_root_length": _op_step_root_length,
    "step_radius": _op_step_radius,
    "step_bend_rpy": _op_step_bend_rpy,
    "step_bend_offset": _op_step_bend_offset,
    "add_minimal_digit": _op_add_minimal_digit,
    "remove_digit_minimal": _op_remove_digit_minimal,
    "add_palm_body": _op_add_palm_body,
    "remove_palm_body": _op_remove_palm_body,
    "remove_palm_body_empty": _op_remove_palm_body_empty,
    "toggle_palm_joint": _op_toggle_palm_joint,
    "add_branch_digit": _op_add_branch_digit,
    "remove_branch_digit": _op_remove_branch_digit,
}

# --------------------------------------------------------------------------
# EVOLUTION_OPERATORS (grammar 0.5, iteration B / balanced-grammar-synthesis
# .md section 3's operator table): the exact-inverse operator pool -- five
# growth/shrink pairs (digit, phalanx, palm body, palm joint -- self-inverse
# -- branch) plus every small-step operator. Each pair's growth move draws
# new material from ``dist.insertion`` when set (``_growth_dist``, see
# above); each pair's shrink move accepts an optional ``target`` uid so a
# caller can undo a specific growth application precisely (see
# ``apply_operator`` below and the reversibility test in
# ``experiments/e12_balance.py`` / ``grammar_bench/tests/test_grammar05_b.py``).
# Deliberately excludes ``regrow_subtree``/full-size ``add_digit``/
# ``remove_digit``/``resample_parameter`` (no exact inverse, large jumps --
# see balanced-grammar-synthesis.md section 3's "dropped from the default
# pool"). I18 fix 2: the palm-body pair uses ``remove_palm_body_empty`` (an
# EXACT inverse of ``add_palm_body`` -- it only ever removes a leaf with no
# children, exactly what ``add_palm_body`` produces), not the general
# ``remove_palm_body`` (which accepts any body and approximately
# re-attaches its children) -- that general operator stays defined and in
# ``_OPERATOR_FNS``/``TARGETABLE_OPERATORS`` as a larger move available
# OUTSIDE this pool.
# --------------------------------------------------------------------------

EVOLUTION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("add_minimal_digit", "remove_digit_minimal"),
    ("insert_phalanx", "delete_phalanx"),
    ("add_palm_body", "remove_palm_body_empty"),
    ("toggle_palm_joint", "toggle_palm_joint"),
    ("add_branch_digit", "remove_branch_digit"),
)

INVERSE_OF: Dict[str, str] = {}
for _growth, _shrink in EVOLUTION_PAIRS:
    INVERSE_OF[_growth] = _shrink
    INVERSE_OF[_shrink] = _growth
del _growth, _shrink

EVOLUTION_OPERATORS: Tuple[str, ...] = (
    "add_minimal_digit", "remove_digit_minimal",
    "insert_phalanx", "delete_phalanx",
    "add_palm_body", "remove_palm_body_empty",
    "toggle_palm_joint",
    "add_branch_digit", "remove_branch_digit",
) + tuple(SMALL_STEP_OPERATORS)
assert len(EVOLUTION_OPERATORS) == len(set(EVOLUTION_OPERATORS)), "EVOLUTION_OPERATORS has duplicates"

# Shrink operators (and self-inverse ``toggle_palm_joint``) that accept an
# optional ``target`` uid, per ``_OPERATOR_FNS`` above.
TARGETABLE_OPERATORS = frozenset({
    "remove_digit_minimal", "delete_phalanx", "remove_palm_body", "remove_palm_body_empty",
    "toggle_palm_joint", "remove_branch_digit",
})


def apply_operator(derivation: Derivation, rng: np.random.Generator, dist: Distribution,
                    operator: str, target: Optional[int] = None) -> Optional[Derivation]:
    """Apply ``operator`` exactly ONCE (unlike ``vary``: no 32-attempt retry
    loop, no re-drawing a different operator on failure) to ``derivation``,
    optionally passing ``target`` through to the underlying ``_op_*``
    function when ``operator in TARGETABLE_OPERATORS`` (a uid selecting
    which digit/phalanx/palm-body to act on, instead of drawing uniformly
    -- see each targetable ``_op_*``'s own docstring). Returns ``None`` when
    the operator has no valid application (the underlying function returned
    ``None``, or the candidate failed to derive/validate) -- the caller
    decides what "not applicable" means for its own accounting (E1's
    ``VariationImpossible`` convention does not apply here, since this
    function never retries)."""
    if operator not in _OPERATOR_FNS:
        raise ValueError(f"unknown vary operator {operator!r}")
    fn = _OPERATOR_FNS[operator]
    if operator in TARGETABLE_OPERATORS:
        candidate_steps = fn(rng, dist, derivation, target=target)
    else:
        candidate_steps = fn(rng, dist, derivation)
    if candidate_steps is None:
        return None
    if tuple(candidate_steps) == derivation.steps:
        return None
    candidate = Derivation(
        seed=derivation.seed, grammar_version=derivation.grammar_version,
        steps=tuple(candidate_steps), lineage=derivation.lineage + ((operator, derivation.seed),),
    )
    try:
        derive(candidate)
    except ModelError:
        return None
    return candidate


def vary(derivation: Derivation, rng: np.random.Generator, dist: Distribution = DEFAULT_DISTRIBUTION,
         operator: Optional[str] = None, operators: Optional[Sequence[str]] = None) -> Derivation:
    """Apply ``operator`` (random from ``operators`` if given, else random
    from ``OPERATORS`` -- unchanged default behaviour -- if both are
    omitted) to ``derivation``, retrying up to 32 times with fresh
    randomness until the result derives to a valid model and differs from
    the parent. Raises ``VariationImpossible`` if no valid application is
    found. Passing ``operators=SMALL_STEP_OPERATORS`` (or any other
    explicit operator/pool) is the only way to reach an operator outside
    ``OPERATORS``; nothing here changes what a bare ``vary(derivation,
    rng, dist)`` call does."""
    if operator is not None:
        op = operator
    else:
        pool = operators if operators is not None else OPERATORS
        op = pool[int(rng.integers(0, len(pool)))]
    if op not in _OPERATOR_FNS:
        raise ValueError(f"unknown vary operator {op!r}")
    fn = _OPERATOR_FNS[op]
    for _ in range(32):
        candidate_steps = fn(rng, dist, derivation)
        if candidate_steps is None:
            continue
        if tuple(candidate_steps) == derivation.steps:
            continue
        candidate = Derivation(
            seed=derivation.seed, grammar_version=derivation.grammar_version,
            steps=tuple(candidate_steps),
            lineage=derivation.lineage + ((op, derivation.seed),),
        )
        try:
            derive(candidate)
        except ModelError:
            continue
        return candidate
    raise VariationImpossible(f"could not apply operator {op!r} to derivation after 32 attempts")


def vary_tracked(derivation: Derivation, rng: np.random.Generator, dist: Distribution,
                  operators: Sequence[str]) -> Tuple[Optional[Derivation], str]:
    """Like ``vary(derivation, rng, dist, operators=operators)`` -- one
    operator drawn uniformly from ``operators``, retried up to 32 times via
    ``apply_operator`` (exactly ``vary``'s own per-attempt logic, since
    ``apply_operator`` runs the same ``fn`` / no-op / ``ModelError`` checks)
    -- but returns ``(candidate_or_None, op)`` instead of raising
    ``VariationImpossible`` on total failure, so a caller (E12's per-pair
    growth/shrink applicability-rate reporting, grammar 0.5 I18 fix 4) can
    tally which operator was drawn and whether IT was applicable, without
    duplicating ``vary``'s algorithm."""
    op = operators[int(rng.integers(0, len(operators)))]
    for _ in range(32):
        candidate = apply_operator(derivation, rng, dist, op)
        if candidate is not None:
            return candidate, op
    return None, op
