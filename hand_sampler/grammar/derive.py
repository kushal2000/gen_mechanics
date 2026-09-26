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
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .coords import CONTINUOUS_SAMPLE_RANGE
from .distributions import (
    DEFAULT_DISTRIBUTION,
    Distribution,
    sample_axis,
    sample_grid_angle_rad,
    sample_grid_length_m,
    sample_module,
    sample_palm_joint_limits_rad,
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


def _sample_phalanx(rng, dist: Distribution, steps: List[DerivationStep], digit_id: str, p: int,
                     depth: int, is_last: bool) -> None:
    module = sample_module(rng, dist, p, _revolute_source_indices(steps, digit_id, p))
    length = sample_grid_length_m(rng, dist.link_length_range_m, dist.link_length_grid_m)
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
    steps.append(DerivationStep(path=f"digit/{digit_id}/phalanx/{p}", production="Phalanx", params={
        "digit_id": digit_id, "p": p, "module": module, "length": length,
        "branch_digit_count": branch_digit_count,
    }))
    if branch_digit_count:
        # The branch mounts on THIS phalanx's own body (distal to its
        # joint), never on a palm body, so that body genuinely gets >= 2
        # child joints (its own next phalanx plus one per branch digit).
        phalanx_body = f"d{digit_id}p{p + 1}"
        for b in range(branch_digit_count):
            sub_id = f"{digit_id}p{p + 1}b{b}"
            _emit_digit(rng, dist, steps, sub_id, [phalanx_body], top_level=False, depth=depth + 1)


def _emit_digit(rng, dist: Distribution, steps: List[DerivationStep], digit_id: str,
                 mount_bodies: List[str], top_level: bool, depth: int) -> None:
    mount = mount_bodies[int(rng.integers(0, len(mount_bodies)))]
    mount_frac = float(dist.mount_frac_choices[int(rng.integers(0, len(dist.mount_frac_choices)))])
    mount_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
    phalanx_count = int(rng.integers(dist.phalanx_count_range[0], dist.phalanx_count_range[1] + 1))
    steps.append(DerivationStep(path=f"digit/{digit_id}", production="Digit", params={
        "digit_id": digit_id, "mount": mount, "mount_frac": mount_frac, "mount_rpy": mount_rpy,
        "phalanx_count": phalanx_count, "top_level": top_level, "depth": depth,
    }))
    for p in range(phalanx_count):
        _sample_phalanx(rng, dist, steps, digit_id, p, depth, is_last=(p == phalanx_count - 1))


def _sample_digit(rng, dist: Distribution, steps: List[DerivationStep], next_id: List[int],
                   mount_bodies: List[str]) -> None:
    """Sample a fresh *top-level* digit (id is the next 1-based integer)."""
    digit_id = str(next_id[0])
    next_id[0] += 1
    _emit_digit(rng, dist, steps, digit_id, mount_bodies, top_level=True, depth=0)


def sample_derivation(rng_or_seed, dist: Distribution = DEFAULT_DISTRIBUTION) -> Derivation:
    seed, rng = _coerce_rng(rng_or_seed)
    steps: List[DerivationStep] = []

    digit_count = int(rng.integers(dist.digit_count_range[0], dist.digit_count_range[1] + 1))
    # Additional palm bodies beyond the root -- the root is always a palm
    # body with a real segment of its own (root_length below), so a hand
    # never lacks a palm even when palm_body_count == 0.
    palm_body_count = int(rng.integers(dist.palm_body_count_range[0], dist.palm_body_count_range[1] + 1))
    root_length = sample_grid_length_m(rng, dist.palm_length_range_m, dist.link_length_grid_m)
    steps.append(DerivationStep(path="hand", production="Hand", params={
        "digit_count": digit_count, "palm_body_count": palm_body_count, "root_length": root_length,
    }))

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
        steps.append(DerivationStep(path=f"palm/{i}", production="PalmBody", params={
            "name": name, "parent": parent, "mount_frac": mount_frac, "length": length,
            "direction_rpy": direction_rpy, "has_joint": has_joint, "axis": axis, "limits": limits,
        }))
        palm_names.append(name)

    mount_bodies = ["root"] + palm_names
    next_id = [1]
    for _ in range(digit_count):
        _sample_digit(rng, dist, steps, next_id, mount_bodies)

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


def derive(derivation: Derivation) -> KinematicModel:
    issues = validate_derivation(derivation)
    if issues:
        raise DerivationError(issues)

    steps_by_path = {s.path: s for s in derivation.steps}
    hand = steps_by_path["hand"].params
    palm_body_count = hand["palm_body_count"]
    root_length = hand["root_length"]

    # The root always owns a real segment (see rules.py's RootProduction /
    # convention-change note): a hand always has a palm, so body_length for
    # "root" is never 0, and the root gets its own "<body>_tip" frame just
    # like every other segment-owning body.
    bodies: List[Body] = [Body(name="root", palm=True)]
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
        bodies.append(Body(name=name, palm=True))
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
            if pi == 0:
                origin = Pose(xyz=base_xyz, rpy=base_rpy)
            else:
                origin = Pose(xyz=(0.0, 0.0, prev_len), rpy=(0.0, 0.0, 0.0))

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
            bodies.append(Body(name=body_name))
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
    _emit_digit(rng, dist, new_steps, digit_id, mount_bodies, top_level, depth)
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
    new_phalanx_list.append(DerivationStep(path=f"digit/{digit_id}/phalanx/{ins_p}", production="Phalanx", params={
        "digit_id": digit_id, "p": ins_p, "module": module, "length": length, "branch_digit_count": 0,
    }))

    new_dstep = DerivationStep(path=dstep.path, production="Digit",
                                params={**dstep.params, "phalanx_count": old_count + 1})
    result = non_digit_others + [new_dstep] + new_phalanx_list
    return _rename_branch_mounts(result, digit_id, renumber)


def _op_delete_phalanx(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    # A phalanx that hosts a branch is refused as a deletion target: deleting
    # it would orphan its sub-digit(s)' mount, so it is never offered to
    # ``del_p`` (a digit is only a candidate if it has a branch-free phalanx
    # to delete and >1 phalanx overall). Deleting any phalanx except the
    # digit's *current last one* never changes which phalanx is last, so it
    # is always safe. Deleting the last one promotes the previous phalanx to
    # "last" -- refused too when that phalanx hosts exactly 1 branch digit
    # (valid only for a non-last phalanx, which already has a next-phalanx
    # child; as the new last phalanx it would drop to a single child).
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
            p_idx for p_idx, pp in phalanx_by_p.items()
            if pp.params["branch_digit_count"] == 0 and (p_idx != last_idx or last_deletion_safe)
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
    _sample_digit(rng, dist, new_steps, next_id, mount_bodies)
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


_OPERATOR_FNS = {
    "resample_parameter": _op_resample_parameter,
    "perturb_parameter": _op_perturb_parameter,
    "regrow_subtree": _op_regrow_subtree,
    "insert_phalanx": _op_insert_phalanx,
    "delete_phalanx": _op_delete_phalanx,
    "add_digit": _op_add_digit,
    "remove_digit": _op_remove_digit,
}


def vary(derivation: Derivation, rng: np.random.Generator, dist: Distribution = DEFAULT_DISTRIBUTION,
         operator: Optional[str] = None) -> Derivation:
    """Apply ``operator`` (one of ``OPERATORS``; random if omitted) to
    ``derivation``, retrying up to 32 times with fresh randomness until the
    result derives to a valid model and differs from the parent. Raises
    ``VariationImpossible`` if no valid application is found."""
    op = operator if operator is not None else OPERATORS[int(rng.integers(0, len(OPERATORS)))]
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
