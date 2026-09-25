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
from .fk import rpy_to_matrix
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


def _sample_phalanx(rng, dist: Distribution, steps: List[DerivationStep], gid: int, p: int,
                     allow_branch: bool, mount_bodies: List[str], gid_counter: List[int]) -> None:
    module = sample_module(rng, dist, p)
    length = sample_grid_length_m(rng, dist.link_length_range_m, dist.link_length_grid_m)
    branch_digit_count = 0
    if allow_branch and float(rng.random()) < dist.branch_probability:
        branch_digit_count = int(rng.integers(1, dist.max_branch_digits + 1))
    steps.append(DerivationStep(path=f"digit/{gid}/phalanx/{p}", production="Phalanx", params={
        "gid": gid, "p": p, "module": module, "length": length,
        "branch_digit_count": branch_digit_count,
    }))
    for _b in range(branch_digit_count):
        _sample_digit(rng, dist, steps, gid_counter, mount_bodies, top_level=False, allow_branch=False)


def _emit_digit(rng, dist: Distribution, steps: List[DerivationStep], gid: int,
                 mount_bodies: List[str], top_level: bool, gid_counter: List[int],
                 allow_branch: bool = True) -> None:
    mount = mount_bodies[int(rng.integers(0, len(mount_bodies)))]
    mount_frac = float(dist.mount_frac_choices[int(rng.integers(0, len(dist.mount_frac_choices)))])
    mount_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
    phalanx_count = int(rng.integers(dist.phalanx_count_range[0], dist.phalanx_count_range[1] + 1))
    steps.append(DerivationStep(path=f"digit/{gid}", production="Digit", params={
        "gid": gid, "mount": mount, "mount_frac": mount_frac, "mount_rpy": mount_rpy,
        "phalanx_count": phalanx_count, "top_level": top_level,
    }))
    for p in range(phalanx_count):
        _sample_phalanx(rng, dist, steps, gid, p, allow_branch, mount_bodies, gid_counter)


def _sample_digit(rng, dist: Distribution, steps: List[DerivationStep], gid_counter: List[int],
                   mount_bodies: List[str], top_level: bool, allow_branch: bool = True) -> None:
    gid = gid_counter[0]
    gid_counter[0] += 1
    _emit_digit(rng, dist, steps, gid, mount_bodies, top_level, gid_counter, allow_branch)


def sample_derivation(rng_or_seed, dist: Distribution = DEFAULT_DISTRIBUTION) -> Derivation:
    seed, rng = _coerce_rng(rng_or_seed)
    steps: List[DerivationStep] = []

    digit_count = int(rng.integers(dist.digit_count_range[0], dist.digit_count_range[1] + 1))
    palm_body_count = int(rng.integers(dist.palm_body_count_range[0], dist.palm_body_count_range[1] + 1))
    steps.append(DerivationStep(path="hand", production="Hand", params={
        "digit_count": digit_count, "palm_body_count": palm_body_count,
    }))

    parent = "root"
    palm_names: List[str] = []
    for i in range(palm_body_count):
        name = f"palm{i}"
        length = sample_grid_length_m(rng, dist.palm_length_range_m, dist.link_length_grid_m)
        direction_rpy = (sample_grid_angle_rad(rng), sample_grid_angle_rad(rng), sample_grid_angle_rad(rng))
        has_joint = bool(float(rng.random()) < dist.palm_joint_probability)
        axis = sample_axis(rng)
        limits: Optional[Tuple[float, float]] = None
        if has_joint:
            limits = sample_palm_joint_limits_rad(rng, dist)
        steps.append(DerivationStep(path=f"palm/{i}", production="PalmBody", params={
            "name": name, "parent": parent, "length": length, "direction_rpy": direction_rpy,
            "has_joint": has_joint, "axis": axis, "limits": limits,
        }))
        palm_names.append(name)
        parent = name

    mount_bodies = ["root"] + palm_names
    gid_counter = [0]
    for _ in range(digit_count):
        _sample_digit(rng, dist, steps, gid_counter, mount_bodies, top_level=True)

    return Derivation(seed=seed, grammar_version=GRAMMAR_VERSION, steps=tuple(steps))


# --------------------------------------------------------------------------
# derive: pure, deterministic reconstruction
# --------------------------------------------------------------------------


def derive(derivation: Derivation) -> KinematicModel:
    steps_by_path = {s.path: s for s in derivation.steps}
    hand = steps_by_path["hand"].params
    palm_body_count = hand["palm_body_count"]

    bodies: List[Body] = [Body(name="root", palm=True)]
    joints: List[Joint] = []
    frames: List[Frame] = []
    couplings: List[AffineCoupling] = []
    body_length: Dict[str, float] = {"root": 0.0}
    joints_by_name: Dict[str, Joint] = {}

    parent = "root"
    for i in range(palm_body_count):
        p = steps_by_path[f"palm/{i}"].params
        name = p["name"]
        length = p["length"]
        rpy = tuple(p["direction_rpy"])
        R = rpy_to_matrix(rpy)
        offset = R @ np.array([0.0, 0.0, length])
        jtype = "revolute" if p["has_joint"] else "fixed"
        axis = tuple(p["axis"]) if p["has_joint"] else (1.0, 0.0, 0.0)
        limits = tuple(p["limits"]) if p["has_joint"] else None
        j = Joint(
            name=f"{name}_j", type=jtype, parent=parent, child=name,
            origin=Pose(xyz=tuple(float(v) for v in offset), rpy=rpy),
            axis=axis, limits=limits,
        )
        joints.append(j)
        joints_by_name[j.name] = j
        bodies.append(Body(name=name, palm=True))
        frames.append(Frame(name=f"{name}_tip", body=name, pose=Pose(xyz=(0.0, 0.0, length))))
        body_length[name] = length
        parent = name

    for s in derivation.steps:
        if s.production != "Digit":
            continue
        p = s.params
        gid = p["gid"]
        mount = p["mount"]
        mount_len = body_length.get(mount, 0.0)
        Rm = rpy_to_matrix(tuple(p["mount_rpy"]))
        off = Rm @ np.array([0.0, 0.0, p["mount_frac"] * mount_len])
        base_xyz = tuple(float(v) for v in off)
        base_rpy = tuple(p["mount_rpy"])

        prev_body = mount
        prev_len = 0.0
        phalanx_joint_names: Dict[int, str] = {}
        for pi in range(p["phalanx_count"]):
            pp = steps_by_path[f"digit/{gid}/phalanx/{pi}"].params
            body_name = f"d{gid + 1}p{pi + 1}"
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

    model = KinematicModel(
        name="grammar_hand", root="root", bodies=tuple(bodies), joints=tuple(joints),
        frames=tuple(frames), couplings=tuple(couplings),
    )
    validate(model)
    return model


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
    }


def derivation_from_dict(d: Dict[str, Any]) -> Derivation:
    if d.get("schema") != DERIVATION_SCHEMA:
        raise ValueError(f"unsupported schema {d.get('schema')!r}, expected {DERIVATION_SCHEMA!r}")
    return Derivation(
        seed=d["seed"],
        grammar_version=d["grammar_version"],
        steps=tuple(_step_from_dict(s) for s in d["steps"]),
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
        module = sample_module(rng, dist, p["p"])
        length = sample_grid_length_m(rng, dist.link_length_range_m, dist.link_length_grid_m)
        p.update({"module": module, "length": length})
        steps[idx] = DerivationStep(path=s.path, production=s.production, params=p)
        return steps
    if s.production == "Digit":
        p = dict(s.params)
        mount_bodies = _mount_bodies_from_steps(steps)
        p["mount"] = mount_bodies[int(rng.integers(0, len(mount_bodies)))]
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
    gid = s.params["gid"]
    top_level = s.params["top_level"]
    mount_bodies = _mount_bodies_from_steps(steps)

    prefix = f"digit/{gid}/phalanx/"
    kept = [st for st in steps if not st.path.startswith(prefix) and st.path != s.path]

    max_gid = max(st.params["gid"] for st in steps if st.production == "Digit")
    gid_counter = [max_gid + 1]
    new_steps: List[DerivationStep] = []
    _emit_digit(rng, dist, new_steps, gid, mount_bodies, top_level, gid_counter)
    return kept + new_steps


def _op_insert_phalanx(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    digit_idxs = [i for i, s in enumerate(steps)
                  if s.production == "Digit" and s.params["phalanx_count"] < dist.phalanx_count_range[1]]
    if not digit_idxs:
        return None
    idx = digit_idxs[int(rng.integers(0, len(digit_idxs)))]
    dstep = steps[idx]
    gid = dstep.params["gid"]
    old_count = dstep.params["phalanx_count"]
    ins_p = int(rng.integers(0, old_count + 1))

    others = [st for st in steps if st is not dstep]
    phalanx_steps = {st.params["p"]: st for st in others if st.production == "Phalanx" and st.params["gid"] == gid}
    non_digit_others = [st for st in others if not (st.production == "Phalanx" and st.params["gid"] == gid)]

    new_phalanx_list: List[DerivationStep] = []
    for old_p in range(old_count):
        new_p = old_p if old_p < ins_p else old_p + 1
        st = phalanx_steps[old_p]
        params = dict(st.params)
        mod = dict(params["module"])
        if mod.get("kind") == "Coupled":
            sp = mod["source_p"]
            mod["source_p"] = sp if sp < ins_p else sp + 1
        params["module"] = mod
        params["p"] = new_p
        new_phalanx_list.append(DerivationStep(path=f"digit/{gid}/phalanx/{new_p}", production="Phalanx", params=params))

    module = sample_module(rng, dist, ins_p)
    length = sample_grid_length_m(rng, dist.link_length_range_m, dist.link_length_grid_m)
    new_phalanx_list.append(DerivationStep(path=f"digit/{gid}/phalanx/{ins_p}", production="Phalanx", params={
        "gid": gid, "p": ins_p, "module": module, "length": length, "branch_digit_count": 0,
    }))

    new_dstep = DerivationStep(path=dstep.path, production="Digit",
                                params={**dstep.params, "phalanx_count": old_count + 1})
    return non_digit_others + [new_dstep] + new_phalanx_list


def _op_delete_phalanx(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    digit_idxs = [i for i, s in enumerate(steps) if s.production == "Digit" and s.params["phalanx_count"] > 1]
    if not digit_idxs:
        return None
    idx = digit_idxs[int(rng.integers(0, len(digit_idxs)))]
    dstep = steps[idx]
    gid = dstep.params["gid"]
    old_count = dstep.params["phalanx_count"]
    del_p = int(rng.integers(0, old_count))

    others = [st for st in steps if st is not dstep]
    phalanx_steps = {st.params["p"]: st for st in others if st.production == "Phalanx" and st.params["gid"] == gid}
    non_digit_others = [st for st in others if not (st.production == "Phalanx" and st.params["gid"] == gid)]

    new_phalanx_list: List[DerivationStep] = []
    for old_p in range(old_count):
        if old_p == del_p:
            continue
        new_p = old_p if old_p < del_p else old_p - 1
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
        new_phalanx_list.append(DerivationStep(path=f"digit/{gid}/phalanx/{new_p}", production="Phalanx", params=params))

    new_dstep = DerivationStep(path=dstep.path, production="Digit",
                                params={**dstep.params, "phalanx_count": old_count - 1})
    return non_digit_others + [new_dstep] + new_phalanx_list


def _op_add_digit(rng, dist: Distribution, derivation: Derivation) -> Optional[List[DerivationStep]]:
    steps = list(derivation.steps)
    hand_idx = next(i for i, s in enumerate(steps) if s.path == "hand")
    hand_params = steps[hand_idx].params
    if hand_params["digit_count"] >= dist.digit_count_range[1]:
        return None
    mount_bodies = _mount_bodies_from_steps(steps)
    max_gid = max((s.params["gid"] for s in steps if s.production == "Digit"), default=-1)
    gid_counter = [max_gid + 1]
    new_steps: List[DerivationStep] = []
    _sample_digit(rng, dist, new_steps, gid_counter, mount_bodies, top_level=True)
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
    top_gids = [s.params["gid"] for s in steps if s.production == "Digit" and s.params.get("top_level")]
    if not top_gids:
        return None
    gid = top_gids[int(rng.integers(0, len(top_gids)))]
    prefix = f"digit/{gid}/"
    kept = [s for s in steps if s.path != f"digit/{gid}" and not s.path.startswith(prefix)]
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
        candidate = Derivation(seed=derivation.seed, grammar_version=derivation.grammar_version,
                                steps=tuple(candidate_steps))
        try:
            derive(candidate)
        except ModelError:
            continue
        if candidate == derivation:
            continue
        return candidate
    raise VariationImpossible(f"could not apply operator {op!r} to derivation after 32 attempts")
