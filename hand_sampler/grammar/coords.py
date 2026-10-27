"""Independent coordinates, coupling expansion, admissible box and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np

from .kinematics import AffineCoupling, KinematicModel, MOVABLE_TYPES

CONTINUOUS_SAMPLE_RANGE = (-2.0 * np.pi, 2.0 * np.pi)


@dataclass(frozen=True)
class LimitConflict:
    dependent: str
    image: Tuple[float, float]
    limit: Tuple[float, float]
    excess: float


def movable_joints(model: KinematicModel) -> List[str]:
    return [j.name for j in model.joints if j.type in MOVABLE_TYPES]


def _dependents(model: KinematicModel) -> set:
    return {c.dependent for c in model.couplings}


def independent_joints(model: KinematicModel) -> List[str]:
    if model.independent:
        return list(model.independent)
    deps = _dependents(model)
    return [j for j in movable_joints(model) if j not in deps]


def _coupling_by_dependent(model: KinematicModel) -> Dict[str, AffineCoupling]:
    return {c.dependent: c for c in model.couplings}


def _resolve_to_root(model: KinematicModel, joint_name: str, coupling_map=None):
    """Compose the affine chain from ``joint_name`` back to its ultimate independent
    source. Returns (multiplier, offset, root_name) such that
    q[joint_name] = multiplier * q[root_name] + offset."""
    if coupling_map is None:
        coupling_map = _coupling_by_dependent(model)
    M, B = 1.0, 0.0
    cur = joint_name
    seen = set()
    while cur in coupling_map:
        if cur in seen:
            raise ValueError(f"coupling cycle detected at {cur!r}")
        seen.add(cur)
        c = coupling_map[cur]
        M, B = M * c.multiplier, M * c.offset + B
        cur = c.source
    return M, B, cur


def q_from_u(model: KinematicModel, u: Mapping[str, float]) -> Dict[str, float]:
    indep = set(independent_joints(model))
    given = set(u)
    if given != indep:
        missing = indep - given
        extra = given - indep
        raise ValueError(
            f"u must contain exactly the independent joints; missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    q: Dict[str, float] = {name: float(val) for name, val in u.items()}
    coupling_map = _coupling_by_dependent(model)
    # Evaluate couplings in topological order (dependent depends on source).
    remaining = dict(coupling_map)
    progress = True
    while remaining and progress:
        progress = False
        for dep, c in list(remaining.items()):
            if c.source in q:
                q[dep] = c.multiplier * q[c.source] + c.offset
                del remaining[dep]
                progress = True
    if remaining:
        raise ValueError(f"unresolved couplings (cycle?): {sorted(remaining)}")
    return q


def admissible_box(model: KinematicModel):
    """Return (box, conflicts).

    ``box``: dict[independent joint name] -> (lo, hi), the intersection of the
    joint's own declared limits (or [-2pi, 2pi] default range if continuous) with
    the pre-images of every dependent's declared limits, following affine
    coupling chains (single-source affine, exact).

    ``conflicts``: list of LimitConflict, one per coupling whose dependent's own
    declared limit is strictly tighter than the image obtained by mapping the
    ultimate independent source's own declared limits through the composed
    affine chain.
    """
    indep = independent_joints(model)
    joint_by_name = {j.name: j for j in model.joints}
    coupling_map = _coupling_by_dependent(model)

    box: Dict[str, Tuple[float, float]] = {}
    for name in indep:
        j = joint_by_name[name]
        if j.type == "continuous":
            box[name] = CONTINUOUS_SAMPLE_RANGE
        else:
            box[name] = tuple(j.limits)

    conflicts: List[LimitConflict] = []

    for c in model.couplings:
        dep_joint = joint_by_name[c.dependent]
        M, B, root = _resolve_to_root(model, c.dependent, coupling_map)
        root_joint = joint_by_name[root]
        if root_joint.type == "continuous":
            root_lo, root_hi = CONTINUOUS_SAMPLE_RANGE
        else:
            root_lo, root_hi = root_joint.limits

        if M >= 0:
            img_lo, img_hi = M * root_lo + B, M * root_hi + B
        else:
            img_lo, img_hi = M * root_hi + B, M * root_lo + B

        if dep_joint.limits is not None:
            dep_lo, dep_hi = dep_joint.limits
            excess_lo = max(0.0, dep_lo - img_lo)
            excess_hi = max(0.0, img_hi - dep_hi)
            excess = excess_lo + excess_hi
            if excess > 0.0:
                conflicts.append(
                    LimitConflict(
                        dependent=c.dependent,
                        image=(img_lo, img_hi),
                        limit=(dep_lo, dep_hi),
                        excess=excess,
                    )
                )

            # Shrink the root's box by the pre-image of the dependent's own limits.
            if M > 0:
                pre_lo, pre_hi = (dep_lo - B) / M, (dep_hi - B) / M
            elif M < 0:
                pre_lo, pre_hi = (dep_hi - B) / M, (dep_lo - B) / M
            else:
                pre_lo, pre_hi = box[root][0], box[root][1]
            if root in box:
                blo, bhi = box[root]
                box[root] = (max(blo, pre_lo), min(bhi, pre_hi))

    return box, conflicts


def check_limits(model: KinematicModel, q: Mapping[str, float]) -> List[str]:
    violations = []
    for j in model.joints:
        if j.limits is None:
            continue
        val = float(q.get(j.name, 0.0))
        lo, hi = j.limits
        if not (lo <= val <= hi):
            violations.append(
                f"joint {j.name!r} value {val!r} outside declared limits ({lo!r}, {hi!r})"
            )
    return violations


def _shrink(lo: float, hi: float, rel: float) -> Tuple[float, float]:
    width = hi - lo
    amt = width * rel
    new_lo, new_hi = lo + amt, hi - amt
    if new_lo > new_hi:
        mid = (lo + hi) / 2.0
        return mid, mid
    return new_lo, new_hi


def sample_configurations(model: KinematicModel, n: int, seed: int, box_shrink_rel: float = 1e-9):
    """Return a list of u-configurations (dict: independent joint -> value):
    ``n`` random uniform samples inside the shrunk admissible box, plus extremal
    configurations (all-lower, all-upper, zero if inside the box, and 4 mixed
    corners chosen deterministically)."""
    box, _conflicts = admissible_box(model)
    names = sorted(box)  # deterministic order
    shrunk = {name: _shrink(*box[name], box_shrink_rel) for name in names}

    rng = np.random.default_rng(seed)
    configs: List[Dict[str, float]] = []
    for _ in range(n):
        u = {name: float(rng.uniform(shrunk[name][0], shrunk[name][1])) for name in names}
        configs.append(u)

    # Extremals.
    all_lower = {name: shrunk[name][0] for name in names}
    all_upper = {name: shrunk[name][1] for name in names}
    configs.append(all_lower)
    configs.append(all_upper)

    if all(shrunk[name][0] <= 0.0 <= shrunk[name][1] for name in names):
        configs.append({name: 0.0 for name in names})

    for k in range(4):
        corner = {}
        for idx, name in enumerate(names):
            lo, hi = shrunk[name]
            corner[name] = hi if ((idx + k) % 2 == 0) else lo
        configs.append(corner)

    return configs
