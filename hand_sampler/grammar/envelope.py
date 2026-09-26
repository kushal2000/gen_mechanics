"""Hard structural envelope for the E5b simulator-matched evolution loop.

The simulator this loop is meant to match cannot actuate an arbitrary
grammar-sampled hand: it has a bounded number of fingers, a bounded number
of joints per finger, no palm degrees of freedom, and no in-digit branching.
``fits_envelope`` is a pure, cheap structural check (no forward kinematics)
against a derived ``KinematicModel`` -- ``e5b_evolve_sim.py`` REJECTS any
``vary`` proposal that fails it (see that module's docstring for the
retry/clone policy), rather than silently repairing or ignoring the
violation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .kinematics import KinematicModel


def fits_envelope(
    model: KinematicModel,
    max_digits: int = 5,
    max_joints_per_digit: int = 6,
    allow_palm_joints: bool = False,
    allow_branches: bool = False,
) -> Tuple[bool, List[str]]:
    """``(ok, reasons)``: ``ok`` is ``True`` iff every check below passes;
    ``reasons`` lists every VIOLATED check (empty iff ``ok``), so a caller
    can report which operator/violation combination is actually driving a
    pool's rejection rate.

    - ``digit_count`` (see ``phenodist._digit_count``'s definition: a joint
      whose parent is palm-flagged and whose child is not) ``<= max_digits``.
    - every top-level digit's OWN subtree (every joint reachable, by
      following joint parent->child edges, from that digit's root joint
      downward -- includes any branch joints hanging off it) has
      ``<= max_joints_per_digit`` joints.
    - ``allow_palm_joints=False`` (the simulator loop's default): no joint
      may have BOTH endpoints palm-flagged (a joint between two palm
      bodies -- the root itself has no parent joint, so this only ever
      matches an ADDITIONAL palm body's own joint).
    - ``allow_branches=False`` (the simulator loop's default): no non-palm
      body may be the ``parent`` of >= 2 joints (the same "real in-digit
      branching" structural measure ``test_acceptance_grammar.py``'s
      support audit uses).
    """
    reasons: List[str] = []
    palm_names = {b.name for b in model.bodies if b.palm}
    non_palm_names = {b.name for b in model.bodies if not b.palm}

    children_of: Dict[str, List] = {}
    child_count: Dict[str, int] = {}
    for j in model.joints:
        children_of.setdefault(j.parent, []).append(j)
        child_count[j.parent] = child_count.get(j.parent, 0) + 1

    digit_root_joints = [j for j in model.joints if j.parent in palm_names and j.child in non_palm_names]
    n_digits = len(digit_root_joints)
    if n_digits > max_digits:
        reasons.append(f"digit_count {n_digits} exceeds max_digits {max_digits}")

    if not allow_palm_joints:
        palm_joints = [j for j in model.joints if j.parent in palm_names and j.child in palm_names]
        if palm_joints:
            reasons.append(
                f"{len(palm_joints)} palm joint(s) present ({sorted(j.name for j in palm_joints)}) "
                "but allow_palm_joints=False"
            )

    if not allow_branches:
        branchers = sorted(name for name, n in child_count.items() if name not in palm_names and n >= 2)
        if branchers:
            reasons.append(f"{len(branchers)} branching non-palm body(ies) present ({branchers}) but allow_branches=False")  # noqa: E501

    for root_j in digit_root_joints:
        count = 1  # root_j itself.
        stack = [root_j.child]
        while stack:
            body = stack.pop()
            for j in children_of.get(body, []):
                count += 1
                stack.append(j.child)
        if count > max_joints_per_digit:
            reasons.append(
                f"digit rooted at {root_j.child!r} has {count} joints, exceeds "
                f"max_joints_per_digit {max_joints_per_digit}"
            )

    return (len(reasons) == 0, reasons)
