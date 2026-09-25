"""Typed grammar productions for the hand-kinematics grammar (iteration 3).

These dataclasses are the typed schema for what ``distributions.py`` samples
and ``derive.py`` interprets into a ``KinematicModel``. They document the
grammar in the design note's own vocabulary:

    Hand -> Root PalmBody* Digit*
    PalmBody(parent: a palm body) -> segment(length, direction) + optional PalmJoint(R, axis, limits)
    Digit(mount: a palm body, mount pose) -> Phalanx+
    Phalanx -> Module + segment(length) + optional Branch(Digit*)
    Module in {R(axis, limits), C(axis), P(axis, limits),
               Coupled(source, multiplier, offset)}

Convention (revised in iteration 3b, superseding the prior "branches mount
on a palm body" choice): ``Branch(Digit*)`` reuses the same ``Digit``
production as ``Hand``'s top-level digits, but a branch digit's ``mount`` is
*the branching phalanx's own body* (the body distal to that phalanx's
joint) -- not a palm body -- so that body genuinely acquires two or more
child joints (the next phalanx in its own chain, plus one joint per branch
digit). This is what makes "in-digit branching" a real structural fact,
checked directly by the benchmark against the derived ``KinematicModel``
rather than inferred from a derivation-step flag. A top-level ``Digit``
still always mounts on a palm body. Branches may themselves branch, up to
``Distribution.max_branch_depth`` (default 2, counted from the top-level
digit at depth 0).

Digit identity is a hierarchical path, not a flat counter: a top-level
digit's id is its 1-based index as a string (``"1"``, ``"2"``, ...); a
branch digit spawned from phalanx ``p`` (0-based) of host digit ``H`` at
branch slot ``b`` (0-based, when a phalanx spawns more than one sub-digit)
has id ``f"{H}p{p+1}b{b}"``. Body/joint naming is deterministic from that
id: the root is ``"root"``; palm bodies are ``"palm{i}"`` with joint
``"palm{i}_j"`` (``i`` 0-based, each parented to the root or to an
already-created palm body, forming a tree rather than a fixed chain); a
digit's phalanx bodies are ``"d{digit_id}p{p+1}"`` with joint
``"d{digit_id}p{p+1}_j"`` (``p`` the 0-based phalanx index within that
digit) -- e.g. phalanx 1 (0-based) of the branch digit above is body
``"d1p2b0p1"``. A digit id, once assigned, is never reused or renamed even
if a later ``vary`` operator renumbers its host's phalanx indices (see
``derive.py``'s ``_rename_branch_mounts``); it stays a valid, globally
unique identifier, it just may no longer literally spell out the host's
*current* phalanx index.

Segment length/direction for the later geometry step is not a new ``Body``
field: every body produced by a ``segment(...)`` production gets a
``Frame`` named ``"<body>_tip"`` whose pose is ``(0, 0, length)`` in the
body's own frame -- the geometry step reads that frame instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

GRAMMAR_VERSION = "0.1"


@dataclass(frozen=True)
class RootProduction:
    """``Root``: the fixed hand-frame body. Always body name ``"root"``,
    always ``Body.palm == True``."""


@dataclass(frozen=True)
class PalmBodyProduction:
    """``PalmBody(parent) -> segment(length, direction) + optional PalmJoint(R, axis, limits)``.
    ``parent`` is sampled among the root and every already-created palm body,
    so palm bodies fan out into a tree rather than a fixed chain."""

    parent: str
    length: float
    direction_rpy: Tuple[float, float, float]
    has_joint: bool
    axis: Tuple[float, float, float]
    limits: Optional[Tuple[float, float]]


@dataclass(frozen=True)
class DigitProduction:
    """``Digit(mount, mount_pose) -> Phalanx+``. ``mount`` is a palm body for
    a top-level digit, or the host phalanx's own body for a branch digit."""

    digit_id: str
    mount: str
    mount_frac: float
    mount_rpy: Tuple[float, float, float]
    phalanx_count: int
    top_level: bool
    depth: int


@dataclass(frozen=True)
class ModuleR:
    axis: Tuple[float, float, float]
    limits: Tuple[float, float]


@dataclass(frozen=True)
class ModuleC:
    axis: Tuple[float, float, float]


@dataclass(frozen=True)
class ModuleP:
    axis: Tuple[float, float, float]
    limits: Tuple[float, float]


@dataclass(frozen=True)
class ModuleCoupled:
    """Creates a revolute joint plus an ``AffineCoupling``. ``source_p`` is
    the (earlier) phalanx index within the *same digit* whose joint is the
    coupling source -- an index, not an absolute joint name, so that
    ``vary``'s insert/delete-phalanx operators can renumber without
    invalidating the reference."""

    axis: Tuple[float, float, float]
    source_p: int
    multiplier: float
    offset: float


ModuleSpec = Union[ModuleR, ModuleC, ModuleP, ModuleCoupled]


@dataclass(frozen=True)
class PhalanxProduction:
    """``Phalanx -> Module + segment(length) + optional Branch(Digit*)``.
    ``branch_digit_count > 0`` means this phalanx's own body (``"d{digit_id}p{p+1}"``)
    is the mount for that many sub-``Digit``s, recorded as separate ``Digit``/
    ``Phalanx`` steps whose ids are this digit's id extended with
    ``f"p{p+1}b{slot}"``."""

    digit_id: str
    p: int
    module: ModuleSpec
    length: float
    branch_digit_count: int
