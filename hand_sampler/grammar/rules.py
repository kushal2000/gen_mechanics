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

Convention (decided in this iteration): ``Branch(Digit*)`` reuses the same
``Digit`` production as ``Hand``'s top-level digits -- every ``Digit``,
whether spawned directly by ``Hand`` or by a ``Branch``, mounts on *a palm
body* (never on the branching phalanx itself), matching the ``Digit(mount:
a palm body, ...)`` signature literally and keeping "every digit mounts on
a palm body" true unconditionally (checked by the benchmark). A branch is
still a real generative event tied to its phalanx (recorded as
``branch_digit_count > 0`` on that ``Phalanx`` step) -- it just adds more
digits to the hand rather than forking the finger's own tip.

Body/joint naming is deterministic from the derivation: the root is
``"root"``; palm bodies are ``"palm{i}"`` with joint ``"palm{i}_j"``
(``i`` 0-based, chained off the root); a digit's phalanx bodies are
``"d{gid+1}p{p+1}"`` with joint ``"d{gid+1}p{p+1}_j"`` (``gid`` a global,
0-based digit id shared by top-level and branch digits alike; ``p`` the
0-based phalanx index within that digit).

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
    """``PalmBody(parent) -> segment(length, direction) + optional PalmJoint(R, axis, limits)``."""

    parent: str
    length: float
    direction_rpy: Tuple[float, float, float]
    has_joint: bool
    axis: Tuple[float, float, float]
    limits: Optional[Tuple[float, float]]


@dataclass(frozen=True)
class DigitProduction:
    """``Digit(mount, mount_pose) -> Phalanx+``."""

    gid: int
    mount: str
    mount_frac: float
    mount_rpy: Tuple[float, float, float]
    phalanx_count: int
    top_level: bool


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
    """``Phalanx -> Module + segment(length) + optional Branch(Digit*)``."""

    gid: int
    p: int
    module: ModuleSpec
    length: float
    branch_digit_count: int
