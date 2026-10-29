"""Which finger each hand joint belongs to, and where along it: the per-finger token layout (29 Oct). Kit-free.

Every hand presents as a fixed grid of finger slots x ``N_SLOTS`` joint slots, palm -> tip, plus one slot for a
palm joint (a grammar hand's palm cut). A joint's identity is its position in its chain, a geometric fact, not a
name; fingers are unordered (the policy's weights are shared across them).

The rule, from the URDF tree and the spec: each fingertip body's path up to the palm body lists its finger's hand
joints in order. A joint on the path of two or more fingertips, or whose child link is a palm section (named
``<palm body><digits>``, the grammar exporter's ``palm0``), is a palm joint; the fingers whose path crosses it are
the ones the cut MOVES. Checked on every hand in the registry, the 100 random grammar hands and the primitive hands
(tests/test_finger_layout.py).

Two grids (``SlotGrid``). The default, ``N_FINGERS`` = 6 fingers in fingertip order and ``PALM_SLOT`` = 30, is the
per-finger token grid of the 29 Oct network and of ``multi:finger:`` hand sets, where every hand keeps its own
articulation and a slot is only a column. A ``multi:template:`` hand set shares ONE authored articulation across its
envs (inhand_olympics/author_template.py), so the grid is also a physical tree: ``n_rigid`` finger slots hang off
the palm body and ``n_moving`` off the palm section behind the palm joint, and a cut hand's moving fingers must take
moving slots (``grid_for`` sizes the grid from the set: 5 + 2 for the 100 random grammar hands).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

N_FINGERS, N_SLOTS = 6, 5                 # the grammar's maxima (hand.Rules); the commercial hands fit
PALM_SLOT = N_FINGERS * N_SLOTS           # 30: the palm joint's slot
N_HAND_SLOTS = PALM_SLOT + 1              # 31


@dataclass(frozen=True)
class SlotGrid:
    """``n_rigid`` finger slots on the palm, then ``n_moving`` on the palm section the palm joint moves, each of
    ``n_slots`` joint slots; the palm joint's slot comes last."""
    n_rigid: int = N_FINGERS
    n_moving: int = 0
    n_slots: int = N_SLOTS

    @property
    def n_fingers(self) -> int:
        return self.n_rigid + self.n_moving

    @property
    def palm_slot(self) -> int:
        return self.n_fingers * self.n_slots

    @property
    def n_hand_slots(self) -> int:
        return self.palm_slot + 1

    def is_moving_slot(self, finger_slot: int) -> bool:
        return finger_slot >= self.n_rigid


DEFAULT_GRID = SlotGrid()                 # 6 fingers in fingertip order, sections ignored: the 29 Oct token grid


@dataclass(frozen=True)
class FingerLayout:
    fingers: tuple[tuple[int, ...], ...]  # per finger, its hand-joint indices palm -> tip (spec order indices)
    palm: tuple[int, ...]                 # the palm joint(s): at most one
    n_joints: int
    moving: tuple[bool, ...] = ()         # per finger: behind the palm joint (a cut hand); () = no cut
    grid: SlotGrid = DEFAULT_GRID

    def with_grid(self, grid: SlotGrid) -> "FingerLayout":
        lay = FingerLayout(self.fingers, self.palm, self.n_joints, self.moving, grid)
        lay.validate()
        return lay

    @property
    def n_fingers(self) -> int:
        return len(self.fingers)

    @property
    def finger_moving(self) -> tuple[bool, ...]:
        return self.moving or (False,) * self.n_fingers

    @property
    def finger_slot_of_finger(self) -> tuple[int, ...]:
        """Each finger's finger slot. On the default grid (no moving slots) fingers take slots in order. On a grid
        with moving slots a cut hand's moving fingers take moving slots and its rigid fingers rigid slots; a hand
        without a cut fills the rigid slots first and spills into the moving ones (its palm joint slot is locked, so
        the section is rigid)."""
        g = self.grid
        if g.n_moving == 0:
            return tuple(range(self.n_fingers))
        rigid = [i for i, m in enumerate(self.finger_moving) if not m]
        mov = [i for i, m in enumerate(self.finger_moving) if m]
        if len(mov) > g.n_moving:
            raise ValueError(f"{len(mov)} moving fingers, the grid has {g.n_moving} moving slots")
        out = [-1] * self.n_fingers
        for k, i in enumerate(mov):
            out[i] = g.n_rigid + k
        free = [s for s in range(g.n_rigid)] + ([] if self.palm else [s for s in range(g.n_rigid + len(mov), g.n_fingers)])
        for i, s in zip(rigid, free):
            out[i] = s
        if any(s < 0 for s in out):
            raise ValueError(f"{self.n_fingers} fingers ({len(mov)} moving) do not fit a {g.n_rigid}+{g.n_moving} grid")
        return tuple(out)

    @property
    def slot_of_joint(self) -> tuple[int, ...]:
        """Each hand joint's slot: finger slot s, position k -> s * n_slots + k; the palm joint -> the palm slot."""
        out = [-1] * self.n_joints
        for f, joints in enumerate(self.fingers):
            s = self.finger_slot_of_finger[f]
            for k, j in enumerate(joints):
                out[j] = s * self.grid.n_slots + k
        for j in self.palm:
            out[j] = self.grid.palm_slot
        assert all(s >= 0 for s in out), "a hand joint is in no finger and not the palm joint"
        return tuple(out)

    @property
    def joint_of_slot(self) -> tuple[int, ...]:
        """Each of the grid's hand slots' hand-joint index, or -1 for a ghost slot."""
        out = [-1] * self.grid.n_hand_slots
        for j, s in enumerate(self.slot_of_joint):
            out[s] = j
        return tuple(out)

    @property
    def tip_slot_of_finger(self) -> tuple[int, ...]:
        """Each finger's fingertip body's slot: its last joint's."""
        return tuple(self.slot_of_joint[f[-1]] for f in self.fingers)

    def validate(self, name: str = "") -> None:
        g = self.grid
        if self.moving and len(self.moving) != self.n_fingers:
            raise ValueError(f"{name}: {len(self.moving)} moving flags for {self.n_fingers} fingers")
        if self.n_fingers > g.n_fingers:
            raise ValueError(f"{name}: {self.n_fingers} fingers, at most {g.n_fingers}")
        if any(len(f) > g.n_slots for f in self.fingers):
            raise ValueError(f"{name}: a finger has {max(len(f) for f in self.fingers)} joints, at most {g.n_slots}")
        if len(self.palm) > 1:
            raise ValueError(f"{name}: {len(self.palm)} palm joints, at most 1")
        if self.palm and not any(self.finger_moving):
            raise ValueError(f"{name}: a palm joint that moves no finger")
        covered = sorted(j for f in self.fingers for j in f) + sorted(self.palm)
        if sorted(covered) != list(range(self.n_joints)):
            raise ValueError(f"{name}: joints {covered} do not cover 0..{self.n_joints - 1} exactly once")
        self.finger_slot_of_finger            # raises when the fingers do not fit the grid's sections


def joint_moving(layout: FingerLayout) -> tuple[bool, ...]:
    """Per hand joint (spec order): on a finger the palm joint moves. The palm joint itself is not."""
    out = [False] * layout.n_joints
    for f, joints in enumerate(layout.fingers):
        if layout.finger_moving[f]:
            for j in joints:
                out[j] = True
    return tuple(out)


def joint_moving_of(spec) -> tuple[bool, ...]:
    """``joint_moving`` for a spec, all False where no layout can be read (a generated population has no URDF)."""
    try:
        return joint_moving(finger_layout(spec))
    except ValueError:
        return (False,) * spec.num_hand_joints


def grid_for(layouts, n_slots: int = N_SLOTS) -> SlotGrid:
    """The smallest grid whose sections hold every layout: as many moving slots as the most moving fingers of any
    hand, and rigid slots for the most rigid fingers of a cut hand or the most fingers of any hand less that."""
    layouts = list(layouts)
    n_moving = max((sum(l.finger_moving) for l in layouts), default=0)
    rigid_cut = max((l.n_fingers - sum(l.finger_moving) for l in layouts if l.palm), default=0)
    n_rigid = max(rigid_cut, max((l.n_fingers for l in layouts), default=0) - n_moving)
    return SlotGrid(n_rigid=n_rigid, n_moving=n_moving, n_slots=n_slots)


def finger_layout(spec) -> FingerLayout:
    """The layout of ``spec`` (a RobotSpec) from its URDF. Finger i is fingertip i (``spec.fingertip_body_names``).
    A multi-hand template has no URDF: ``robots.get_finger_layout`` resolves those (``template_layout``)."""
    if not spec.urdf_path:
        raise ValueError(f"{spec.name}: no URDF to read a finger layout from (a multi-hand ref goes through "
                         "robots.get_finger_layout; a generated population has no per-finger layout)")
    from hand_sampler import resolve
    root = ET.parse(resolve(spec.urdf_path)).getroot()
    hand = set(spec.hand_joint_names)
    up: dict[str, tuple[str, str]] = {}            # child link -> (joint, parent link)
    for j in root.findall("joint"):
        up[j.find("child").get("link")] = (j.get("name"), j.find("parent").get("link"))
    palm_re = re.compile(re.escape(spec.palm_body_name) + r"\d+$")
    paths = []                                       # per fingertip: [(joint, child link)] palm -> tip
    for tip in spec.fingertip_body_names:
        chain, link = [], tip
        while link != spec.palm_body_name:
            if link not in up:
                raise ValueError(f"{spec.name}: fingertip {tip} does not descend from {spec.palm_body_name}")
            jn, parent = up[link]
            if jn in hand:
                chain.append((jn, link))
            link = parent
        paths.append(chain[::-1])
    count: dict[str, int] = {}
    for path in paths:
        for jn, _ in path:
            count[jn] = count.get(jn, 0) + 1
    palm_joints = {jn for path in paths for jn, child in path if count[jn] >= 2 or palm_re.match(child)}
    idx = {n: i for i, n in enumerate(spec.hand_joint_names)}
    fingers = tuple(tuple(idx[jn] for jn, _ in path if jn not in palm_joints) for path in paths)
    if any(not f for f in fingers):
        raise ValueError(f"{spec.name}: a fingertip has no finger joints of its own (finger i must be fingertip i)")
    palm = tuple(idx[jn] for jn in spec.hand_joint_names if jn in palm_joints)
    moving = tuple(any(jn in palm_joints for jn, _ in path) for path in paths) if palm else ()
    lay = FingerLayout(fingers=fingers, palm=palm, n_joints=spec.num_hand_joints, moving=moving)
    lay.validate(spec.name)
    return lay


def template_layout(grid: SlotGrid = DEFAULT_GRID) -> FingerLayout:
    """A finger-major multi-hand template's own layout (multi_hand.py): the identity over ``grid`` -- slot k is finger
    k // n_slots, position k % n_slots, the palm slot the palm joint, the moving finger slots marked moving."""
    n = grid.n_slots
    fingers = tuple(tuple(range(f * n, (f + 1) * n)) for f in range(grid.n_fingers))
    moving = tuple(grid.is_moving_slot(f) for f in range(grid.n_fingers)) if grid.n_moving else ()
    return FingerLayout(fingers=fingers, palm=(grid.palm_slot,), n_joints=grid.n_hand_slots, moving=moving, grid=grid)


__all__ = ["DEFAULT_GRID", "FingerLayout", "SlotGrid", "finger_layout", "grid_for", "joint_moving", "joint_moving_of",
           "template_layout", "N_FINGERS", "N_SLOTS", "PALM_SLOT", "N_HAND_SLOTS"]
