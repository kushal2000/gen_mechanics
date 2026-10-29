"""Which finger each hand joint belongs to, and where along it: the per-finger token layout (29 Oct). Kit-free.

Every hand presents as a fixed grid of ``N_FINGERS`` finger slots x ``N_SLOTS`` joint slots, palm -> tip, plus one slot
for a palm joint (a grammar hand's palm cut): ``N_HAND_SLOTS`` = 31. A joint's identity is its position in its chain,
a geometric fact, not a name; fingers are unordered (the policy's weights are shared across them).

The rule, from the URDF tree and the spec: each fingertip body's path up to the palm body lists its finger's hand
joints in order. A joint on the path of two or more fingertips, or whose child link is a palm section (named
``<palm body><digits>``, the grammar exporter's ``palm0``), is a palm joint. Checked on every hand in the registry,
the 100 random grammar hands and the primitive hands (tests/test_finger_layout.py).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

N_FINGERS, N_SLOTS = 6, 5                 # the grammar's maxima (hand.Rules); the commercial hands fit
PALM_SLOT = N_FINGERS * N_SLOTS           # 30: the palm joint's slot
N_HAND_SLOTS = PALM_SLOT + 1              # 31


@dataclass(frozen=True)
class FingerLayout:
    fingers: tuple[tuple[int, ...], ...]  # per finger, its hand-joint indices palm -> tip (spec order indices)
    palm: tuple[int, ...]                 # the palm joint(s): at most one
    n_joints: int

    @property
    def slot_of_joint(self) -> tuple[int, ...]:
        """Each hand joint's finger-major slot: finger f, position k -> f * N_SLOTS + k; the palm joint -> PALM_SLOT."""
        out = [-1] * self.n_joints
        for f, joints in enumerate(self.fingers):
            for k, j in enumerate(joints):
                out[j] = f * N_SLOTS + k
        for j in self.palm:
            out[j] = PALM_SLOT
        assert all(s >= 0 for s in out), "a hand joint is in no finger and not the palm joint"
        return tuple(out)

    @property
    def joint_of_slot(self) -> tuple[int, ...]:
        """Each of the N_HAND_SLOTS slots' hand-joint index, or -1 for a ghost slot."""
        out = [-1] * N_HAND_SLOTS
        for j, s in enumerate(self.slot_of_joint):
            out[s] = j
        return tuple(out)

    def validate(self, name: str = "") -> None:
        if len(self.fingers) > N_FINGERS:
            raise ValueError(f"{name}: {len(self.fingers)} fingers, at most {N_FINGERS}")
        if any(len(f) > N_SLOTS for f in self.fingers):
            raise ValueError(f"{name}: a finger has {max(len(f) for f in self.fingers)} joints, at most {N_SLOTS}")
        if len(self.palm) > 1:
            raise ValueError(f"{name}: {len(self.palm)} palm joints, at most 1")
        covered = sorted(j for f in self.fingers for j in f) + sorted(self.palm)
        if sorted(covered) != list(range(self.n_joints)):
            raise ValueError(f"{name}: joints {covered} do not cover 0..{self.n_joints - 1} exactly once")


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
    lay = FingerLayout(fingers=fingers, palm=palm, n_joints=spec.num_hand_joints)
    lay.validate(spec.name)
    return lay


def template_layout(n_hand_joints: int) -> FingerLayout:
    """A finger-major multi-hand template (multi_hand.py, ``multi:finger:...``): slot k is finger k // N_SLOTS,
    position k % N_SLOTS, and PALM_SLOT the palm joint."""
    if n_hand_joints != N_HAND_SLOTS:
        raise ValueError(f"a finger-major template has {N_HAND_SLOTS} hand-joint slots, not {n_hand_joints}")
    fingers = tuple(tuple(range(f * N_SLOTS, (f + 1) * N_SLOTS)) for f in range(N_FINGERS))
    return FingerLayout(fingers=fingers, palm=(PALM_SLOT,), n_joints=N_HAND_SLOTS)


__all__ = ["FingerLayout", "finger_layout", "template_layout", "N_FINGERS", "N_SLOTS", "PALM_SLOT", "N_HAND_SLOTS"]
