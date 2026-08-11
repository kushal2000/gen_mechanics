"""The generated hand design space.

A ``HandParams`` is one point in morphology space: five finger slots, each with a
mount pose, segment lengths, joint limits, and a flag per joint saying whether it
is actuated or ghosted. ``build_hand_urdf.py`` turns it into a robot.

**Fixed topology.** Every hand has the same 6 joints x 5 slots = 30 hand joints
(+7 arm = 37), whatever it looks like. Fewer fingers, or fewer joints in a
finger, are expressed by *ghosting*: the joint stays in the articulation with its
limits locked to ~0, its link massless and geometry-free. This is House of
Dextra's trick (``Generation/converters/generate_rand_gram_joint_finger.py``), and
it is what lets one Isaac Lab ``Articulation`` view hold every design -- see
``genmech/tools/probe_multi_articulation.py`` for the constraint it dodges.

It is also not foreign to this robot: SHARPA already represents its zero-length
virtual links with ``mass = 1e-6``, which is exactly the ghosting convention.

**Two representations.** The dataclasses below are the *full* representation --
explicit rigid transforms -- so that ``SHARPA_LIKE`` reproduces SHARPA's
kinematics exactly rather than approximately. ``sample()`` is a map from a much
smaller set of interpretable knobs into that representation. Search operates on
the knobs; the URDF builder consumes the transforms.

**The chain.** Per finger::

    palm --mount--> [CMC_FE] --cmc--> [CMC_AA] --mc--> [MCP_FE] --mcp--> [MCP_AA]
         --pp_length--> [PIP] --mp_length--> [DIP] --dp_length--> tip

``mount``, ``cmc`` and ``mc`` are arbitrary rigid transforms because SHARPA's are
(the thumb's metacarpal carries a -30 deg roll and an off-axis translation, and
the pinky's is rotated (-90, +90, 0)). From ``MCP_AA`` outward the pattern is
regular: translate along local +x, then roll +-90 deg about x to alternate the
joint axis between flexion and abduction. Every joint ``axis`` is ``[0 0 1]``,
following SHARPA -- orientation lives in the origin rpy, never in the axis.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Sequence

from genmech.robots.generated import sharpa_anchors as anchors


Vec3 = tuple[float, float, float]

# Joint slots, in chain order. This ordering is load-bearing: it defines the
# canonical hand-joint order, which becomes the policy's action layout.
JOINT_SLOTS: tuple[str, ...] = (
    "CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "PIP", "DIP",
)
N_JOINT_SLOTS = len(JOINT_SLOTS)
N_FINGER_SLOTS = 5

D = math.radians


@dataclass(frozen=True)
class Segment:
    """One rigid step in the chain: translate by ``xyz``, then rotate by ``rpy``.

    Matches URDF ``<origin>`` semantics exactly, so it maps to the emitted joint
    origin with no conversion.
    """

    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)

    @property
    def length(self) -> float:
        return math.sqrt(sum(v * v for v in self.xyz))


IDENTITY = Segment()

# Canonical rolls in the regular part of the chain. FE and AA axes must be
# perpendicular, and since every axis is [0 0 1] that perpendicularity is carried
# entirely by these rolls.
ROLL_FE_TO_AA = Segment(rpy=(D(90.0), 0.0, 0.0))
ROLL_AA_TO_FE = Segment(rpy=(D(-90.0), 0.0, 0.0))


@dataclass(frozen=True)
class FingerParams:
    """One finger slot. ``active=False`` ghosts the whole finger."""

    name: str
    active: bool = True

    # Which of the 6 slots are actuated. A False entry keeps the joint in the
    # articulation but locks it, and drops its link's geometry and mass.
    enabled: tuple[bool, ...] = (True,) * N_JOINT_SLOTS

    # Arbitrary transforms (see module docstring).
    mount: Segment = IDENTITY   # palm -> CMC_FE
    cmc: Segment = IDENTITY     # CMC_FE -> CMC_AA
    mc: Segment = IDENTITY      # CMC_AA -> MCP_FE   (metacarpal)
    mcp: Segment = ROLL_FE_TO_AA  # MCP_FE -> MCP_AA

    # Regular part: pure +x translations, metres.
    pp_length: float = anchors.TIER_NOMINAL_LENGTH_M["pp"]
    mp_length: float = anchors.TIER_NOMINAL_LENGTH_M["mp"]
    dp_length: float = anchors.TIER_NOMINAL_LENGTH_M["dp"]

    # Per-slot (lower, upper) in radians, in JOINT_SLOTS order.
    limits: tuple[tuple[float, float], ...] = ()

    # Scales every tier radius on this finger. Thickness matters for grasping and
    # is cheap to vary, but it is one knob rather than four so that a sampled
    # finger stays proportioned.
    radius_scale: float = 1.0

    def __post_init__(self) -> None:
        if len(self.enabled) != N_JOINT_SLOTS:
            raise ValueError(
                f"{self.name}: enabled has {len(self.enabled)} flags, "
                f"expected {N_JOINT_SLOTS}"
            )
        if len(self.limits) != N_JOINT_SLOTS:
            raise ValueError(
                f"{self.name}: limits has {len(self.limits)} entries, "
                f"expected {N_JOINT_SLOTS}"
            )
        for slot, (lo, hi) in zip(JOINT_SLOTS, self.limits):
            if hi < lo:
                raise ValueError(f"{self.name}.{slot}: upper {hi} < lower {lo}")
        if self.radius_scale <= 0.0:
            raise ValueError(f"{self.name}: radius_scale must be positive")

    @property
    def n_active_joints(self) -> int:
        return sum(self.enabled) if self.active else 0

    def segment_length(self, tier: str) -> float:
        return {"pp": self.pp_length, "mp": self.mp_length,
                "dp": self.dp_length, "mc": self.mc.length}[tier]

    def reach(self) -> float:
        """Straight-line palm-to-tip distance with the finger extended.

        An upper bound on how far this finger can reach, used by the sampler's
        validity check. Sums segment magnitudes rather than composing transforms,
        so it is an over-estimate -- which is the safe direction for a bound.
        """
        return (self.mount.length + self.cmc.length + self.mc.length
                + self.pp_length + self.mp_length + self.dp_length)


@dataclass(frozen=True)
class HandParams:
    """A complete hand: five finger slots plus the palm."""

    name: str
    fingers: tuple[FingerParams, ...]
    palm_extents: Vec3 = anchors.PALM_EXTENTS_M
    notes: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if len(self.fingers) != N_FINGER_SLOTS:
            raise ValueError(
                f"{self.name}: {len(self.fingers)} finger slots, "
                f"expected exactly {N_FINGER_SLOTS} (inactive ones are ghosted, "
                f"not omitted -- the articulation shape is fixed)"
            )
        names = [f.name for f in self.fingers]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: duplicate finger names {names}")
        if self.n_active_fingers < 2:
            raise ValueError(
                f"{self.name}: {self.n_active_fingers} active finger(s); a hand "
                f"needs at least 2 to oppose anything"
            )

    @property
    def active_fingers(self) -> tuple[FingerParams, ...]:
        return tuple(f for f in self.fingers if f.active)

    @property
    def n_active_fingers(self) -> int:
        return len(self.active_fingers)

    @property
    def n_active_joints(self) -> int:
        return sum(f.n_active_joints for f in self.fingers)


# ---------------------------------------------------------------------------
# SHARPA_LIKE -- the reference point, measured not guessed
# ---------------------------------------------------------------------------
#
# Every number below is read from
# assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf.
# This vector is the generator's default and the regression target: a hand built
# from it must reproduce SHARPA's fingertip positions and phalanx masses.
#
# Slot mapping per SHARPA finger:
#
#   index/middle/ring   ghost CMC_FE + CMC_AA          -> 4 active
#   pinky               ghost CMC_AA (its CMC is FE)   -> 5 active
#   thumb               ghost PIP    (its IP is DIP)   -> 5 active
#
# The thumb ghosts PIP rather than DIP so its distal segment lands in the DP
# tier. Ghosting DIP instead would put a 27.6 mm segment in the MP tier, whose
# density is fitted to actuator-bearing links -- 22.8 g against a measured 3.9 g.

_GHOST_LIMIT = (0.0, 1e-8)

# SHARPA's finger limits, by role.
_LIM_FINGER = {
    "CMC_FE": _GHOST_LIMIT,
    "CMC_AA": _GHOST_LIMIT,
    "MCP_FE": (-0.17453293, 1.5708),
    "MCP_AA": (-0.03491, 0.03491),
    "PIP": (0.0, 1.7453),
    "DIP": (0.0, 1.3963),
}
_LIM_PINKY = {**_LIM_FINGER, "CMC_FE": (0.0, 0.2618)}
_LIM_THUMB = {
    "CMC_FE": (-0.1745, 1.9199),
    "CMC_AA": (-0.3491, 0.1309),
    "MCP_FE": (-0.5236, 1.3963),
    "MCP_AA": (-0.3491, 0.3491),
    "PIP": _GHOST_LIMIT,
    "DIP": (0.0, 1.7453),
}


def _limits(table: dict[str, tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    return tuple(table[s] for s in JOINT_SLOTS)


def _sharpa_finger(name: str, mount_xyz: Vec3) -> FingerParams:
    """index / middle / ring: identical fingers at different mounts.

    SHARPA varies placement, not finger -- all three share PP 47.0, MP 31.5,
    DP 26.0 and the same (0, -90, -90) mount orientation.
    """
    return FingerParams(
        name=name,
        enabled=(False, False, True, True, True, True),
        mount=Segment(xyz=mount_xyz, rpy=(0.0, D(-90.0), D(-90.0))),
        pp_length=0.0470, mp_length=0.0315, dp_length=0.02601,
        limits=_limits(_LIM_FINGER),
    )


SHARPA_LIKE = HandParams(
    name="sharpa_like",
    notes=(
        "Measured from the SHARPA URDF. Reproduces its kinematics inside the "
        "generated template; the only deviations are capsule geometry in place "
        "of meshes and the thumb's distal segment landing in the DP tier."
    ),
    fingers=(
        # Thumb: the only finger with a real metacarpal and real abduction.
        FingerParams(
            name="thumb",
            enabled=(True, True, True, True, False, True),
            mount=Segment(xyz=(0.0100, -0.0260, 0.0212), rpy=(0.0, 0.0, D(-90.0))),
            cmc=Segment(xyz=(0.0, -0.0050, 0.0), rpy=(D(90.0), D(-45.0), 0.0)),
            mc=Segment(xyz=(0.0650, -0.0060, -0.01039), rpy=(D(-30.0), 0.0, 0.0)),
            pp_length=0.0390,
            mp_length=0.0,        # PIP ghosted
            dp_length=0.02761,
            limits=_limits(_LIM_THUMB),
        ),
        _sharpa_finger("index", (0.0010, -0.0303, 0.0957)),
        _sharpa_finger("middle", (0.0000, -0.0100, 0.0987)),
        _sharpa_finger("ring", (0.0015, 0.0103, 0.0927)),
        # Pinky: a short metacarpal gives the palm its arch.
        FingerParams(
            name="pinky",
            enabled=(True, False, True, True, True, True),
            mount=Segment(xyz=(0.01136, 0.0263, 0.0867),
                          rpy=(D(-180.0), 0.0, D(180.0))),
            mc=Segment(xyz=(0.00836, 0.0048, 0.0), rpy=(D(-90.0), D(90.0), 0.0)),
            pp_length=0.0470, mp_length=0.0315, dp_length=0.02601,
            limits=_limits(_LIM_PINKY),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Sampling ranges -- centred on SHARPA
# ---------------------------------------------------------------------------
#
# Anchoring on SHARPA is deliberate: it is the hand known to solve this task, so
# a sampled hand that scores badly is telling us about morphology rather than
# about being outside any sane region. Ranges bracket SHARPA's measured values
# with roughly +-40% headroom on lengths.

MOUNT_Y_RANGE = (-0.045, 0.040)     # SHARPA: -0.0303 .. +0.0263
MOUNT_Z_RANGE = (0.010, 0.115)      # SHARPA: 0.0212 (thumb) .. 0.0987
MOUNT_X_RANGE = (-0.010, 0.020)     # SHARPA: 0.0 .. 0.01136
MOUNT_RPY_JITTER = D(25.0)          # about the role's SHARPA orientation

MC_LENGTH_RANGE = (0.0, 0.110)      # SHARPA: 0 (fingers), 0.0096, 0.0661
PP_LENGTH_RANGE = (0.028, 0.066)    # SHARPA: 0.039 (thumb), 0.047
MP_LENGTH_RANGE = (0.019, 0.044)    # SHARPA: 0.0315
DP_LENGTH_RANGE = (0.016, 0.036)    # SHARPA: 0.026, 0.0276
RADIUS_SCALE_RANGE = (0.6, 1.2)

# Abduction is the interesting knob. SHARPA's fingers are locked to +-2 deg and
# only its thumb abducts; House of Dextra cannot vary abduction at all (its joint
# axes are CAD constants). Opening it is a genuine hardware choice.
AA_HALF_RANGE = (D(2.0), D(25.0))

N_FINGERS_CHOICES = (2, 3, 4, 5)

# Validity thresholds.
MIN_MOUNT_SEPARATION = 0.015   # m, so fingers do not interpenetrate at the base
MIN_REACH = 0.060              # m, a finger shorter than this cannot enclose the
                               # task's objects
MAX_REACH = 0.260              # m, beyond this the hand outgrows the workspace


class InvalidHand(ValueError):
    """A sampled parameter vector that failed a validity check."""


def _u(rng: random.Random, lo_hi: tuple[float, float]) -> float:
    return rng.uniform(*lo_hi)


def _jitter_rpy(rng: random.Random, base: Vec3, amount: float) -> Vec3:
    return tuple(b + rng.uniform(-amount, amount) for b in base)  # type: ignore[return-value]


def sample(rng: random.Random, *, name: str, n_fingers: int | None = None) -> HandParams:
    """Draw one hand from the design space.

    Raises :class:`InvalidHand` if the draw violates a validity check; callers
    should use :func:`sample_valid`, which resamples. Rejection rather than
    clipping, because clipping piles probability mass on the boundary and would
    make the boundary look artificially good to a surrogate.
    """
    if n_fingers is None:
        n_fingers = rng.choice(N_FINGERS_CHOICES)
    if not 2 <= n_fingers <= N_FINGER_SLOTS:
        raise ValueError(f"n_fingers must be in [2, {N_FINGER_SLOTS}], got {n_fingers}")

    slot_names = [f.name for f in SHARPA_LIKE.fingers]
    fingers: list[FingerParams] = []

    for i, slot in enumerate(slot_names):
        ref = SHARPA_LIKE.fingers[i]
        if i >= n_fingers:
            # Ghosted finger: keep the reference geometry so the URDF stays
            # well-formed, and let `active=False` strip mass and collision.
            fingers.append(replace(ref, active=False))
            continue

        # Slot 0 keeps the thumb's role (a real metacarpal and real abduction);
        # the rest are fingers. Role sets which joints exist, not their values.
        is_thumb = i == 0
        mc_length = (_u(rng, MC_LENGTH_RANGE) if is_thumb
                     else _u(rng, (0.0, 0.030)))
        aa = _u(rng, AA_HALF_RANGE)

        limits = dict(zip(JOINT_SLOTS, ref.limits))
        limits["MCP_AA"] = (-aa, aa)

        fingers.append(replace(
            ref,
            active=True,
            mount=Segment(
                xyz=(_u(rng, MOUNT_X_RANGE), _u(rng, MOUNT_Y_RANGE),
                     _u(rng, MOUNT_Z_RANGE)),
                rpy=_jitter_rpy(rng, ref.mount.rpy, MOUNT_RPY_JITTER),
            ),
            mc=Segment(xyz=(mc_length, 0.0, 0.0), rpy=ref.mc.rpy),
            pp_length=_u(rng, PP_LENGTH_RANGE),
            mp_length=_u(rng, MP_LENGTH_RANGE),
            dp_length=_u(rng, DP_LENGTH_RANGE),
            radius_scale=_u(rng, RADIUS_SCALE_RANGE),
            limits=_limits(limits),
        ))

    hand = HandParams(name=name, fingers=tuple(fingers))
    validate(hand)
    return hand


def validate(hand: HandParams) -> None:
    """Reject hands that are geometrically impossible or useless for the task."""
    active = hand.active_fingers

    for f in active:
        r = f.reach()
        if r < MIN_REACH:
            raise InvalidHand(
                f"{hand.name}.{f.name}: reach {r * 1000:.1f} mm below "
                f"{MIN_REACH * 1000:.0f} mm; cannot enclose the task's objects"
            )
        if r > MAX_REACH:
            raise InvalidHand(
                f"{hand.name}.{f.name}: reach {r * 1000:.1f} mm above "
                f"{MAX_REACH * 1000:.0f} mm; outgrows the workspace"
            )

    # Mounts must not coincide, or the finger bases interpenetrate and PhysX
    # spends every step pushing them apart.
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            d = math.dist(a.mount.xyz, b.mount.xyz)
            if d < MIN_MOUNT_SEPARATION:
                raise InvalidHand(
                    f"{hand.name}: {a.name} and {b.name} mounts are "
                    f"{d * 1000:.1f} mm apart, below "
                    f"{MIN_MOUNT_SEPARATION * 1000:.0f} mm"
                )


def sample_valid(
    rng: random.Random, *, name: str, n_fingers: int | None = None,
    max_tries: int = 200,
) -> HandParams:
    """Resample until a draw passes :func:`validate`."""
    last: InvalidHand | None = None
    for _ in range(max_tries):
        try:
            return sample(rng, name=name, n_fingers=n_fingers)
        except InvalidHand as exc:
            last = exc
    raise InvalidHand(
        f"no valid hand in {max_tries} tries; last failure: {last}"
    )


def sample_population(
    seed: int, count: int, *, prefix: str = "gen", n_fingers: int | None = None,
) -> list[HandParams]:
    """A reproducible population. Seeded, so a run is re-derivable from an int."""
    rng = random.Random(seed)
    return [
        sample_valid(rng, name=f"{prefix}_{seed:04d}_{i:03d}", n_fingers=n_fingers)
        for i in range(count)
    ]


__all__ = [
    "JOINT_SLOTS", "N_JOINT_SLOTS", "N_FINGER_SLOTS",
    "Segment", "IDENTITY", "ROLL_FE_TO_AA", "ROLL_AA_TO_FE",
    "FingerParams", "HandParams", "SHARPA_LIKE",
    "InvalidHand", "sample", "sample_valid", "sample_population", "validate",
]
