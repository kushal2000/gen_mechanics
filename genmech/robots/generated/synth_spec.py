"""Synthesise a :class:`RobotSpec` from ``HandParams``.

The task reads everything hardware-specific from a spec, so a generated hand is
only usable once it has one. Authoring 30 joint names, four gain tables, a
self-collision map and fingertip geometry by hand for every sampled design is not
an option -- all of it is derived from the template here.

**Names encode their parameters.** ``get_generated_spec("gen_0007_003")`` returns
the fourth hand of ``sample_population(seed=7)``, rebuilt from the seed. Nothing
is stored on disk except the URDF, and that is regenerated on demand, so a
population is reproducible from an integer rather than from a directory that has
to be kept in sync.

**Two fixed-size things and one that is not.**

* *Joints* are always 37 (7 arm + 30 hand), ghosted or not. They have to be:
  PhysX reads one ``dof_count`` per articulation view, so a design that changed
  its joint count could not share a view with its neighbours
  (``genmech/tools/probe_multi_articulation.py``).
* *The self-collision map* is a constant of the template rather than per-design
  authoring, which retires the silent failure ``RobotSpec.validate`` guards
  against -- a hand whose adjacency map matched nothing used to filter zero
  pairs and collide with itself at every joint.
* *Fingertips* are the exception: only ACTIVE fingers are listed, so a 3-finger
  hand reports 3 fingertips and its observation is correspondingly smaller.
  That is right for v1, where each design gets its own spec and the reward's
  fingertip term should not sum over collapsed ghost fingers. A cross-embodied
  policy will need these padded to 5 with a mask -- see docs/proposal_codesign.md
  §7. Unlike the joint count, nothing in PhysX forces this one, so it stays a
  reward and observation choice rather than a constraint.
"""

from __future__ import annotations

import random
import re

from genmech.robots.generated import params as P
from genmech.robots.generated import sharpa_anchors as A
from genmech.robots.iiwa14_arm import (
    ARM_ADJACENT_LINKS,
    ARM_DAMPING,
    ARM_DEFAULT_JOINT_POS,
    ARM_JOINT_NAMES,
    ARM_STIFFNESS,
    BASE_POS,
    BASE_ROT,
    START_ARM_HIGHER_DELTAS,
)
from genmech.robots.spec import RobotSpec
from genmech.tools.build_hand_urdf import (
    OUT_DIR, joint_name, link_name, urdf_path_for, write_urdf,
)


# Palm body AFTER merge_fixed_joints. gen_palm is fixed-jointed to
# iiwa14_link_ee, which is fixed-jointed to iiwa14_link_7, so the importer
# collapses both into the arm's last link -- exactly as it does for SHARPA and
# Allegro.
PALM_BODY = "iiwa14_link_7"

# Identical to SHARPA's. The generated palm mounts at SHARPA's own composed
# flange-to-palm transform (sharpa_anchors.FLANGE_TO_PALM_*), so the same offset
# names the same physical point -- the property that makes palm-relative
# observations comparable across robots (docs/methodology.md).
PALM_CENTER_OFFSET = (-0.0, -0.02, 0.16)

# A ghosted joint is held shut by its URDF limits, so its actuator only has to
# avoid fighting them. Full tier gains on a joint with 1e-8 rad of travel is a
# PD controller pushing against a hard stop every step.
GHOST_STIFFNESS = 0.01
GHOST_DAMPING = 0.001
GHOST_ARMATURE = 1e-5

_GEN_NAME_RE = re.compile(r"^gen_(\d{4})_(\d{3})$")

GENERATED_PREFIX = "gen_"
SHARPA_LIKE_NAME = "gen_sharpa_like"


def hand_joint_names() -> tuple[str, ...]:
    """All 30 hand joints, in canonical order: finger slot major, chain minor.

    Ghosted joints are included. The action space is the same size for every
    generated hand by construction, which is the point of the template.
    """
    return tuple(
        joint_name(i, slot)
        for i in range(P.N_FINGER_SLOTS)
        for slot in P.JOINT_SLOTS
    )


def _slot_of(name: str) -> str:
    return name.split("_", 2)[2]


def _finger_index_of(name: str) -> int:
    return int(name.split("_")[1][1:])


def template_adjacent_links() -> dict[str, list[str]]:
    """Self-collision pairs to filter. A constant of the template.

    PhysX auto-filters directly-jointed parent/child pairs, so what earns its
    place here is the non-jointed geometric neighbours: the palm against each
    finger's first two segments, and each segment against its grandparent across
    a zero-length virtual link.

    Cross-finger pairs are deliberately absent -- fingers *should* collide with
    each other, or the hand can pass through itself while grasping.
    """
    pairs: dict[str, set[str]] = {}

    def link(a: str, b: str) -> None:
        pairs.setdefault(a, set()).add(b)
        pairs.setdefault(b, set()).add(a)

    for i in range(P.N_FINGER_SLOTS):
        cmc_vl = link_name(i, "CMC_VL")
        mc = link_name(i, "MC")
        mcp_vl = link_name(i, "MCP_VL")
        pp = link_name(i, "PP")
        mp = link_name(i, "MP")
        dp = link_name(i, "DP")

        # The palm overlaps the base of every finger.
        for near_palm in (cmc_vl, mc, mcp_vl, pp):
            link(PALM_BODY, near_palm)

        # Grandparent pairs across the virtual links, plus the phalanx chain.
        link(cmc_vl, mc)
        link(cmc_vl, mcp_vl)
        link(mc, mcp_vl)
        link(mc, pp)
        link(mcp_vl, pp)
        link(mcp_vl, mp)
        link(pp, mp)
        link(pp, dp)
        link(mp, dp)

    return {k: sorted(v) for k, v in pairs.items()}


def synth_spec(hand: P.HandParams, *, ensure_urdf: bool = True) -> RobotSpec:
    """Build a spec for one generated hand, writing its URDF if absent."""
    urdf = urdf_path_for(hand)
    if ensure_urdf and not urdf.exists():
        write_urdf(hand, urdf)

    names = hand_joint_names()
    active: dict[str, bool] = {}
    for name in names:
        fp = hand.fingers[_finger_index_of(name)]
        slot_on = dict(zip(P.JOINT_SLOTS, fp.enabled))[_slot_of(name)]
        active[name] = fp.active and slot_on

    def table(src: dict[str, float], ghost_value: float) -> dict[str, float]:
        return {
            n: (src[_slot_of(n)] if active[n] else ghost_value)
            for n in names
        }

    # Fingertips: active fingers only (see module docstring). The tip link is
    # fixed-jointed onto DP, so post-merge the fingertip BODY is DP itself.
    tip_bodies: list[str] = []
    tip_offsets: list[tuple[float, float, float]] = []
    for i, fp in enumerate(hand.fingers):
        if not fp.active:
            continue
        tip_bodies.append(link_name(i, "DP"))
        # The capsule runs along +x from the link origin, so its distal cap --
        # the part that actually touches the object -- is one segment length out.
        tip_offsets.append((fp.dp_length, 0.0, 0.0))

    adjacency = {**ARM_ADJACENT_LINKS, **template_adjacent_links()}
    # The palm key exists in both; the arm's entry must not be dropped.
    arm_palm = ARM_ADJACENT_LINKS.get(PALM_BODY, [])
    if arm_palm:
        adjacency[PALM_BODY] = sorted(
            set(adjacency.get(PALM_BODY, [])) | set(arm_palm)
        )

    return RobotSpec(
        name=hand.name,
        arm_name="iiwa14",
        hand_name="generated",
        urdf_path=f"{OUT_DIR}/{hand.name}.urdf",

        arm_joint_names=ARM_JOINT_NAMES,
        hand_joint_names=names,

        palm_body_name=PALM_BODY,
        fingertip_body_names=tuple(tip_bodies),

        arm_stiffness=ARM_STIFFNESS,
        arm_damping=ARM_DAMPING,
        hand_stiffness=table(A.SLOT_STIFFNESS, GHOST_STIFFNESS),
        hand_damping=table(A.SLOT_DAMPING, GHOST_DAMPING),
        hand_armature=table(A.SLOT_ARMATURE, GHOST_ARMATURE),

        arm_default_joint_pos=ARM_DEFAULT_JOINT_POS,
        # Zero is inside every slot's limits, active or ghosted: SHARPA's own
        # home pose is all-zeros and the sampled abduction ranges are symmetric
        # about it.
        hand_default_joint_pos={n: 0.0 for n in names},
        start_arm_higher_deltas=START_ARM_HIGHER_DELTAS,

        palm_center_offset=PALM_CENTER_OFFSET,
        fingertip_offsets=tuple(tip_offsets),

        adjacent_links=adjacency,
        link_prim_regexes=("iiwa14_link_.*", "gen_.*"),

        base_pos=BASE_POS,
        base_rot=BASE_ROT,

        # Capsules: URDF has no capsule primitive, so build_hand_urdf emits
        # cylinders and the importer rounds their ends.
        replace_cylinders_with_capsules=True,

        notes=(
            f"Procedurally generated. {hand.n_active_fingers} active finger(s), "
            f"{hand.n_active_joints} active of {len(names)} hand joints; the rest "
            f"are ghosted so the articulation shape is fixed. Geometry is "
            f"capsules with densities calibrated against SHARPA "
            f"(genmech/robots/generated/sharpa_anchors.py). {hand.notes}"
        ),
    )


def params_for_name(name: str) -> P.HandParams:
    """Recover the parameter vector a generated spec name encodes.

    ``gen_sharpa_like``   -> the measured SHARPA reference vector
    ``gen_<seed>_<index>`` -> that element of ``sample_population(seed)``

    Rebuilding from the seed rather than reading a file means a population is
    reproducible from an integer, and a name cannot go stale against a directory.
    """
    if name == SHARPA_LIKE_NAME:
        return P.SHARPA_LIKE

    m = _GEN_NAME_RE.match(name)
    if m is None:
        raise KeyError(
            f"{name!r} is not a generated hand name; expected "
            f"{SHARPA_LIKE_NAME!r} or gen_<4-digit seed>_<3-digit index>"
        )
    seed, index = int(m.group(1)), int(m.group(2))
    # sample_population draws sequentially from one RNG, so element i depends on
    # every draw before it -- generate the prefix rather than seeking.
    rng = random.Random(seed)
    hand = None
    for i in range(index + 1):
        hand = P.sample_valid(rng, name=f"gen_{seed:04d}_{i:03d}")
    assert hand is not None
    return hand


def get_generated_spec(name: str) -> RobotSpec:
    """Registry entry point for ``gen_*`` names."""
    hand = params_for_name(name)
    if hand.name != name:
        # SHARPA_LIKE is authored as "sharpa_like"; the registry name carries the
        # gen_ prefix so the fallback can recognise it.
        from dataclasses import replace as _replace

        hand = _replace(hand, name=name)
    return synth_spec(hand)


def is_generated_name(name: str) -> bool:
    return name == SHARPA_LIKE_NAME or _GEN_NAME_RE.match(name) is not None


__all__ = [
    "PALM_BODY", "PALM_CENTER_OFFSET", "GENERATED_PREFIX", "SHARPA_LIKE_NAME",
    "hand_joint_names", "template_adjacent_links", "synth_spec",
    "params_for_name", "get_generated_spec", "is_generated_name",
]
