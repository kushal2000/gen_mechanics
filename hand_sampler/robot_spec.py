"""Fixed robots: what the task needs to know about a specific arm+hand.

A ``RobotSpec`` is joint names, PD gains, palm and fingertip geometry, the
self-collision map and the action dimension. Before it existed, SHARPA was
hardcoded across ~9 sites. The measured constants and the arm definition it is
built from live here too, so one file answers "what robot is this".
"""

from __future__ import annotations


# --- the frozen description a task reads -------------------------------------

from dataclasses import dataclass, field
from typing import Mapping


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotSpec:
    """A frozen description of one arm+hand combination."""

    # --- identity -----------------------------------------------------------
    name: str
    """Registry key, e.g. "sharpa_iiwa14"."""
    arm_name: str
    """Arm family, e.g. "iiwa14". Must match across hands for a valid comparison."""
    hand_name: str
    """Hand family, e.g. "sharpa" / "allegro"."""
    urdf_path: str
    """Repo-relative URDF path; resolved against REPO_ROOT, not the CWD."""

    # --- joints (ORDERED; together they define canonical policy order) -------
    arm_joint_names: tuple[str, ...]
    hand_joint_names: tuple[str, ...]

    # --- bodies -------------------------------------------------------------
    palm_body_name: str
    """Palm body name AFTER merge_fixed_joints. The URDF's palm link is usually
    merged into the arm's last link, so this is typically an arm link name."""
    fingertip_body_names: tuple[str, ...]
    """Ordered fingertip bodies, likewise post-merge."""

    # --- actuation (keyed by joint name) ------------------------------------
    arm_stiffness: Mapping[str, float]
    arm_damping: Mapping[str, float]
    hand_stiffness: Mapping[str, float]
    hand_damping: Mapping[str, float]
    hand_armature: Mapping[str, float]

    # --- home pose ----------------------------------------------------------
    arm_default_joint_pos: Mapping[str, float]
    hand_default_joint_pos: Mapping[str, float]
    start_arm_higher_deltas: Mapping[str, float]
    """Radian offsets applied to the arm home pose when reset.start_arm_higher is
    set (the DexToolBench eval pose). Replaces a hardcoded joint_2/joint_4 nudge."""

    # --- observation geometry ----------------------------------------------
    palm_center_offset: Vec3
    """Offset from the palm body origin to the grasp center, in the palm frame.
    Defines the frame that palm_pos, keypoints_rel_palm, and fingertip_pos_rel_palm
    are expressed in, so it must mean the same physical thing for every hand or
    the policies see semantically different observations of the same world."""
    fingertip_offsets: tuple[Vec3, ...]
    """Per-fingertip offset from body origin to pad center. Per-fingertip rather
    than one shared constant because hands use different distal geometry per
    finger (Allegro's thumb carries a different sensor mesh than its fingers)."""

    # --- physics ------------------------------------------------------------
    adjacent_links: Mapping[str, list[str]]
    """Link pairs whose self-collision is filtered out, in POST-merge body names.
    PhysX already auto-filters directly-jointed parent/child pairs, so the value
    here is the non-kinematic-neighbor pairs (palm-to-proximal-phalanx, etc.)."""

    # --- scene prim patterns -------------------------------------------------
    link_prim_regexes: tuple[str, ...]
    """Prim-path patterns matching this robot's visual meshes, for the depth
    raycaster and viewers."""

    # --- base placement ------------------------------------------------------
    base_pos: Vec3 = (0.0, 0.8, 0.0)
    base_rot: Quat = (1.0, 0.0, 0.0, 0.0)

    # --- asset conversion ----------------------------------------------------
    replace_cylinders_with_capsules: bool = False
    """Convert ``<cylinder>`` collision geometry to PhysX capsules on import.

    URDF has no capsule primitive, so procedurally generated hands emit cylinders
    and rely on this to get rounded ends — which matter for contact (a cylinder's
    rim is a sharp edge) and are the cheapest shape PhysX has. Defaults to False
    so the mesh-based hands convert exactly as before and SHARPA stays
    bit-identical to simtoolreal (docs/methodology.md §2)."""

    # --- cross-embodiment padding -------------------------------------------
    fingertip_slot_names: tuple[str, ...] = ()
    """ALL fingertip slots the morphology template defines, active or not.

    A cross-embodied policy needs one observation layout for every design it may
    see, but designs differ in how many fingers they actually use: the generated
    population runs 2, 3 or 4 active fingertips against a 5-slot template. The
    slot list is the padded, template-constant axis the observation is built on,
    and ``fingertip_slot_active`` says which entries are real.

    This works because ghosting removes a finger's ACTUATION and GEOMETRY, not
    its links -- every generated design carries all 5 distal links, so the body
    indices are the same in every env and only the mask varies. Verified across
    the 64-hand population: 0 designs missing any of the 5 slots.

    Empty means "no padding": slots are exactly ``fingertip_body_names`` and the
    mask is all-true, so single-robot specs keep their existing observation
    layout byte for byte. Do not populate this for a fixed hand."""

    fingertip_slot_active: tuple[bool, ...] = ()
    """Per-slot validity mask, parallel to ``fingertip_slot_names``.

    Masked-out slots must not reach a reward, a termination test or a running
    minimum -- a ghosted finger's distal link still has a pose, and it is
    meaningless. Empty means all slots are active."""

    fingertip_slot_offsets: tuple[Vec3, ...] = ()
    """Pad-center offsets for ALL slots, parallel to ``fingertip_slot_names``.

    Separate from ``fingertip_offsets`` because these are per-design even for the
    same slot index -- the distal phalanx length varies across the population --
    so the env carries them per env rather than as one shared table."""

    notes: str = field(default="", compare=False)
    """Provenance: where gains, offsets, and mount transforms came from."""

    # --- derived -------------------------------------------------------------
    @property
    def fingertip_slots(self) -> tuple[str, ...]:
        """Padded slot names, falling back to the active fingertips."""
        return self.fingertip_slot_names or self.fingertip_body_names

    @property
    def num_fingertip_slots(self) -> int:
        """Observation width for fingertip fields. Equals ``num_fingertips``
        for an unpadded spec, so obs dims are unchanged for fixed hands."""
        return len(self.fingertip_slots)

    @property
    def fingertip_slot_mask(self) -> tuple[bool, ...]:
        """Validity mask over the padded slots; all-true when unpadded."""
        if self.fingertip_slot_active:
            return self.fingertip_slot_active
        return (True,) * self.num_fingertip_slots

    @property
    def fingertip_slot_offsets_padded(self) -> tuple[Vec3, ...]:
        """Offsets over the padded slots; the active table when unpadded."""
        return self.fingertip_slot_offsets or self.fingertip_offsets

    @property
    def joint_names_canonical(self) -> tuple[str, ...]:
        """Canonical policy joint order: arm joints first, then hand joints."""
        return self.arm_joint_names + self.hand_joint_names

    @property
    def num_arm_joints(self) -> int:
        return len(self.arm_joint_names)

    @property
    def num_hand_joints(self) -> int:
        return len(self.hand_joint_names)

    @property
    def num_joints(self) -> int:
        """Action-space size; the env derives cfg.action_space from this."""
        return self.num_arm_joints + self.num_hand_joints

    @property
    def num_fingertips(self) -> int:
        return len(self.fingertip_body_names)

    def arm_default_joint_pos_resolved(self, *, start_arm_higher: bool) -> dict[str, float]:
        """Arm home pose, with the start_arm_higher offsets applied if requested."""
        pose = dict(self.arm_default_joint_pos)
        if start_arm_higher:
            for joint, delta in self.start_arm_higher_deltas.items():
                pose[joint] += delta
        return pose

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Fail loudly on a malformed spec.

        These checks replace the module-level ``assert len(...) == 22`` lines
        that used to sit in scene_utils. Each one guards a failure that is
        otherwise silent: a joint missing from a gain table gets Isaac Lab's
        default gains, a duplicate name corrupts the canonical permutation, and
        an empty adjacency map leaves self-collisions unmasked.
        """
        who = f"RobotSpec({self.name!r})"

        if not self.arm_joint_names:
            raise ValueError(f"{who}: arm_joint_names is empty")
        if not self.hand_joint_names:
            raise ValueError(f"{who}: hand_joint_names is empty")
        if not self.fingertip_body_names:
            raise ValueError(f"{who}: fingertip_body_names is empty")

        canonical = self.joint_names_canonical
        if len(set(canonical)) != len(canonical):
            dupes = sorted({n for n in canonical if canonical.count(n) > 1})
            raise ValueError(f"{who}: duplicate joint names {dupes}")

        overlap = set(self.arm_joint_names) & set(self.hand_joint_names)
        if overlap:
            raise ValueError(f"{who}: joints in both arm and hand: {sorted(overlap)}")

        if len(set(self.fingertip_body_names)) != len(self.fingertip_body_names):
            raise ValueError(f"{who}: duplicate fingertip_body_names")

        if len(self.fingertip_offsets) != self.num_fingertips:
            raise ValueError(
                f"{who}: fingertip_offsets has {len(self.fingertip_offsets)} entries "
                f"but there are {self.num_fingertips} fingertips"
            )
        for i, off in enumerate(self.fingertip_offsets):
            if len(off) != 3:
                raise ValueError(f"{who}: fingertip_offsets[{i}] is not a 3-vector: {off}")

        # Padded slots: either all three tables are supplied and agree, or none are.
        pad = (self.fingertip_slot_names, self.fingertip_slot_active,
               self.fingertip_slot_offsets)
        if any(pad) and not all(pad):
            supplied = [n for n, v in zip(
                ("fingertip_slot_names", "fingertip_slot_active",
                 "fingertip_slot_offsets"), pad) if v]
            raise ValueError(
                f"{who}: padded fingertip slots are partially specified "
                f"(got {supplied}); supply all three or none")
        if self.fingertip_slot_names:
            n_slots = len(self.fingertip_slot_names)
            if len(self.fingertip_slot_active) != n_slots:
                raise ValueError(
                    f"{who}: fingertip_slot_active has "
                    f"{len(self.fingertip_slot_active)} entries for "
                    f"{n_slots} slots")
            if len(self.fingertip_slot_offsets) != n_slots:
                raise ValueError(
                    f"{who}: fingertip_slot_offsets has "
                    f"{len(self.fingertip_slot_offsets)} entries for "
                    f"{n_slots} slots")
            if len(set(self.fingertip_slot_names)) != n_slots:
                raise ValueError(f"{who}: duplicate fingertip_slot_names")
            # The active slots must be exactly the declared fingertips, in the same order -- otherwise the...
            active = tuple(n for n, ok in zip(self.fingertip_slot_names,
                                              self.fingertip_slot_active) if ok)
            if active != tuple(self.fingertip_body_names):
                raise ValueError(
                    f"{who}: active slots {list(active)} do not match "
                    f"fingertip_body_names {list(self.fingertip_body_names)}")
            for i, off in enumerate(self.fingertip_slot_offsets):
                if len(off) != 3:
                    raise ValueError(
                        f"{who}: fingertip_slot_offsets[{i}] is not a 3-vector: {off}")
        if len(self.palm_center_offset) != 3:
            raise ValueError(f"{who}: palm_center_offset is not a 3-vector")

        # Every joint must appear in every table that governs it.
        tables = [
            ("arm_stiffness", self.arm_stiffness, self.arm_joint_names),
            ("arm_damping", self.arm_damping, self.arm_joint_names),
            ("arm_default_joint_pos", self.arm_default_joint_pos, self.arm_joint_names),
            ("hand_stiffness", self.hand_stiffness, self.hand_joint_names),
            ("hand_damping", self.hand_damping, self.hand_joint_names),
            ("hand_armature", self.hand_armature, self.hand_joint_names),
            ("hand_default_joint_pos", self.hand_default_joint_pos, self.hand_joint_names),
        ]
        for label, table, expected in tables:
            missing = [j for j in expected if j not in table]
            extra = [j for j in table if j not in expected]
            if missing or extra:
                raise ValueError(
                    f"{who}: {label} keys do not match its joint list "
                    f"(missing={missing}, unexpected={extra})"
                )

        unknown = [j for j in self.start_arm_higher_deltas if j not in self.arm_joint_names]
        if unknown:
            raise ValueError(f"{who}: start_arm_higher_deltas names non-arm joints {unknown}")

        if not self.adjacent_links:
            raise ValueError(
                f"{who}: adjacent_links is empty. Self-collisions are enabled on the "
                f"articulation, so an empty map leaves the hand colliding with itself "
                f"at every joint."
            )

        if len(self.base_rot) != 4:
            raise ValueError(f"{who}: base_rot is not a wxyz quaternion")


__all__ = ["RobotSpec", "Vec3", "Quat"]


# --- iiwa14 arm names and adjacency ------------------------------------------

import math


ARM_NAME = "iiwa14"

ARM_JOINT_NAMES: tuple[str, ...] = (
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
)

ARM_STIFFNESS: dict[str, float] = {
    "iiwa14_joint_1": 600.0,
    "iiwa14_joint_2": 600.0,
    "iiwa14_joint_3": 500.0,
    "iiwa14_joint_4": 400.0,
    "iiwa14_joint_5": 200.0,
    "iiwa14_joint_6": 200.0,
    "iiwa14_joint_7": 200.0,
}

ARM_DAMPING: dict[str, float] = {
    "iiwa14_joint_1": 27.027026473513512,
    "iiwa14_joint_2": 27.027026473513512,
    "iiwa14_joint_3": 24.672186769721083,
    "iiwa14_joint_4": 22.067474708266914,
    "iiwa14_joint_5": 9.752538131173853,
    "iiwa14_joint_6": 9.147747263670984,
    "iiwa14_joint_7": 9.147747263670984,
}

ARM_DEFAULT_JOINT_POS: dict[str, float] = {
    "iiwa14_joint_1": -1.571,
    "iiwa14_joint_2": 1.571,
    "iiwa14_joint_3": 0.0,
    "iiwa14_joint_4": 1.376,
    "iiwa14_joint_5": 0.0,
    "iiwa14_joint_6": 1.485,
    "iiwa14_joint_7": 1.308,
}

# Applied on top of the home pose when reset.start_arm_higher is set, which the DexToolBench...
START_ARM_HIGHER_DELTAS: dict[str, float] = {
    "iiwa14_joint_2": -math.radians(10.0),
    "iiwa14_joint_4": +math.radians(10.0),
}

# Base placement on the table.
BASE_POS: tuple[float, float, float] = (0.0, 0.8, 0.0)
BASE_ROT: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

# Serial chain, so only consecutive links need filtering.
ARM_ADJACENT_LINKS: dict[str, list[str]] = {
    "iiwa14_link_0": ["iiwa14_link_1"],
    "iiwa14_link_1": ["iiwa14_link_0", "iiwa14_link_2"],
    "iiwa14_link_2": ["iiwa14_link_1", "iiwa14_link_3"],
    "iiwa14_link_3": ["iiwa14_link_2", "iiwa14_link_4"],
    "iiwa14_link_4": ["iiwa14_link_3", "iiwa14_link_5"],
    "iiwa14_link_5": ["iiwa14_link_4", "iiwa14_link_6"],
    "iiwa14_link_6": ["iiwa14_link_5", "iiwa14_link_7"],
}

# The link the hand's palm merges into under merge_fixed_joints.
ARM_TIP_LINK = "iiwa14_link_7"

ARM_LINK_PRIM_REGEX = "/World/envs/env_.*/Robot/iiwa14_link_.*/visuals"


assert len(ARM_JOINT_NAMES) == 7
assert set(ARM_STIFFNESS) == set(ARM_JOINT_NAMES)
assert set(ARM_DAMPING) == set(ARM_JOINT_NAMES)
assert set(ARM_DEFAULT_JOINT_POS) == set(ARM_JOINT_NAMES)


__all__ = [
    "ARM_NAME",
    "ARM_JOINT_NAMES",
    "ARM_STIFFNESS",
    "ARM_DAMPING",
    "ARM_DEFAULT_JOINT_POS",
    "START_ARM_HIGHER_DELTAS",
    "BASE_POS",
    "BASE_ROT",
    "ARM_ADJACENT_LINKS",
    "ARM_TIP_LINK",
    "ARM_LINK_PRIM_REGEX",
]


# --- measured SHARPA constants -----------------------------------------------


# --- capsule tiers --------------------------------------------------------- Radius is half...
TIER_RADIUS_M: dict[str, float] = {
    "mc": 0.019348576781339943,
    "pp": 0.010165722109377384,
    "mp": 0.00893329898826778,
    "dp": 0.007338061812333763,
}

# Nominal length is the URDF *joint spacing*, not the mesh extent.
TIER_NOMINAL_LENGTH_M: dict[str, float] = {
    "mc": 0.0661,   # thumb CMC_AA -> MCP_FE
    "pp": 0.0470,   # MCP_AA -> PIP
    "mp": 0.0315,   # PIP -> DIP
    "dp": 0.0260,   # DIP -> fingertip
}

# Measured link masses the densities are fitted to.
TIER_MASS_KG: dict[str, float] = {
    "mc": 0.116,      # left_thumb_MC
    "pp": 0.04138,    # left_index_PP
    "mp": 0.02509,    # left_index_MP
    "dp": 0.004103,   # left_index_DP + _elastomer + _fingertip
}

# rho = mass / capsule_volume(nominal_length, radius), where the capsule's TOTAL length --...
TIER_DENSITY_KG_M3: dict[str, float] = {
    "mc": 1853.918713613545,
    "pp": 3168.7741471413146,
    "mp": 3917.691174722447,
    "dp": 1149.0599166926079,
}

# The metacarpal density comes from the THUMB, not the pinky.
MC_MIN_LENGTH_M: float = 0.005

# --- palm ------------------------------------------------------------------ Modelled as a...
PALM_EXTENTS_M: tuple[float, float, float] = (
    0.04989549145102501,
    0.08517111465334892,
    0.08640973269939425,
)
PALM_MASS_KG: float = 0.72045
PALM_DENSITY_KG_M3: float = 1961.9482523719728

# Where the palm box sits in the palm link frame.
PALM_BOX_CENTER_M: tuple[float, float, float] = (0.00034, -0.00109, 0.04320)

# --- actuation, per joint slot --------------------------------------------- URDF...
SLOT_EFFORT_NM: dict[str, float] = {
    "CMC_FE": 3.3,
    "CMC_AA": 3.3,
    "MCP_FE": 1.864,
    "MCP_AA": 1.864,
    "PIP": 0.638,
    "DIP": 0.189369,
}
SLOT_VELOCITY_RAD_S: dict[str, float] = {
    "CMC_FE": 11.84076833,
    "CMC_AA": 11.84076833,
    "MCP_FE": 16.07692878,
    "MCP_AA": 16.07692878,
    "PIP": 11.61831603,
    "DIP": 14.66594768,
}

# Per-slot PD gains, damping and armature, from the SHARPA spec's per-tier values...
SLOT_STIFFNESS: dict[str, float] = {
    "CMC_FE": 1.38, "CMC_AA": 1.38,
    "MCP_FE": 4.76, "MCP_AA": 6.62,
    "PIP": 0.9, "DIP": 0.9,
}
SLOT_DAMPING: dict[str, float] = {
    "CMC_FE": 0.02782345, "CMC_AA": 0.02782345,
    "MCP_FE": 0.20859232, "MCP_AA": 0.24595532,
    "PIP": 0.04243185, "DIP": 0.03504461,
}
SLOT_ARMATURE: dict[str, float] = {
    "CMC_FE": 0.0032, "CMC_AA": 0.0032,
    "MCP_FE": 0.00265, "MCP_AA": 0.00265,
    "PIP": 0.0006, "DIP": 0.00042,
}

# --- virtual / ghost links ------------------------------------------------- SHARPA already...
VIRTUAL_LINK_MASS_KG: float = 1e-6
VIRTUAL_LINK_INERTIA: float = 1e-6


# --- flange -> palm -------------------------------------------------------- SHARPA reaches...
FLANGE_TO_PALM_Z_M: float = 0.05
FLANGE_TO_PALM_YAW_RAD: float = -1.3089969389957472   # -75 deg


def cylinder_part(total_length: float, radius: float) -> float:
    """Cylindrical section of a capsule whose TOTAL length is ``total_length``.

    One definition, shared by the URDF builder and the self-collision checker,
    so the shape they assume cannot drift from the shape PhysX simulates.
    """
    return max(total_length - 2.0 * radius, 0.0)


__all__ = [
    "cylinder_part",
    "TIER_RADIUS_M",
    "TIER_NOMINAL_LENGTH_M",
    "TIER_MASS_KG",
    "TIER_DENSITY_KG_M3",
    "MC_MIN_LENGTH_M",
    "PALM_EXTENTS_M",
    "PALM_MASS_KG",
    "PALM_DENSITY_KG_M3",
    "PALM_BOX_CENTER_M",
    "SLOT_EFFORT_NM",
    "SLOT_VELOCITY_RAD_S",
    "SLOT_STIFFNESS",
    "SLOT_DAMPING",
    "SLOT_ARMATURE",
    "VIRTUAL_LINK_MASS_KG",
    "VIRTUAL_LINK_INERTIA",
    "FLANGE_TO_PALM_Z_M",
    "FLANGE_TO_PALM_YAW_RAD",
]


# --- URDF splicing and mesh rewriting ----------------------------------------

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from hand_sampler import resolve as resolve_repo_path
from hand_sampler.geometry import mat_to_rpy, rpy_to_mat


SHARPA_URDF = "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
ALLEGRO_SRC = "assets/urdf/kuka_allegro_iiwa14_description/_source/kuka_allegro.urdf"
OUT_URDF = "assets/urdf/kuka_allegro_iiwa14_description/iiwa14_allegro.urdf"

# Mirrored (left-hand) mesh copies live here; generated on demand.
MIRRORED_MESH_DIR = "allegro_meshes_mirrored"

# The iiwa14 chain to keep, verbatim, so the arm is byte-identical to SHARPA's.
ARM_LINKS = tuple(f"iiwa14_link_{i}" for i in range(8)) + ("iiwa14_link_ee",)
ARM_JOINTS = tuple(f"iiwa14_joint_{i}" for i in range(1, 8)) + ("iiwa14_joint_ee",)

FINGERS = ("index", "middle", "ring", "thumb")
HAND_LINKS = ("allegro_mount", "palm_link") + tuple(
    f"{f}_link_{i}" for f in FINGERS for i in range(4)
)
HAND_JOINTS = ("allegro_mount_joint",) + tuple(
    f"{f}_joint_{i}" for f in FINGERS for i in range(4)
)

# Flange offsets from each arm's own URDF.
IIWA7_FLANGE_TO_EE_Z = 0.071
IIWA14_FLANGE_TO_EE_Z = 0.045
MOUNT_Z = IIWA7_FLANGE_TO_EE_Z - IIWA14_FLANGE_TO_EE_Z  # 0.026

# Rotation of the hand about the flange axis.
MOUNT_YAW = math.radians(150.0)

# Allegro meshes were copied out of the isaacgym asset tree; rewrite its package-rooted prefix...
MESH_PREFIX_FROM = "kuka_allegro_description/meshes/"
MESH_PREFIX_TO = "allegro_meshes/"
# The arm meshes live in the SHARPA asset directory, so the generated URDF reaches them with a...
ARM_MESH_PREFIX_TO = "../kuka_sharpa_description/"




def _mirror_hand(root: ET.Element, hand_links, hand_joints, mesh_map: dict) -> None:
    """Reflect the hand subtree about the y=0 plane of its mount frame.

    The stock Allegro asset is a RIGHT hand. Its own comment says a left hand
    only needs the sign of each finger's y offset and splay angle flipped, but
    that is incomplete: the palm and thumb-base meshes are not mirror-symmetric
    (34.8 mm and 30.3 mm maximum residual when reflected about y), so that
    recipe yields left-hand kinematics wearing a right-hand palm. This applies
    the full reflection M = diag(1, -1, 1) to origins, rotations, axes, and
    geometry.

    Under a reflection a rotation R maps to M R M -- still a rotation, since
    det(M R M) = det(R) = 1. A revolute axis is a pseudovector, so it maps to
    -M a rather than M a; that extra sign keeps a positive joint angle meaning
    the same motion (flexion stays flexion), which matters because the limits
    are asymmetric and are carried over unchanged.
    """
    import numpy as np

    M = np.diag([1.0, -1.0, 1.0])

    def mirror_origin(el: ET.Element) -> None:
        xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
        el.set("xyz", " ".join(f"{v:.9g}" for v in (M @ np.array(xyz))))
        el.set("rpy", " ".join(f"{v:.9g}" for v in mat_to_rpy(M @ rpy_to_mat(rpy) @ M)))

    for joint in root.findall("joint"):
        if joint.get("name") not in hand_joints:
            continue
        origin = joint.find("origin")
        if origin is not None:
            mirror_origin(origin)
        axis = joint.find("axis")
        if axis is not None:
            a = np.array([float(v) for v in axis.get("xyz", "0 0 1").split()])
            axis.set("xyz", " ".join(f"{v:.9g}" for v in (-(M @ a))))

    for link in root.findall("link"):
        if link.get("name") not in hand_links:
            continue
        for tag in ("visual", "collision", "inertial"):
            for el in link.findall(tag):
                o = el.find("origin")
                if o is not None:
                    mirror_origin(o)
        # Point every mesh at its mirrored copy.
        for mesh in link.iter("mesh"):
            fn = mesh.get("filename", "")
            if fn in mesh_map:
                mesh.set("filename", mesh_map[fn])


def _write_mirrored_meshes(urdf_dir, hand_mesh_names) -> dict:
    """Mirror each hand mesh about y and fix winding; return old -> new paths."""
    import numpy as np
    import trimesh

    out_dir = urdf_dir / MIRRORED_MESH_DIR
    mapping = {}
    for rel in sorted(hand_mesh_names):
        src = urdf_dir / rel
        if not src.exists():
            raise SystemExit(f"mesh not found while mirroring: {src}")
        m = trimesh.load(src, force="mesh")
        V = np.asarray(m.vertices).copy()
        V[:, 1] *= -1.0
        F = np.asarray(m.faces).copy()
        # Reflection flips orientation; reverse winding so normals point out.
        F = F[:, ::-1]
        dst_rel = f"{MIRRORED_MESH_DIR}/{pathlib_name(rel)}"
        dst = urdf_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=V, faces=F, process=False).export(dst)
        mapping[rel] = dst_rel
    return mapping


def pathlib_name(rel: str) -> str:
    """Flatten a nested mesh path into a unique single filename."""
    return rel.replace("/", "__")


def _indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not (child.tail or "").strip():
            child.tail = pad
    if level and not (elem.tail or "").strip():
        elem.tail = pad


def _rewrite_meshes(elem: ET.Element, frm: str, to: str) -> int:
    n = 0
    for mesh in elem.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith(frm):
            mesh.set("filename", to + fn[len(frm):])
            n += 1
        elif frm == "" and not fn.startswith(to) and not fn.startswith("/"):
            mesh.set("filename", to + fn)
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount_yaw", type=float, default=MOUNT_YAW,
                        help="Rotation of the hand about the flange axis, radians. "
                             "Sets where the thumb points relative to the arm. "
                             f"Default {MOUNT_YAW:.4f} rad "
                             f"({math.degrees(MOUNT_YAW):.0f} deg), chosen visually.")
    parser.add_argument("--mount_z", type=float, default=MOUNT_Z)
    parser.add_argument("--out", default=OUT_URDF)
    parser.add_argument("--hand", choices=("right", "left"), default="left",
                        help="Handedness. The stock asset is right-handed; 'left' "
                             "mirrors the hand subtree AND its meshes so it matches "
                             "SHARPA, which is a left hand. Default left.")
    args = parser.parse_args()

    sharpa = ET.parse(resolve_repo_path(SHARPA_URDF)).getroot()
    allegro = ET.parse(resolve_repo_path(ALLEGRO_SRC)).getroot()

    out = ET.Element("robot", {"name": f"iiwa14_allegro_{args.hand}"})
    out.append(ET.Comment(
        " GENERATED by hand_sampler/allegro_urdf.py; do not hand-edit.\n"
        "     arm:  iiwa14 chain copied verbatim from the SHARPA URDF, so the arm is\n"
        "           identical across hands (docs/methodology.md 1).\n"
        "     hand: Allegro subtree from the stock kuka_allegro.urdf (iiwa7 asset).\n"
        f"     mount: iiwa14_link_ee -> allegro_mount at z={args.mount_z:.4f}\n"
        f"            (0.071 iiwa7 flange-to-ee minus 0.045 iiwa14 flange-to-ee), so the\n"
        f"            shipped flange-to-palm geometry is reproduced exactly.\n"
        f"     mount_yaw = {args.mount_yaw:.6f} rad "
        f"({math.degrees(args.mount_yaw):.1f} deg), chosen visually.\n"
        f"     handedness = {args.hand}\n"
    ))

    # Materials from both sources, first definition wins.
    seen_materials: set[str] = set()
    for src in (sharpa, allegro):
        for mat in src.findall("material"):
            name = mat.get("name")
            if name and name not in seen_materials:
                seen_materials.add(name)
                out.append(mat)

    # --- arm, verbatim ---
    arm_n = 0
    for link in sharpa.findall("link"):
        if link.get("name") in ARM_LINKS:
            _rewrite_meshes(link, "", ARM_MESH_PREFIX_TO)
            out.append(link)
            arm_n += 1
    for joint in sharpa.findall("joint"):
        if joint.get("name") in ARM_JOINTS:
            out.append(joint)

    # --- the graft ---
    mount = ET.SubElement(out, "joint", {"name": "iiwa14_allegro", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": "iiwa14_link_ee"})
    ET.SubElement(mount, "child", {"link": "allegro_mount"})
    ET.SubElement(mount, "origin", {
        "xyz": f"0 0 {args.mount_z}",
        "rpy": f"0 0 {args.mount_yaw}",
    })

    # --- hand ---
    hand_n = mesh_n = 0
    for link in allegro.findall("link"):
        if link.get("name") in HAND_LINKS:
            mesh_n += _rewrite_meshes(link, MESH_PREFIX_FROM, MESH_PREFIX_TO)
            out.append(link)
            hand_n += 1
    hand_j = 0
    for joint in allegro.findall("joint"):
        if joint.get("name") in HAND_JOINTS:
            out.append(joint)
            hand_j += 1

    if args.hand == "left":
        urdf_dir = resolve_repo_path(args.out).parent
        hand_meshes = {
            mesh.get("filename")
            for link in out.findall("link") if link.get("name") in HAND_LINKS
            for mesh in link.iter("mesh")
        }
        mesh_map = _write_mirrored_meshes(urdf_dir, hand_meshes)
        _mirror_hand(out, set(HAND_LINKS), set(HAND_JOINTS), mesh_map)
        print(f"[build]   mirrored hand subtree + {len(mesh_map)} meshes -> LEFT hand")

    missing_links = set(ARM_LINKS + HAND_LINKS) - {
        l.get("name") for l in out.findall("link")
    }
    missing_joints = set(ARM_JOINTS + HAND_JOINTS + ("iiwa14_allegro",)) - {
        j.get("name") for j in out.findall("joint")
    }
    if missing_links or missing_joints:
        raise SystemExit(
            f"splice incomplete: missing links {sorted(missing_links)}, "
            f"joints {sorted(missing_joints)}"
        )

    _indent(out)
    path = resolve_repo_path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(out).write(path, encoding="utf-8", xml_declaration=True)

    actuated = [j.get("name") for j in out.findall("joint") if j.get("type") == "revolute"]
    print(f"[build] wrote {path}")
    print(f"[build]   arm links {arm_n}, hand links {hand_n}, hand joints {hand_j}, "
          f"{mesh_n} allegro mesh paths rewritten")
    print(f"[build]   {len(actuated)} actuated joints: "
          f"{actuated[:7]} + {actuated[7:]}")
    print(f"[build]   mount: iiwa14_link_ee -> allegro_mount "
          f"xyz=(0,0,{args.mount_z}) rpy=(0,0,{args.mount_yaw})")


if __name__ == "__main__":
    main()
