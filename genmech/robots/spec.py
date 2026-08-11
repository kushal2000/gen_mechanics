"""RobotSpec: everything the task needs to know about a specific arm+hand.

Before this existed, the SHARPA hand was hardcoded across ~9 sites — joint-name
tuples, joint regexes, PD-gain tables, palm and fingertip geometry offsets,
self-collision adjacency, and a literal ``action_space = 29``. The only
config knob was a URDF path, and pointing it at a different robot tripped five
asserts and silently corrupted four more things.

A spec is a frozen, self-validating description of one robot. The task reads
everything hardware-specific from it, so adding a hand is a new module plus a
registry line rather than an edit to the env internals.

Two design points that matter for the study:

**Joint names, not regexes.** The old code found joints with
``ARM_JOINT_REGEX = "iiwa14_joint_.*"`` and bodies with a fingertip regex, then
relied on whatever order Isaac Lab's matcher returned. Specs list exact names in
canonical order and look them up with ``preserve_order=True``, so ordering is
declared rather than inherited from a regex engine.

**Arm tables are shared, not copied.** The whole comparison rests on the arm
being identical across hands, so arm joint names, gains, and home pose live in
one module (``iiwa14_arm.py``) that every spec imports. A spec cannot quietly
bring its own arm, and ``test_robot_spec_invariants.py`` asserts equality across
the registry. See docs/methodology.md §1.
"""

from __future__ import annotations

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

    notes: str = field(default="", compare=False)
    """Provenance: where gains, offsets, and mount transforms came from."""

    # --- derived -------------------------------------------------------------
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
