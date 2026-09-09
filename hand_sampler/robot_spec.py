"""What the task needs to know about a specific arm+hand."""

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
    """Hand family, e.g. "sharpa"."""
    urdf_path: str
    """Repo-relative URDF path; resolved against REPO_ROOT, not the CWD."""

    # --- joints (ORDERED; together they define canonical policy order) -------
    arm_joint_names: tuple[str, ...]
    hand_joint_names: tuple[str, ...]

    # --- bodies -------------------------------------------------------------
    palm_body_name: str
    """Palm body name AFTER merge_fixed_joints."""
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
    """Radian offsets applied to the arm home pose when reset.start_arm_higher is set (the..."""

    # --- observation geometry ----------------------------------------------
    palm_center_offset: Vec3

    # --- physics ------------------------------------------------------------
    adjacent_links: Mapping[str, list[str]]
    """Link pairs whose self-collision is filtered out, in POST-merge body names."""

    # --- scene prim patterns -------------------------------------------------
    link_prim_regexes: tuple[str, ...]
    """Prim-path patterns matching this robot's visual meshes, for the depth raycaster and..."""

    # --- base placement ------------------------------------------------------
    base_pos: Vec3 = (0.0, 0.8, 0.0)
    base_rot: Quat = (1.0, 0.0, 0.0, 0.0)

    # --- asset conversion ----------------------------------------------------
    replace_cylinders_with_capsules: bool = False
    """Convert ``<cylinder>`` collision geometry to PhysX capsules on import."""




    # --- joint tokens -------------------------------------------------------
    # The policy's whole view of the hand: four points per joint in its child
    # link's frame. Carried here rather than derived at run start, so the env
    # never asks where a hand came from -- SHARPA imports them from its URDF
    # once, a generated hand computes them from its tree.
    joint_link_bodies: tuple[str, ...] = ()
    """Child body of each hand joint, in hand_joint_names order."""
    joint_link_boxes: tuple = ()
    """(J, 4, 3) nested tuples: p0, and its three adjacent corners."""
    joint_geometry_valid: tuple[bool, ...] = ()
    hand_scale: float = 0.0
    """Longest encoded edge, in metres. Also given to the policy."""

    # Coulomb friction at the joint. simtoolreal set this and the first port
    # dropped it, so SHARPA ran frictionless here where the reference did not.
    hand_friction: Mapping[str, float] = field(default_factory=dict)
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
        """Fail loudly on a malformed spec."""
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

        if self.joint_link_bodies:
            n = self.num_hand_joints
            for name, got in (("joint_link_bodies", len(self.joint_link_bodies)),
                              ("joint_link_boxes", len(self.joint_link_boxes)),
                              ("joint_geometry_valid", len(self.joint_geometry_valid))):
                if got != n:
                    raise ValueError(f"{who}: {name} has {got} entries for {n} hand joints")
            if self.hand_scale <= 0.0:
                raise ValueError(f"{who}: hand_scale must be positive, got {self.hand_scale}")

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




def robot_spec_from_hand(hand, *, name: str, urdf_path: str = "",
                         hand_name: str = "generated") -> RobotSpec:
    """Project a ``design_space.Hand`` onto the flat description the env reads.

    The Hand is the single source: structure, geometry, joint travel and drive.
    This derives the joint-name-keyed view Isaac Lab wants, so a generated hand
    needs no URDF -- which is the point, since we do not maintain one for it.

    Joint names are positional, ``f{finger}_j{segment}``, because a generated
    design has no anatomy to name them after. Order is finger by finger,
    proximal to distal, matching ``design_space.joint_boxes``.
    """
    from hand_sampler import design_space
    from hand_sampler import robot_param_constants as rpc

    boxes, valid, scale = design_space.joint_boxes(hand)

    names, stiffness, damping, armature, friction, tips = [], {}, {}, {}, {}, []
    for f, finger in enumerate(hand.fingers):
        for d, seg in enumerate(finger.segments):
            jn = f"f{f}_j{d}"
            names.append(jn)
            _e, _v, k, b, a, fr = seg.joint.drive or rpc.gen_joint_drive(d, seg.joint.theta)
            stiffness[jn], damping[jn], armature[jn], friction[jn] = k, b, a, fr
        tips.append(f"f{f}_link{finger.n_joints - 1}")

    return RobotSpec(
        name=name, arm_name=rpc.ARM_NAME, hand_name=hand_name, urdf_path=urdf_path,
        arm_joint_names=rpc.ARM_JOINT_NAMES, hand_joint_names=tuple(names),
        palm_body_name=rpc.ARM_TIP_LINK, fingertip_body_names=tuple(tips),
        arm_stiffness=rpc.ARM_STIFFNESS, arm_damping=rpc.ARM_DAMPING,
        hand_stiffness=stiffness, hand_damping=damping, hand_armature=armature,
        hand_friction=friction,
        joint_link_bodies=tuple(f"{n.split('_j')[0]}_link{n.split('_j')[1]}" for n in names),
        joint_link_boxes=tuple(tuple(map(tuple, b)) for b in boxes),
        joint_geometry_valid=tuple(bool(v) for v in valid), hand_scale=float(scale),
        arm_default_joint_pos=rpc.ARM_DEFAULT_JOINT_POS,
        hand_default_joint_pos={n: 0.0 for n in names},
        start_arm_higher_deltas=rpc.START_ARM_HIGHER_DELTAS,
        palm_center_offset=(0.0, 0.0, 0.0),
        adjacent_links=dict(rpc.ARM_ADJACENT_LINKS),
        link_prim_regexes=(".*",),
        base_pos=rpc.BASE_POS, base_rot=rpc.BASE_ROT,
        notes=f"derived from a Hand: {hand.n_fingers} fingers, {hand.n_joints} joints",
    )


@dataclass(frozen=True)
class HandPopulation:
    """Many designs sharing one articulation template.

    A scene is one articulation with one DOF count, so every design is authored
    into the same ``MAX_FINGERS x MAX_JOINTS_PER_FINGER`` envelope and the ones
    it does not use are locked: equal limits, zero-length links. ``spec`` is that
    template, identical for every env; the arrays are per design, gathered by
    env at reset.

    Ghost links are zero length, so finger ``i``'s template tip body sits exactly
    where its real tip does -- but a ghost FINGER's tip sits at the palm, which
    is why ``fingertip_valid`` exists.
    """

    spec: RobotSpec
    hands: tuple
    joint_link_boxes: "np.ndarray"   # (n, J, 4, 3)
    joint_valid: "np.ndarray"        # (n, J) bool
    joint_limits: "np.ndarray"       # (n, J, 2)
    hand_scale: "np.ndarray"         # (n,)
    fingertip_valid: "np.ndarray"    # (n, F) bool

    @property
    def n_designs(self) -> int:
        return len(self.hands)

    def per_env(self, index) -> dict:
        """Gather the per-design tables onto ``(N, ...)``, one row per env.

        ``index`` is ``(N,)`` design ids. Kept here rather than in the env so it
        can be checked without booting a simulator: the shapes and the masking
        are where a padded population goes silently wrong.
        """
        import numpy as np
        idx = np.asarray(index, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError(f"design index must be 1-D, got shape {idx.shape}")
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_designs):
            raise ValueError(
                f"design index out of range for {self.n_designs} designs: "
                f"[{idx.min()}, {idx.max()}]")
        return {
            "joint_link_bbox_local": self.joint_link_boxes[idx],   # (N, J, 4, 3)
            "joint_geometry_valid": self.joint_valid[idx],         # (N, J)
            "joint_limits": self.joint_limits[idx],                # (N, J, 2)
            "hand_scale": self.hand_scale[idx][:, None],           # (N, 1)
            "fingertip_valid": self.fingertip_valid[idx],          # (N, F)
        }


def population_spec(hands, *, name: str = "generated_population") -> HandPopulation:
    """Project many ``Hand`` trees onto one template plus per-design tables."""
    import numpy as np

    from hand_sampler import design_space
    from hand_sampler import robot_param_constants as rpc

    F, D = design_space.MAX_FINGERS, design_space.MAX_JOINTS_PER_FINGER
    J = F * D
    slot = lambda f, d: f * D + d

    names = tuple(f"f{f}_j{d}" for f in range(F) for d in range(D))
    tips = tuple(f"f{f}_link{D - 1}" for f in range(F))
    # Every generated joint has the same actuator, so the template carries the
    # gains and no per-design override is needed -- only geometry and limits.
    e, v, k, b, a, fr = rpc.gen_joint_drive()
    spec = RobotSpec(
        name=name, arm_name=rpc.ARM_NAME, hand_name="generated", urdf_path="",
        arm_joint_names=rpc.ARM_JOINT_NAMES, hand_joint_names=names,
        palm_body_name=rpc.ARM_TIP_LINK, fingertip_body_names=tips,
        arm_stiffness=rpc.ARM_STIFFNESS, arm_damping=rpc.ARM_DAMPING,
        hand_stiffness={n: k for n in names}, hand_damping={n: b for n in names},
        hand_armature={n: a for n in names}, hand_friction={n: fr for n in names},
        arm_default_joint_pos=rpc.ARM_DEFAULT_JOINT_POS,
        hand_default_joint_pos={n: 0.0 for n in names},
        start_arm_higher_deltas=rpc.START_ARM_HIGHER_DELTAS,
        palm_center_offset=(0.0, 0.0, 0.0),
        adjacent_links=dict(rpc.ARM_ADJACENT_LINKS),
        link_prim_regexes=(".*",),
        base_pos=rpc.BASE_POS, base_rot=rpc.BASE_ROT,
        joint_link_bodies=tuple(f"f{f}_link{d}" for f in range(F) for d in range(D)),
        joint_link_boxes=tuple(((0.0,) * 3,) * 4 for _ in range(J)),
        joint_geometry_valid=(False,) * J, hand_scale=1.0,
        notes=f"template for {len(hands)} designs, envelope {F}x{D}={J}",
    )

    n = len(hands)
    boxes = np.zeros((n, J, 4, 3), dtype=np.float32)
    valid = np.zeros((n, J), dtype=bool)
    limits = np.zeros((n, J, 2), dtype=np.float32)
    limits[..., 1] = 1e-8   # ghosts: locked, and not exactly coincident
    scale = np.zeros((n,), dtype=np.float32)
    ft_valid = np.zeros((n, F), dtype=bool)

    for i, hand in enumerate(hands):
        if hand.n_fingers > F:
            raise ValueError(f"design {i} has {hand.n_fingers} fingers, envelope allows {F}")
        b_i, v_i, s_i = design_space.joint_boxes(hand)
        scale[i] = s_i
        seen = 0
        for f, finger in enumerate(hand.fingers):
            if finger.n_joints > D:
                raise ValueError(
                    f"design {i} finger {f} has {finger.n_joints} joints, envelope allows {D}")
            ft_valid[i, f] = True
            for d, seg in enumerate(finger.segments):
                s = slot(f, d)
                boxes[i, s] = b_i[seen]
                valid[i, s] = bool(v_i[seen])
                limits[i, s] = seg.joint.limits or design_space.JOINT_LIMIT
                seen += 1
    return HandPopulation(spec=spec, hands=tuple(hands), joint_link_boxes=boxes,
                          joint_valid=valid, joint_limits=limits, hand_scale=scale,
                          fingertip_valid=ft_valid)


def design_index(n_envs: int, n_designs: int) -> "np.ndarray":
    """Which design each env holds: ``i % n_designs``, so every design appears
    an equal number of times up to the remainder, and env 0 always holds
    design 0. Deterministic, because a run has to be reproducible and a
    scrambled assignment would also scramble the reward attribution."""
    import numpy as np
    if n_designs <= 0:
        raise ValueError("a population needs at least one design")
    return np.arange(n_envs, dtype=np.int64) % n_designs
