"""Hand-only ``RobotSpec``-shaped objects: no arm, palm body = hand_root.

``hand_sampler.robot_spec.RobotSpec`` stays untouched (its ``validate()``
requires a non-empty ``arm_joint_names``, and the plan is explicit that file
does not change). ``HandOnlySpec`` here is a separate, duck-typed stand-in:
every field the scene/articulation code actually reads
(``isaacsimenvs.pose_reaching_6d.scene_utils.assembly.build_robot_articulation_cfg``
and ``_convert_fixed_robot``, both spec-generic) is present with the same
name, with the arm fields simply empty.

Pure stdlib + numpy at the top level (``design_space.joint_link_boxes`` needs
numpy and, only inside its mesh branch, ``trimesh`` -- lazily imported
there). No torch, no Isaac: this runs under system python3 wherever numpy is
available, or the venv otherwise (see the package docstring in
``urdf_cutter.py`` for why importing anything under ``isaacsimenvs`` at all
pulls in gymnasium).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .urdf_cutter import (
    CutURDF, cut_urdf_to_hand, leaf_link_names, merged_body_name, movable_joint_names,
)

__all__ = [
    "HandOnlySpec", "ManifestHand", "manifest_entry", "resolve_hand_urdf",
    "build_hand_only_spec", "DEFAULT_HAND_STIFFNESS", "DEFAULT_HAND_DAMPING",
    "DEFAULT_HAND_ARMATURE",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAMMAR_BENCH_DIR = REPO_ROOT / "hand_sampler" / "grammar_bench"
MANIFEST_PATH = GRAMMAR_BENCH_DIR / "manifest.json"

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def _sharpa_gain_defaults() -> tuple[float, float, float]:
    """Mean of SHARPA's transcribed per-joint stiffness/damping/armature --
    "reuse SHARPA's hand gains as defaults" for a hand with no anatomical
    correspondence to SHARPA's named joints."""
    from hand_sampler.robot_param_constants import HAND_ARMATURE, HAND_DAMPING, HAND_STIFFNESS

    mean = lambda d: sum(d.values()) / len(d)
    return mean(HAND_STIFFNESS), mean(HAND_DAMPING), mean(HAND_ARMATURE)


DEFAULT_HAND_STIFFNESS, DEFAULT_HAND_DAMPING, DEFAULT_HAND_ARMATURE = _sharpa_gain_defaults()


@dataclass(frozen=True)
class HandOnlySpec:
    """A frozen, no-arm stand-in for ``RobotSpec``. Field names match what
    ``scene_utils.assembly`` reads off a spec; ``arm_*`` fields are always
    empty so ``build_robot_articulation_cfg`` authors zero arm actuators."""

    name: str
    hand_name: str
    urdf_path: str
    """Path to the CUT, mesh-resolved URDF (not the original fixture)."""
    hand_root: str

    hand_joint_names: tuple[str, ...]
    hand_joint_limits: tuple[tuple[float, float], ...]
    palm_body_name: str
    fingertip_body_names: tuple[str, ...]

    hand_stiffness: Mapping[str, float]
    hand_damping: Mapping[str, float]
    hand_armature: Mapping[str, float]
    hand_default_joint_pos: Mapping[str, float]

    palm_center_offset: Vec3
    adjacent_links: Mapping[str, list[str]]
    link_prim_regexes: tuple[str, ...] = (".*",)

    base_pos: Vec3 = (0.0, 0.0, 0.5)
    base_rot: Quat = (1.0, 0.0, 0.0, 0.0)
    replace_cylinders_with_capsules: bool = False

    joint_link_bodies: tuple[str, ...] = ()
    joint_link_boxes: tuple = ()
    joint_geometry_valid: tuple[bool, ...] = ()
    hand_scale: float = 1.0
    palm_keypoints: tuple = ()
    fingertip_offsets: tuple = ()
    notes: str = field(default="", compare=False)

    # --- empty arm surface, kept only so spec-generic code that reads these
    # names (build_robot_articulation_cfg) sees a well-formed, empty table.
    arm_name: str = "none"
    arm_joint_names: tuple[str, ...] = ()
    arm_stiffness: Mapping[str, float] = field(default_factory=dict)
    arm_damping: Mapping[str, float] = field(default_factory=dict)
    arm_default_joint_pos: Mapping[str, float] = field(default_factory=dict)
    start_arm_higher_deltas: Mapping[str, float] = field(default_factory=dict)

    def arm_default_joint_pos_resolved(self, *, start_arm_higher: bool = False) -> dict[str, float]:
        return {}

    @property
    def joint_names_canonical(self) -> tuple[str, ...]:
        return self.hand_joint_names

    @property
    def num_arm_joints(self) -> int:
        return 0

    @property
    def num_hand_joints(self) -> int:
        return len(self.hand_joint_names)

    @property
    def num_joints(self) -> int:
        return self.num_hand_joints

    @property
    def num_fingertips(self) -> int:
        return len(self.fingertip_body_names)

    def validate(self) -> None:
        who = f"HandOnlySpec({self.name!r})"
        if not self.hand_joint_names:
            raise ValueError(f"{who}: hand_joint_names is empty")
        if len(set(self.hand_joint_names)) != len(self.hand_joint_names):
            raise ValueError(f"{who}: duplicate hand_joint_names")
        if len(self.hand_joint_limits) != len(self.hand_joint_names):
            raise ValueError(f"{who}: hand_joint_limits does not match hand_joint_names")
        if not self.fingertip_body_names:
            raise ValueError(f"{who}: fingertip_body_names is empty")
        for label, table in (("hand_stiffness", self.hand_stiffness),
                              ("hand_damping", self.hand_damping),
                              ("hand_armature", self.hand_armature),
                              ("hand_default_joint_pos", self.hand_default_joint_pos)):
            missing = [j for j in self.hand_joint_names if j not in table]
            if missing:
                raise ValueError(f"{who}: {label} is missing {missing}")
        if not self.adjacent_links:
            raise ValueError(f"{who}: adjacent_links is empty (self-collision would be unfiltered)")

    def __post_init__(self) -> None:
        self.validate()


@dataclass(frozen=True)
class ManifestHand:
    hand_id: str
    hand_root: str | None
    urdf_path: Path
    """Absolute path to the SOURCE (uncut) URDF."""


def manifest_entry(hand_id: str) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    for h in manifest["hands"]:
        if h["id"] == hand_id:
            return h
    raise KeyError(f"{hand_id!r} is not in {MANIFEST_PATH}")


def resolve_hand_urdf(hand_id: str) -> ManifestHand:
    """Which URDF and hand_root a manifest hand id uses.

    SHARPA is special-cased to the repo asset
    (``hand_sampler.robot_param_constants.SHARPA_URDF``): its manifest entry
    (``sharpa_left_on_iiwa14``) carries no ``fixture_path`` because it is not
    a grammar_bench fixture, only the ``hand_root`` annotation.
    """
    if hand_id in ("sharpa", "sharpa_left_on_iiwa14"):
        from hand_sampler.robot_param_constants import SHARPA_URDF

        entry = manifest_entry("sharpa_left_on_iiwa14")
        return ManifestHand(hand_id="sharpa", hand_root=entry["hand_root"],
                            urdf_path=(REPO_ROOT / SHARPA_URDF).resolve())

    entry = manifest_entry(hand_id)
    fixture = entry.get("fixture_path")
    if not fixture:
        raise ValueError(
            f"{hand_id!r} has no fixture_path in the manifest (commit_allowed="
            f"{entry.get('commit_allowed')}); its URDF is not available locally")
    return ManifestHand(hand_id=hand_id, hand_root=entry["hand_root"],
                        urdf_path=(GRAMMAR_BENCH_DIR / fixture).resolve())


def _joint_limits(root: ET.Element, joint_names: tuple[str, ...]) -> tuple[tuple[float, float], ...]:
    joints = {j.get("name"): j for j in root.findall("joint")}
    out = []
    for name in joint_names:
        limit = joints[name].find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            out.append((-3.1416, 3.1416))  # continuous / unlimited: a generous default
        else:
            out.append((float(limit.get("lower")), float(limit.get("upper"))))
    return tuple(out)


def _self_adjacent_links(root: ET.Element, joint_names: tuple[str, ...]) -> dict[str, list[str]]:
    """Filter consecutive parent/child pairs on every kept joint -- the same
    minimal adjacency ``ARM_ADJACENT_LINKS`` encodes for the arm's serial
    chain, generalised to however many joints share a parent (a multi-finger
    palm)."""
    adjacency: dict[str, list[str]] = {}
    for j in root.findall("joint"):
        if j.get("name") not in joint_names and j.get("type") != "fixed":
            continue
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        adjacency.setdefault(parent, []).append(child)
        adjacency.setdefault(child, []).append(parent)
    return adjacency


def build_hand_only_spec(
    hand_id: str, *, out_dir: str | Path, name: str | None = None,
) -> tuple[HandOnlySpec, CutURDF]:
    """Cut ``hand_id``'s URDF and project it onto a ``HandOnlySpec``.

    ``out_dir`` is where the cut, mesh-resolved URDF is written (a scene-setup
    temp dir in production; a pytest ``tmp_path`` in tests). Gains: SHARPA
    uses its own transcribed per-joint table; every other hand gets the
    ``DEFAULT_HAND_*`` constants (SHARPA's mean), since it has no joints
    anatomically corresponding to SHARPA's names.
    """
    from hand_sampler.design_space import joint_link_boxes

    manifest_hand = resolve_hand_urdf(hand_id)
    cut = cut_urdf_to_hand(manifest_hand.urdf_path, manifest_hand.hand_root)
    cut_path = cut.write(Path(out_dir) / f"{hand_id}_hand_only.urdf")

    joint_names = movable_joint_names(cut.root)
    # Post-merge names (merged_body_name): a hand-only cut's leaf links are
    # sometimes a fixed-joint sensor/pad stub past the last actuated joint
    # (e.g. SHARPA's "*_fingertip" -> "*_elastomer" -> "*_DP"), which the USD
    # importer folds into that ancestor -- see merged_body_name's docstring.
    # dict.fromkeys dedupes (preserving order) in the unlikely case two leaf
    # names collapse onto the same surviving body.
    tips = tuple(dict.fromkeys(merged_body_name(cut.root, t) for t in leaf_link_names(cut.root)))
    limits = _joint_limits(cut.root, joint_names)
    adjacency = _self_adjacent_links(cut.root, joint_names)

    bodies, boxes, valid, scale = joint_link_boxes(cut_path, joint_names)

    if hand_id in ("sharpa", "sharpa_left_on_iiwa14"):
        from hand_sampler.robot_param_constants import HAND_ARMATURE, HAND_DAMPING, HAND_STIFFNESS

        stiffness = {j: HAND_STIFFNESS[j] for j in joint_names}
        damping = {j: HAND_DAMPING[j] for j in joint_names}
        armature = {j: HAND_ARMATURE[j] for j in joint_names}
    else:
        stiffness = {j: DEFAULT_HAND_STIFFNESS for j in joint_names}
        damping = {j: DEFAULT_HAND_DAMPING for j in joint_names}
        armature = {j: DEFAULT_HAND_ARMATURE for j in joint_names}

    spec = HandOnlySpec(
        name=name or f"{hand_id}_hand_only", hand_name=hand_id, urdf_path=str(cut_path),
        hand_root=cut.hand_root,
        hand_joint_names=joint_names, hand_joint_limits=limits,
        palm_body_name=cut.hand_root, fingertip_body_names=tips,
        hand_stiffness=stiffness, hand_damping=damping, hand_armature=armature,
        # Clamped into each joint's own limits: 0.0 for most hands, but some
        # commercial hands (e.g. allegro_right's thumb joint_12, limits
        # [0.263, 1.396]) do not have 0 in range at all, and Isaac Lab's
        # Articulation._validate_cfg() hard-errors on a default outside the
        # limits ("default positions out of the limits") rather than
        # clamping it itself.
        hand_default_joint_pos={
            j: min(max(0.0, lo), hi) for j, (lo, hi) in zip(joint_names, limits)
        },
        palm_center_offset=(0.0, 0.0, 0.0),
        adjacent_links=adjacency,
        joint_link_bodies=tuple(bodies),
        joint_link_boxes=tuple(tuple(map(tuple, b)) for b in boxes),
        joint_geometry_valid=tuple(bool(v) for v in valid), hand_scale=float(scale),
        notes=(f"hand-only cut of {manifest_hand.urdf_path} at hand_root="
               f"{manifest_hand.hand_root!r}; {len(cut.unresolved_meshes)} unresolved "
               f"mesh refs dropped"),
    )
    return spec, cut
