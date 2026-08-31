"""Scene construction, asset conversion, and runtime material setup."""

from __future__ import annotations

import math
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_from_angle_axis, quat_mul
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage

from hand_sampler.paths import resolve as resolve_repo_path

from .objects.generate_objects import generate_handle_head_urdfs


# ----------------------------------------------------------------------------
# Joint names / regexes / body names
# ----------------------------------------------------------------------------

# Joint names, regexes, PD-gain tables, the arm home pose, and the palm and
# fingertip body names all used to live here as module constants pinned to the
# left SHARPA hand. They are now fields on the selected RobotSpec
# (isaacsimenvs/robots/), so the scene follows the configured hand.
#
# HAND_JOINT_FRICTION was dropped rather than moved: it was defined and
# length-asserted but never read -- joint friction reaches PhysX from the URDF's
# <dynamics friction> via UrdfConverter.

_CONTACT_OFFSET = 0.002
_REST_OFFSET = 0.0

# group: "rb" (RigidBodyAPI) or "art" (ArticulationRootAPI).
# attr_name: USD attribute path. vtype_str: matched against pxr.Sdf.ValueTypeNames.
_PHYSICS_SPECS: dict[str, tuple[str, str, str]] = {
    "kinematic_enabled": ("rb", "physics:kinematicEnabled", "Bool"),
    "disable_gravity": ("rb", "physxRigidBody:disableGravity", "Bool"),
    "max_depenetration_velocity": ("rb", "physxRigidBody:maxDepenetrationVelocity", "Float"),
    "rb_solver_position_iterations": ("rb", "physxRigidBody:solverPositionIterationCount", "Int"),
    "rb_solver_velocity_iterations": ("rb", "physxRigidBody:solverVelocityIterationCount", "Int"),
    "articulation_enabled": ("art", "physics:articulationEnabled", "Bool"),
    "enabled_self_collisions": ("art", "physxArticulation:enabledSelfCollisions", "Bool"),
    "solver_position_iterations": ("art", "physxArticulation:solverPositionIterationCount", "Int"),
    "solver_velocity_iterations": ("art", "physxArticulation:solverVelocityIterationCount", "Int"),
}


def build_robot_articulation_usd_cfg(
    usd_path: str | list[str], spec, *, start_arm_higher: bool = False
) -> ArticulationCfg:
    """Build the robot articulation for ``spec`` from its baked USD(s).

    Actuator groups are keyed by exact joint name rather than a regex, so a hand
    whose joints do not share a common prefix still gets its gains applied.

    A LIST of USDs spawns one design per env, round-robin, exactly as the object
    pool does -- env i gets ``usds[i % len(usds)]``. All of them must share the
    joint names and count the actuator tables and ``spec`` describe, which is
    what ghosting guarantees for a generated population: every design is padded
    to the same 37-joint template, so an arbitrary number of morphologies costs
    one articulation view. Designs with different joint COUNTS cannot share a
    view at all -- PhysX reads ``shared_metatype.dof_count`` once per view.
    """
    # The sentinel means the robot prims were authored straight onto the stage
    # (one design per env), so there is nothing to spawn. This is what avoids
    # MultiUsdFileCfg's per-design composition at k = n.
    if usd_path == "__authored_in_place__":
        spawn = None
    else:
        many = not isinstance(usd_path, str)
        spawn = (MultiUsdFileCfg(usd_path=list(usd_path), random_choice=False)
                 if many else UsdFileCfg(usd_path=usd_path))
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=spec.base_pos,
            rot=spec.base_rot,
            joint_pos={
                **spec.arm_default_joint_pos_resolved(start_arm_higher=start_arm_higher),
                **spec.hand_default_joint_pos,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(spec.arm_joint_names),
                stiffness=dict(spec.arm_stiffness),
                damping=dict(spec.arm_damping),
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=list(spec.hand_joint_names),
                stiffness=dict(spec.hand_stiffness),
                damping=dict(spec.hand_damping),
                armature=dict(spec.hand_armature),
            ),
        },
    )


def build_rigid_object_cfg(prim_path: str, usd_paths: list[str]) -> RigidObjectCfg:
    """Spawn a RigidObject from one or more pre-baked USDs (round-robin)."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=MultiUsdFileCfg(usd_path=list(usd_paths), random_choice=False),
    )


def _log_scene_step(start_time: float, message: str) -> None:
    print(f"[scene_utils][+{time.perf_counter() - start_time:.2f}s] {message}", flush=True)


def _set_usd_attr(prim, name: str, value, value_type) -> None:
    # The URDF converter occasionally emits attributes with malformed type
    # names; in that case remove and recreate so the typed Set lands.
    attr = prim.GetAttribute(name)
    if attr and (not attr.GetTypeName() or not str(attr.GetTypeName())):
        prim.RemoveProperty(name)
        attr = None
    (attr or prim.CreateAttribute(name, value_type, False)).Set(value)


@dataclass(frozen=True)
class _UrdfSdfCollisionMarker:
    mesh_stem: str
    mesh_filename: str
    resolution: int | None = None
    margin: float | None = None
    narrow_band_thickness: float | None = None
    subgrid_resolution: int | None = None


def _usd_safe_identifier(name: str) -> str:
    """Mirror the conservative subset of USD identifier rules we need here."""
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = f"mesh_{safe}"
    return safe


def _resolve_urdf_mesh_path(urdf_path: Path, mesh_filename: str) -> Path:
    mesh_path = Path(mesh_filename)
    if mesh_path.is_absolute():
        return mesh_path
    return (urdf_path.parent / mesh_path).resolve()


def _prepare_urdf_for_isaacsim(asset_path: str, usd_work_dir: Path) -> str:
    """Return a URDF path whose mesh stems are valid USD prim identifiers.

    Isaac's URDF importer names USD prims from mesh stems.  Meshes such as
    ``6_hole_patch.obj`` therefore fail conversion because USD identifiers
    cannot start with a digit.  When needed, write a temporary URDF with safe
    mesh aliases while leaving the source asset untouched.
    """
    # Asset paths in the configs are repo-relative. Resolving them against
    # REPO_ROOT rather than the process CWD means eval/SLURM/test scripts can be
    # launched from anywhere, instead of silently failing to find meshes when
    # run outside the repo root.
    urdf_path = resolve_repo_path(asset_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    changed = False
    alias_dir = usd_work_dir / "_mesh_aliases" / urdf_path.stem
    alias_by_source: dict[Path, Path] = {}
    used_alias_names: set[str] = set()

    for mesh_tag in root.findall(".//mesh"):
        filename = mesh_tag.get("filename")
        if not filename:
            continue
        source_mesh = _resolve_urdf_mesh_path(urdf_path, filename)
        original_stem = source_mesh.stem
        safe_stem = _usd_safe_identifier(original_stem)
        if safe_stem == original_stem:
            continue

        changed = True
        alias = alias_by_source.get(source_mesh)
        if alias is None:
            alias_dir.mkdir(parents=True, exist_ok=True)
            alias_name = f"{safe_stem}{source_mesh.suffix}"
            if alias_name in used_alias_names:
                index = 1
                while f"{safe_stem}_{index}{source_mesh.suffix}" in used_alias_names:
                    index += 1
                alias_name = f"{safe_stem}_{index}{source_mesh.suffix}"
            used_alias_names.add(alias_name)
            alias = alias_dir / alias_name
            shutil.copy2(source_mesh, alias)
            alias_by_source[source_mesh] = alias

        mesh_tag.set("filename", str(alias))

    if not changed:
        return asset_path

    # The copied URDF lives in the converter work dir, so make every remaining
    # mesh path absolute to preserve source-relative references.
    for mesh_tag in root.findall(".//mesh"):
        filename = mesh_tag.get("filename")
        if filename:
            mesh_tag.set("filename", str(_resolve_urdf_mesh_path(urdf_path, filename)))

    out_dir = usd_work_dir / "_urdf_preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / urdf_path.name
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return str(out_path)


def _parse_optional_int(value: str | None) -> int | None:
    return None if value is None else int(value)


def _parse_optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _parse_urdf_sdf_collision_markers(asset_path: str) -> list[_UrdfSdfCollisionMarker]:
    urdf_path = Path(asset_path)
    root = ET.parse(urdf_path).getroot()
    markers: list[_UrdfSdfCollisionMarker] = []
    for collision in root.findall(".//collision"):
        sdf_tag = collision.find("sdf")
        if sdf_tag is None:
            continue
        mesh_tag = collision.find("geometry/mesh")
        if mesh_tag is None or not mesh_tag.get("filename"):
            continue
        mesh_filename = str(mesh_tag.get("filename"))
        mesh_path = _resolve_urdf_mesh_path(urdf_path, mesh_filename)
        markers.append(
            _UrdfSdfCollisionMarker(
                mesh_stem=_usd_safe_identifier(mesh_path.stem),
                mesh_filename=mesh_filename,
                resolution=_parse_optional_int(sdf_tag.get("resolution")),
                margin=_parse_optional_float(sdf_tag.get("margin")),
                narrow_band_thickness=_parse_optional_float(
                    sdf_tag.get("narrow_band_thickness")
                    or sdf_tag.get("narrowBandThickness")
                ),
                subgrid_resolution=_parse_optional_int(
                    sdf_tag.get("subgrid_resolution")
                    or sdf_tag.get("subgridResolution")
                ),
            )
        )
    return markers


def _apply_urdf_sdf_collision_markers(
    usd_path: str,
    source_asset_path: str,
    markers: list[_UrdfSdfCollisionMarker],
) -> None:
    if not markers:
        return

    from pxr import Usd, UsdPhysics

    from isaaclab.sim.schemas import SDFMeshPropertiesCfg, define_mesh_collision_properties

    raw_usd_path = Path(usd_path)
    physics_usd_path = raw_usd_path.parent / "configuration" / f"{raw_usd_path.stem}_physics.usd"
    edit_usd_path = physics_usd_path if physics_usd_path.exists() else raw_usd_path

    stage = Usd.Stage.Open(str(edit_usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open USD while applying URDF SDF markers: {edit_usd_path}")
    stage.Load()

    marker_by_stem = {marker.mesh_stem: marker for marker in markers}
    matched: dict[str, int] = {marker.mesh_stem: 0 for marker in markers}

    fallback_matches = []
    collider_matches = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            continue
        path = prim.GetPath().pathString
        path_parts = [part for part in path.split("/") if part]
        marker = next((marker_by_stem[part] for part in path_parts if part in marker_by_stem), None)
        if marker is None:
            continue
        if path.startswith("/colliders/"):
            collider_matches.append((prim, marker))
        else:
            fallback_matches.append((prim, marker))

    for prim, marker in collider_matches or fallback_matches:
        define_mesh_collision_properties(
            str(prim.GetPath()),
            SDFMeshPropertiesCfg(
                sdf_margin=marker.margin,
                sdf_narrow_band_thickness=marker.narrow_band_thickness,
                sdf_resolution=marker.resolution,
                sdf_subgrid_resolution=marker.subgrid_resolution,
            ),
            stage=stage,
        )
        matched[marker.mesh_stem] += 1

    stage.GetRootLayer().Save()

    matched_count = sum(matched.values())
    missing = [stem for stem, count in matched.items() if count == 0]
    if missing:
        print(
            f"[scene_utils] warning: URDF SDF markers in {source_asset_path!r} did not match "
            f"USD collision prims for mesh stems {missing}",
            flush=True,
        )
    if matched_count:
        details = ", ".join(f"{stem}:{count}" for stem, count in matched.items() if count)
        print(
            f"[scene_utils] applied URDF SDF collision markers to {matched_count} prims "
            f"in {edit_usd_path.name} ({details})",
            flush=True,
        )


def _apply_self_collision_filters(
    usd_path: str,
    adjacency: dict[str, list[str]],
    *,
    strict: bool = True,
) -> None:
    """Author USD ``FilteredPairsAPI`` on the robot's articulation links so the
    ``adjacency`` link pairs do NOT self-collide — mirroring Isaac Gym, which
    enables all self-collisions then masks adjacent links.

    Only effective when the articulation has self-collision enabled
    (``enabled_self_collisions=True`` + URDF import ``self_collision=True``).
    Links merged away by ``merge_fixed_joints`` have no rigid-body prim and are
    skipped (a merged link shares its parent's body and cannot self-collide
    anyway) — so a non-empty ``missing`` set is expected, not an error.

    ``strict`` raises when *zero* pairs were filtered. That is the swap-a-new-hand
    failure mode: an adjacency map whose link names match nothing leaves the robot
    with self-collisions fully enabled and no masking, which explodes at reset.
    Before, this only printed a warning.
    """
    from pxr import Usd, UsdPhysics

    raw_usd_path = Path(usd_path)
    physics_usd_path = raw_usd_path.parent / "configuration" / f"{raw_usd_path.stem}_physics.usd"
    edit_usd_path = physics_usd_path if physics_usd_path.exists() else raw_usd_path

    stage = Usd.Stage.Open(str(edit_usd_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open USD while applying self-collision filters: {edit_usd_path}")
    stage.Load()

    body_by_name: dict[str, Usd.Prim] = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_by_name[prim.GetName()] = prim

    pairs_filtered = 0
    missing: set[str] = set()
    for link, neighbors in adjacency.items():
        a = body_by_name.get(link)
        if a is None:
            missing.add(link)
            continue
        rel = UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel()
        existing = set(rel.GetTargets())
        for nb in neighbors:
            b = body_by_name.get(nb)
            if b is None:
                missing.add(nb)
                continue
            if b.GetPath() not in existing:
                rel.AddTarget(b.GetPath())
                existing.add(b.GetPath())
                pairs_filtered += 1

    stage.GetRootLayer().Save()

    print(
        f"[scene_utils] self-collision: filtered {pairs_filtered} adjacent link pairs "
        f"across {len(body_by_name)} robot bodies in {edit_usd_path.name}"
        + (f"; skipped {len(missing)} merged/absent links" if missing else ""),
        flush=True,
    )

    if strict and pairs_filtered == 0:
        raise RuntimeError(
            f"Self-collision filtering matched 0 pairs in {edit_usd_path.name}. The adjacency "
            f"map's link names do not match this robot's USD bodies, so self-collisions are "
            f"enabled with no masking. Adjacency links: {sorted(adjacency)[:5]}...; "
            f"USD bodies: {sorted(body_by_name)[:5]}... "
            f"Note the map must use POST-merge_fixed_joints body names."
        )


def _generate_scaled_table_urdfs(
    base_urdf_path: str,
    num_variants: int,
    scale_range_x: tuple[float, float],
    scale_range_y: tuple[float, float],
    out_dir: Path,
    seed: int = 0,
) -> tuple[list[str], list[tuple[float, float]]]:
    """Write `num_variants` scaled copies of a single-box table URDF.

    Each variant samples (sx, sy) independently from the configured ranges
    (Z scale held at 1.0 so the table surface height matches what the policy
    was trained on). The base URDF must have a single `<box size="X Y Z"/>`
    in both the `<visual>` and `<collision>` blocks (matches the bundled
    `assets/urdf/table_narrow.urdf`).

    Returns the list of written URDF paths, in deterministic order.
    """
    import re
    import numpy as np

    base_text = resolve_repo_path(base_urdf_path).read_text()
    match = re.search(r'<box\s+size="([\d.\-+eE\s]+)"\s*/>', base_text)
    if match is None:
        raise ValueError(
            f"table URDF {base_urdf_path!r} has no <box size=\"...\"/> element; "
            "scaling helper only supports the simple single-box table."
        )
    base_dims = tuple(float(v) for v in match.group(1).split())
    if len(base_dims) != 3:
        raise ValueError(
            f"expected 3-element <box size>, got {base_dims!r} from {base_urdf_path}"
        )

    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    scales: list[tuple[float, float]] = []
    for i in range(int(num_variants)):
        sx = float(rng.uniform(*scale_range_x))
        sy = float(rng.uniform(*scale_range_y))
        new_size = f"{base_dims[0] * sx:.6f} {base_dims[1] * sy:.6f} {base_dims[2]:.6f}"
        new_text = re.sub(
            r'<box\s+size="[\d.\-+eE\s]+"\s*/>',
            f'<box size="{new_size}"/>',
            base_text,
        )
        path = out_dir / f"table_variant_{i:03d}.urdf"
        path.write_text(new_text)
        paths.append(str(path))
        scales.append((sx, sy))
    return paths, scales


def _convert_urdf_to_usd(
    asset_path: str,
    usd_work_dir: Path,
    *,
    fix_base: bool,
    self_collision: bool | None = None,
    replace_cylinders_with_capsules: bool = False,
    joint_drive=None,
    make_instanceable: bool = False,
) -> str:
    # make_instanceable puts geometry in a shared prototype that every env
    # references, instead of deep-copying it per env -- the standard fix for slow
    # cloning at large env counts. It defaults OFF here because the robot path
    # authors FilteredPairsAPI per link afterwards (_apply_self_collision_filters)
    # and instance proxies cannot be edited in place; turning it on without
    # checking that the filters still land would silently restore the adjacent-
    # link self-collisions the filters exist to remove. Measure before adopting.
    converter_asset_path = _prepare_urdf_for_isaacsim(asset_path, usd_work_dir)
    cfg_kwargs = dict(
        asset_path=converter_asset_path,
        usd_dir=str(usd_work_dir / Path(asset_path).stem),
        force_usd_conversion=True,
        fix_base=fix_base,
        merge_fixed_joints=True,
        make_instanceable=make_instanceable,
        replace_cylinders_with_capsules=replace_cylinders_with_capsules,
        joint_drive=joint_drive,
    )
    if self_collision is not None:
        cfg_kwargs["self_collision"] = self_collision
    usd_path = UrdfConverter(UrdfConverterCfg(**cfg_kwargs)).usd_path
    _apply_urdf_sdf_collision_markers(
        usd_path,
        converter_asset_path,
        _parse_urdf_sdf_collision_markers(converter_asset_path),
    )
    return usd_path



def _robot_joint_drive_cfg():
    # DriveAPI prims must exist for ImplicitActuator runtime gains to land.
    return UrdfConverterCfg.JointDriveCfg(
        drive_type="force", target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
    )


def _bake_usd(
    raw_usd_path: str,
    bake_root: Path,
    baked_subdir: str,
    *,
    props: dict | None = None,
    apply_physx_articulation: bool = False,
    collision_enabled: bool | None = None,
) -> str:
    """Copy raw USD into bake_root/baked_subdir and pre-author physics props.

    ``props`` keys come from ``_PHYSICS_SPECS``; ``None`` values are skipped,
    and keys whose group doesn't match a prim's APIs are skipped per-prim.
    """
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    vtypes = {
        "Bool": Sdf.ValueTypeNames.Bool,
        "Float": Sdf.ValueTypeNames.Float,
        "Int": Sdf.ValueTypeNames.Int,
    }
    props = props or {}

    raw = Path(raw_usd_path)
    baked_usd_path = bake_root / baked_subdir / raw.parent.name / raw.name
    baked_usd_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw, baked_usd_path)
    for child in raw.parent.iterdir():
        if child.name.startswith(".") or child.name in (raw.name, "config.yaml"):
            continue
        dst = baked_usd_path.parent / child.name
        if child.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(child, dst)
        elif child.is_file():
            shutil.copy2(child, dst)

    stage = Usd.Stage.Open(str(baked_usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open baked USD: {baked_usd_path}")
    root = stage.GetDefaultPrim()
    if not (root and root.IsValid()):
        root = next((p for p in stage.GetPseudoRoot().GetChildren() if p.IsValid()), None)
    if root is None:
        raise RuntimeError(f"No root prim in USD: {baked_usd_path}")

    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)

    for prim in Usd.PrimRange(root):
        is_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        is_art = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        if is_rb:
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        if is_art and apply_physx_articulation:
            PhysxSchema.PhysxArticulationAPI.Apply(prim)
        for key, val in props.items():
            if val is None:
                continue
            group, attr_name, vtype_str = _PHYSICS_SPECS[key]
            if group == "rb" and not is_rb:
                continue
            if group == "art" and not is_art:
                continue
            _set_usd_attr(prim, attr_name, val, vtypes[vtype_str])
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            px = PhysxSchema.PhysxCollisionAPI(prim) or PhysxSchema.PhysxCollisionAPI.Apply(prim)
            px.CreateContactOffsetAttr().Set(_CONTACT_OFFSET)
            px.CreateRestOffsetAttr().Set(_REST_OFFSET)
            if collision_enabled is not None:
                ce = UsdPhysics.CollisionAPI(prim)
                (ce.GetCollisionEnabledAttr() or ce.CreateCollisionEnabledAttr()).Set(
                    collision_enabled
                )

    stage.GetRootLayer().Save()
    return str(baked_usd_path)


# ----------------------------------------------------------------------------
# Runtime material setup (post-launch, via PhysX views)
# ----------------------------------------------------------------------------

def shape_layouts_from_record(link_names, recorded: dict, arm_counts: dict):
    """Per-design [(link, start, end)] from the authoring record + arm counts.

    ONE implementation, used by the friction pass and by
    tests/test_shape_layout_record.py. An earlier version of that test carried
    its own copy of this arithmetic, inherited the same off-by-one the fast path
    had, and so agreed with the code instead of checking it.
    """
    out = {}
    for design, per_link in recorded.items():
        start, layout = 0, []
        for name in link_names:
            # SUM both sources. A hand link has only a record, an arm link only
            # a measurement, and the merged palm (iiwa14_link_7) has BOTH: the
            # arm's own collision mesh plus the palm box authored onto it.
            n = arm_counts.get(name, 0) + per_link.get(name, 0)
            layout.append((name, start, start + n))
            start += n
        out[design] = (layout, start)
    return out


def arm_counts_from(measured_layout, ref_record: dict) -> dict:
    """The ARM's own per-link shape counts, isolated from a measured env.

    A measured layout comes from a composed robot, so iiwa14_link_7 already
    includes the authored palm box. Subtracting the record for the design that
    was measured leaves what the referenced arm USD contributes on its own;
    adding the record back per design is then exact rather than double-counted.
    """
    return {nm: (e - s) - ref_record.get(nm, 0)
            for nm, s, e in measured_layout if nm.startswith("iiwa14_link")}


def apply_physx_material_properties(env) -> None:
    """Set contact materials through PhysX tensor views.

    Follows Isaac Lab's large-scale randomization path: avoid post-spawn USD
    relationship authoring and per-clone material prims. Must run after
    ``DirectRLEnv`` starts the simulator and ``root_physx_view`` exists.
    """
    assets_cfg = env.cfg.assets
    if not assets_cfg.modify_asset_frictions:
        return

    t0 = time.perf_counter()
    default = torch.tensor(
        [float(assets_cfg.robot_friction), float(assets_cfg.robot_friction), 0.0],
        dtype=torch.float32, device="cpu",
    )
    fingertip = torch.tensor(
        [float(assets_cfg.finger_tip_friction), float(assets_cfg.finger_tip_friction), 0.0],
        dtype=torch.float32, device="cpu",
    )
    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device="cpu")

    dr = env.cfg.domain_randomization
    n_buckets = int(dr.friction_n_buckets)
    ft_lo, ft_hi = float(dr.fingertip_friction_scale_range[0]), float(dr.fingertip_friction_scale_range[1])
    obj_lo, obj_hi = float(dr.object_friction_scale_range[0]), float(dr.object_friction_scale_range[1])
    ft_active = (ft_lo, ft_hi) != (1.0, 1.0)
    obj_active = (obj_lo, obj_hi) != (1.0, 1.0)

    robot_view = env.robot.root_physx_view
    robot_materials = robot_view.get_material_properties()
    robot_materials[:] = default

    fingertip_link_names = set(env.robot_spec.fingertip_body_names)

    # The shape layout is derived PER DESIGN, not once from env 0.
    #
    # A ghosted finger keeps its links but carries no collision geometry, so two
    # designs in one population have different shape counts and different shape
    # ORDERING within the flattened per-env shape array. Deriving the fingertip
    # slice from env 0 and applying it to every env would paint fingertip
    # friction onto whatever shape happened to occupy that index in another
    # design -- silently, since the values are all plausible frictions.
    #
    # Grouping by design keeps this O(k x links) rather than O(n x links). With
    # a single robot there is one group represented by env 0, which is exactly
    # the previous behaviour on exactly the same values.
    pop_specs = getattr(env, "_robot_population_specs", None)
    if pop_specs is None:
        groups = {0: torch.arange(env.num_envs, dtype=torch.int64)}
        tips_of = {0: fingertip_link_names}
    else:
        design_idx = env._robot_design_index_per_env.detach().cpu()
        groups = {int(d): (design_idx == int(d)).nonzero(as_tuple=True)[0]
                  for d in design_idx.unique()}
        tips_of = {d: set(pop_specs[d].fingertip_body_names) for d in groups}

    # (N, max_shapes): which shapes are fingertips, for THIS env's design.
    ft_shape_mask = torch.zeros(
        (env.num_envs, robot_view.max_shapes), dtype=torch.bool, device="cpu")

    def _measure_layout(rep_env: int):
        """(link_name, shape_start, shape_end) per link, and the total.

        One create_rigid_body_view per link, which is the expensive part: at
        ~1.2 ms a call it is the whole cost of this function.
        """
        out, shape_start = [], 0
        for link_name, link_path in zip(robot_view.shared_metatype.link_names,
                                        robot_view.link_paths[rep_env]):
            link_view = env.robot._physics_sim_view.create_rigid_body_view(link_path)
            shape_end = shape_start + link_view.max_shapes
            out.append((link_name, shape_start, shape_end))
            shape_start = shape_end
        return out, shape_start

    # Per DESIGN, measured from PhysX. Correct, and slow: 24,576 designs x 37
    # links is 909,312 create_rigid_body_view calls, ~71 min at 24,576 envs.
    #
    # Two attempts to make this faster have been reverted from here, and both
    # are worth remembering:
    #
    #   1. Cache the layout under a GUESSED signature (per-slot active mask plus
    #      self-collision adjacency). Fast -- 171 s -> 6.9 s at 2,048 -- and
    #      WRONG: segment lengths also decide which links carry geometry, so two
    #      designs with the same active mask can have different layouts. Design
    #      5120 of an 8,192 population took another design's layout and its
    #      fingertip shapes landed at the wrong indices. Only the guard below
    #      caught it; a 2,048-design test passed it clean.
    #
    #   2. Count the shapes in USD instead of inferring them. Correct, but
    #      Usd.PrimRange with TraverseInstanceProxies per link is no cheaper
    #      than the PhysX call it replaced -- measured SLOWER at 8,192 designs
    #      than the code here.
    #
    # The real fix is to record shape counts while AUTHORING the designs, where
    # they are already known, rather than rediscovering them per link afterwards.
    # Until that exists, this stays: being right is not negotiable, and this is
    # a one-time setup cost.
    # Progress, because this loop runs for over an hour at 24,576 designs and
    # used to print nothing at all: a run sitting here was indistinguishable
    # from a run that had hung, which is exactly how two bad "fixes" for it got
    # diagnosed by stopwatch instead of by evidence.
    _n_designs = len(groups)
    _t_layout = time.perf_counter()

    # FAST PATH: use the collider map recorded while the designs were authored.
    #
    # The loop below is correct and costs ~96 min at 24,576 designs. It is slow
    # for one reason: it asks PhysX, per link per design, a question the
    # authoring already answered. When the robots were authored in-process, they
    # told us exactly which links got colliders, so the only thing left to
    # measure is the ARM -- identical across designs, referenced from one USD --
    # which is a single _measure_layout call.
    recorded = getattr(env, "_robot_collider_links", None)
    fast_layouts = None
    if recorded and len(recorded) >= len(groups):
        # Measure one env, then SUBTRACT what the authoring contributed to it.
        #
        # _measure_layout reads a real composed robot, so its iiwa14_link_7
        # already includes the palm box this pipeline authored onto that link.
        # Adding the recorded palm on top counted it twice and pushed the total
        # one over the view's max_shapes. What is wanted is the ARM'S OWN
        # contribution, which is the measurement minus the record for the very
        # design that was measured.
        _ref_design = next(iter(groups))
        arm_layout, _ = _measure_layout(int(groups[_ref_design][0]))
        arm_counts = arm_counts_from(arm_layout, recorded[_ref_design])
        fast_layouts = shape_layouts_from_record(
            list(robot_view.shared_metatype.link_names),
            {d: recorded[d] for d in groups}, arm_counts)
        _log_scene_step(_t_layout,
                        f"shape layouts for {len(groups)} designs from the "
                        f"authoring record")

    for _i, (design, group_env_ids) in enumerate(groups.items()):
        if _n_designs > 1000 and _i and _i % 2000 == 0:
            _el = time.perf_counter() - _t_layout
            print(f"[scene]   shape layouts {_i}/{_n_designs} "
                  f"({_el:.0f}s elapsed, ~{_el / _i * (_n_designs - _i):.0f}s left)",
                  flush=True)
        layout, shape_start = (fast_layouts[design] if fast_layouts is not None
                               else _measure_layout(int(group_env_ids[0])))
        for link_name, s0, s1 in layout:
            if link_name in tips_of[design]:
                robot_materials[group_env_ids, s0:s1] = fingertip
                ft_shape_mask[group_env_ids, s0:s1] = True
        # Designs with ghosted fingers legitimately carry FEWER shapes than the
        # view's maximum; only overrunning it is a bug.
        if shape_start > robot_view.max_shapes:
            raise RuntimeError(
                f"Robot shape count mismatch while assigning materials for "
                f"design {design}: computed {shape_start}, view reports "
                f"{robot_view.max_shapes}."
            )
        if pop_specs is None and shape_start != robot_view.max_shapes:
            raise RuntimeError(
                f"Robot shape count mismatch while assigning materials: "
                f"computed {shape_start}, view reports {robot_view.max_shapes}."
            )

    # VERIFY the fast path against PhysX on a sample. A wrong layout paints
    # fingertip friction onto the wrong shapes with every value still plausible,
    # which is exactly how an earlier version of this optimisation shipped a
    # silent bug. tests/test_shape_layout_record.py checks a WHOLE population;
    # this is the belt-and-braces check that runs in every real scene build.
    if fast_layouts is not None:
        for design in sorted(groups)[::max(1, len(groups) // 8)][:8]:
            physx, _ = _measure_layout(int(groups[design][0]))
            rec = fast_layouts[design][0]
            if physx != rec:
                # Show the first ENTRIES THAT DIFFER, not the first entries.
                # Printing [:6] of each showed six identical arm links and hid
                # the actual disagreement further down the list.
                diff = [(a, b) for a, b in zip(physx, rec) if a != b]
                raise RuntimeError(
                    f"authoring-recorded shape layout disagrees with PhysX for "
                    f"design {design}: {len(diff)} link(s) differ\n"
                    + "\n".join(f"  {a[0]}: physx={a[1:]} recorded={b[1:]}"
                                 for a, b in diff[:6]))

    # A name mismatch here is silent and consequential: every fingertip would
    # quietly fall back to the robot's base friction, changing grasp behavior
    # with no error.
    unmatched = [d for d in groups if not bool(ft_shape_mask[groups[d]].any())]
    # Two very different things produce an unmatched design, and only one is a
    # bug:
    #
    #   NAME MISMATCH -- the spec's fingertip_body_names are not links of this
    #   robot at all. Silent and catastrophic: every fingertip falls back to
    #   base friction and grasping changes with no error. Still fatal.
    #
    #   DEGENERATE FINGERTIP -- the names ARE links, but the design's distal
    #   phalanx is shorter than its own diameter, so has_collision_geometry
    #   deliberately gives it no capsule. That is a legitimate (if useless) hand
    #   the sampler can produce: exactly 1 design in 24,576 of population seed 2,
    #   index 5120, whose distal is 16.4 mm long at 12 mm radius. Its fingertips
    #   cannot touch anything, so there is no fingertip friction to assign and
    #   nothing to fix. Failing the whole run for it blocks every population of
    #   more than 5,120 designs.
    known_links = set(robot_view.shared_metatype.link_names)
    misnamed = [d for d in unmatched if not (set(tips_of[d]) & known_links)]
    if misnamed:
        who = (env.robot_spec.name if pop_specs is None
               else f"designs {sorted(misnamed)[:5]}")
        raise RuntimeError(
            f"{who}: fingertip_body_names={sorted(tips_of[misnamed[0]])} are not "
            f"links of this robot. Robot links: "
            f"{sorted(robot_view.shared_metatype.link_names)}"
        )
    if unmatched:
        print(f"[scene] {len(unmatched)} design(s) have no fingertip collision "
              f"geometry and get no fingertip friction "
              f"(first: {sorted(unmatched)[:5]}); their distal phalanges are "
              f"shorter than their own diameter", flush=True)

    # Per-env bucketed fingertip friction (init-only). Quantizing to
    # `n_buckets` distinct values caps the PhysX material count regardless
    # of n_envs.
    if ft_active:
        ft_base = float(assets_cfg.finger_tip_friction)
        bucket_vals = torch.linspace(ft_lo, ft_hi, n_buckets) * ft_base  # (B,)
        bucket_idx = torch.randint(0, n_buckets, (env.num_envs,))
        per_env_ft = bucket_vals[bucket_idx]  # (N_envs,)
        # Applied through the per-env fingertip mask so each design's own
        # fingertip shapes are the ones scaled. A shared index list would be the
        # union across designs, which for a heterogeneous population scales
        # shapes that are not fingertips in most envs.
        scaled = per_env_ft.unsqueeze(-1).expand(-1, robot_view.max_shapes)
        for channel in (0, 1):
            robot_materials[..., channel] = torch.where(
                ft_shape_mask, scaled, robot_materials[..., channel])

    robot_view.set_material_properties(robot_materials, env_ids)

    for name in ("table", "object", "goal_viz", "hole"):
        if not hasattr(env, name):
            continue
        view = getattr(env, name).root_physx_view
        materials = view.get_material_properties()
        materials[:] = default
        if name == "object" and obj_active:
            obj_base = float(assets_cfg.object_friction)
            bucket_vals = torch.linspace(obj_lo, obj_hi, n_buckets) * obj_base
            bucket_idx = torch.randint(0, n_buckets, (env.num_envs,))
            per_env_obj = bucket_vals[bucket_idx]  # (N_envs,)
            materials[:, :, 0] = per_env_obj.unsqueeze(-1)
            materials[:, :, 1] = per_env_obj.unsqueeze(-1)
        # Column 2 is restitution. Training uses 0.0 (fully inelastic), so the
        # default leaves this exactly as before; the object-physics eval axis
        # raises it to probe bouncier contacts.
        if name == "object":
            materials[:, :, 2] = float(assets_cfg.object_restitution)
        view.set_material_properties(materials, env_ids)

    _log_scene_step(t0, "applied PhysX material properties")


# ----------------------------------------------------------------------------
# Scene assembly
# ----------------------------------------------------------------------------

def _materialize_env_prims(env) -> None:
    stage = get_current_stage()
    for env_path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(env_path).IsValid():
            stage.DefinePrim(env_path, "Xform")


def _env_id_of(prim_path: str) -> int:
    """Numeric env id from a `/World/envs/env_<i>` path or any child of one."""
    for token in prim_path.split("/"):
        if token.startswith("env_"):
            return int(token.removeprefix("env_"))
    raise ValueError(f"no env_<i> component in {prim_path!r}")


def _author_objects_into_envs(env, object_params, n_pool: int,
                              which=("object", "goalviz")) -> None:
    """Author Object and GoalViz prims directly into every env.

    Env i takes pool entry ``i % n_pool`` -- the same assignment MultiUsdFileCfg
    makes, so ``_build_object_scale_tensor``'s per-env scale lookup stays correct
    and the policy sees the object it expects.

    The goal marker is the same geometry with collision omitted, matching how the
    converted path bakes goalviz with ``collision_enabled=False``.
    """
    from pxr import Sdf

    from isaacsimenvs.authoring.author_usd import define
    from isaacsimenvs.pose_reach.utils.objects.author_objects import (
        author_handle_head,
        author_physics_material,
    )

    stage = get_current_stage()
    layer = stage.GetRootLayer()
    t0 = time.perf_counter()
    with Sdf.ChangeBlock():
        # One shared material; the env overwrites its values per env through
        # root_physx_view.set_material_properties, so per-asset materials would
        # buy nothing and cost a prim each.
        mat_path = author_physics_material(
            layer, "/World/PhysicsMaterials/object",
            static_friction=float(env.cfg.assets.object_friction),
            dynamic_friction=float(env.cfg.assets.object_friction),
            restitution=float(env.cfg.assets.object_restitution))
        # Assign pool entries by NUMERIC env id, and record the map so that
        # _build_object_scale_tensor can consume it rather than re-deriving it.
        #
        # Re-deriving is what went wrong before. That function reads
        # find_matching_prim_paths("/World/envs/env_.*/Object") and takes
        # asset_index = position % pool, and this loop used to walk
        # sorted(env_prim_paths) on the belief that the two orders agree.
        # They do not: find_matching_prim_paths returns NUMERIC order
        # (env_0, env_1, env_2, ...) while sorted() returns lexicographic
        # (env_0, env_1, env_10, env_100, ...). Measured on 512 envs, the two
        # disagreed in 510 of them -- so nearly every env held an object whose
        # geometry did not match the object_scales in its own observation, and
        # the policy scored 3.00 goals against 5.07 for the converted path.
        #
        # Note the assignment is deliberately identical to MultiUsdFileCfg's,
        # which walks that same numeric list: env i takes pool entry i % n_pool.
        # Recording the map means a future change to either ordering cannot
        # silently desynchronise the two again.
        env._authored_asset_index = {}
        for env_path in sorted(env.scene.env_prim_paths, key=_env_id_of):
            source_idx = _env_id_of(env_path)
            env._authored_asset_index[source_idx] = source_idx % n_pool
            handle_scale, head_scale, handle_density, head_density = \
                object_params[source_idx % n_pool]
            # GoalViz: no collider AND no motion. The converted path bakes it
            # kinematic with gravity disabled; both are required.
            for name, collision, kinematic in (("Object", True, False),
                                               ("GoalViz", False, True)):
                if name.lower() not in which:
                    continue
                # The parent Xform the converter's USD root maps onto.
                define(layer, f"{env_path}/{name}", "Xform")
                author_handle_head(
                    layer, f"{env_path}/{name}",
                    handle_scale, head_scale, handle_density, head_density,
                    body_at_root=False, collision=collision,
                    material_path=mat_path, kinematic=kinematic)
    # The physics material is bound by _shape_prim, on the collision shapes,
    # inside the ChangeBlock above.
    #
    # It used to be bound here instead, by isaaclab's bind_physics_material,
    # twice per env and outside the ChangeBlock. That helper is decorated with
    # apply_nested so every call traverses the subtree of a live stage, and the
    # authored 24k population makes that stage ~5.8M prims: 49,152 traversals
    # that grow with the robots, for a binding whose target prims we authored
    # ourselves and therefore already know. Without a bound material the env's
    # friction pass fails loudly with "Failed to get rigid body material
    # properties from backend", so a regression here is not silent.

    _log_scene_step(
        t0, f"authored {env.num_envs} Object + GoalViz prims from a "
            f"{n_pool}-entry pool")


def _resolve_robot_population(assets_cfg) -> list | None:
    """Specs for a cached hand population, or None for the single-robot env."""
    seed = getattr(assets_cfg, "robot_population_seed", None)
    path = getattr(assets_cfg, "robot_population_path", None)
    if seed is None and not path:
        return None
    from hand_sampler.population import load_population_any
    from hand_sampler.synth_spec import synth_spec

    hands = load_population_any(seed, path)
    # The AUTHORED path builds every robot's physics from HandParams and never
    # opens a design URDF -- only the converter branch reads design_spec.urdf_path
    # (see _build_robot_usds below). So writing one URDF per design costs 13 min
    # of every scene build and 24,576 files into a directory already holding 172k,
    # for artifacts nothing reads. It is also a race: five concurrent seed-jobs on
    # one population all find the files absent and all write the same paths.
    authored = bool(getattr(assets_cfg, "author_robot_usds", False))
    count = int(getattr(assets_cfg, "robot_population_count", 0) or 0)
    if count:
        if count > len(hands):
            raise ValueError(
                f"robot_population_count={count} exceeds the cached population "
                f"for seed {seed} ({len(hands)} hands). Build a larger one with "
                f"hand_sampler.population.build_population.")
        hands = hands[:count]
    specs = [synth_spec(h, ensure_urdf=not authored) for h in hands]

    # Every design must present the same joint vector: one articulation view has
    # one dof_count, and the actuator tables are keyed by name. Ghosting is what
    # makes this hold; check it rather than trust it, because a mismatch here
    # surfaces as an opaque PhysX view error much later.
    ref = specs[0]
    for s in specs[1:]:
        if s.joint_names_canonical != ref.joint_names_canonical:
            raise RuntimeError(
                f"design {s.name} does not share the joint template with "
                f"{ref.name}: a population must be padded to one joint set "
                f"before it can share an articulation view")
        if s.num_fingertip_slots != ref.num_fingertip_slots:
            raise RuntimeError(
                f"design {s.name} has {s.num_fingertip_slots} fingertip slots, "
                f"{ref.name} has {ref.num_fingertip_slots}: the observation "
                f"layout must be identical across a population")
    return specs


def _build_robot_design_tensor(env, num_designs: int) -> None:
    """Record which design each env holds, and CHECK it against the sim.

    MultiUsdFileCfg assigns proto i to the i-th Robot prim that
    ``find_matching_prim_paths`` returns, so env i holds design i % k. That is an
    assumption about someone else's iteration order, and the identical
    assumption about the object pool silently gave 510 of 512 envs the wrong
    asset -- costing a 5.07 -> 3.00 goals/episode regression that every
    asset-level comparison passed (docs/multi_embodiment.md §4).

    So this does not merely record the map; it verifies it against a quantity
    that actually differs per design and is read back out of PhysX. Joint limits
    are that quantity: the sampler draws each design's abduction range
    independently, so the limit vector is effectively a fingerprint.
    """
    num_envs = env.num_envs
    robot_prim_paths = find_matching_prim_paths("/World/envs/env_.*/Robot")
    if len(robot_prim_paths) != num_envs:
        raise RuntimeError(
            f"Expected {num_envs} Robot prims, got {len(robot_prim_paths)}.")
    observed = [_env_id_of(p) for p in robot_prim_paths]
    if observed != sorted(observed):
        raise RuntimeError(
            "find_matching_prim_paths no longer returns Robot prims in numeric "
            f"env order (first 12: {observed[:12]}); the design assignment "
            "below assumes it does.")

    env._robot_design_index_per_env = torch.tensor(
        [i % num_designs for i in range(num_envs)],
        device=env.device, dtype=torch.long,
    )


def _verify_robot_design_assignment(env, specs) -> None:
    """Confirm env i really holds design i % k, using per-design joint limits.

    Runs after the sim exists, so it reads what PhysX loaded rather than what
    the config asked for. Cheap, and it closes the exact hole that produced the
    object-assignment regression.
    """
    import numpy as np

    limits = env.robot.data.joint_pos_limits.detach().cpu().numpy()  # (N, J, 2)
    idx = env._robot_design_index_per_env.detach().cpu().numpy()
    # Designs whose limit vectors are identical cannot distinguish an assignment
    # error, so report how much signal the check actually has.
    sigs = {}
    for e in range(env.num_envs):
        sigs.setdefault(idx[e], []).append(limits[e])
    distinct = {d: np.round(v[0], 9).tobytes() for d, v in sigs.items()}
    n_unique = len(set(distinct.values()))

    bad = []
    for d, mats in sigs.items():
        ref = mats[0]
        for m in mats[1:]:
            if not np.allclose(ref, m, atol=1e-9):
                bad.append(d)
                break
    if bad:
        raise RuntimeError(
            f"robot design assignment is inconsistent: envs sharing design "
            f"{sorted(bad)[:5]} do not share joint limits. env i is supposed to "
            f"hold design i % {len(specs)}.")
    print(f"[scene] robot design assignment verified: {len(sigs)} designs over "
          f"{env.num_envs} envs, {n_unique} distinguishable by joint limits")


def _build_object_scale_tensor(env, object_scales_normalized, num_object_usds: int) -> None:
    num_envs = env.num_envs
    object_prim_paths = find_matching_prim_paths("/World/envs/env_.*/Object")
    if len(object_prim_paths) != num_envs:
        raise RuntimeError(
            f"Expected {num_envs} Object prims after MultiUsdFileCfg spawn, "
            f"got {len(object_prim_paths)}. Cloner-drop bug may have returned."
        )

    env._object_scale_per_env = torch.zeros(num_envs, 3, device=env.device, dtype=torch.float32)
    env._object_asset_index_per_env = torch.zeros(num_envs, device=env.device, dtype=torch.long)
    # When the objects were authored, that pass already decided which pool entry
    # each env holds; take its map rather than re-deriving one here. Two
    # independent derivations of the same mapping is exactly how the paths
    # desynchronised before -- see the note in _author_objects_into_envs.
    authored_map = getattr(env, "_authored_asset_index", None)
    if authored_map is None:
        # The converted path pairs env i with the i-th spawned proto by relying
        # on MultiUsdFileCfg walking this same list in this same order. That
        # holds only while the order is numeric; if Isaac Lab ever returns
        # lexicographic order instead, every env past env_9 silently gets an
        # object its observation does not describe. Fail loudly instead.
        observed = [_env_id_of(p) for p in object_prim_paths]
        if observed != sorted(observed):
            raise RuntimeError(
                "find_matching_prim_paths no longer returns Object prims in "
                f"numeric env order (first 12: {observed[:12]}). "
                "_build_object_scale_tensor's asset assignment assumes it does; "
                "fix the assignment before training on this."
            )
    for source_idx, obj_path in enumerate(object_prim_paths):
        env_id = _env_id_of(obj_path)
        asset_index = (authored_map[env_id] if authored_map is not None
                       else source_idx % num_object_usds)
        env._object_scale_per_env[env_id] = torch.tensor(
            object_scales_normalized[asset_index], device=env.device, dtype=torch.float32,
        )
        env._object_asset_index_per_env[env_id] = asset_index


def setup_scene(env) -> None:
    """Build and register robot, table, object, goal, ground, and light."""
    assets_cfg = env.cfg.assets
    setup_t0 = time.perf_counter()
    _log_scene_step(
        setup_t0,
        f"setup start num_envs={env.num_envs} "
        f"num_assets_per_type={assets_cfg.num_assets_per_type}",
    )

    # 1. Resolve the object pool: a single named URDF (DexToolBench eval) or
    #    procedural URDFs generated in a per-launch temp dir.
    env._tmp_asset_dir = tempfile.mkdtemp(prefix="genmech_assets_")
    if assets_cfg.object_urdf:
        if assets_cfg.object_scale is None:
            raise ValueError(
                "cfg.assets.object_scale must be set when object_urdf is given "
                "(policy-normalized grasp-bbox scale, NAME_TO_OBJECT convention)."
            )
        urdf_paths = [resolve_repo_path(assets_cfg.object_urdf)]
        if not urdf_paths[0].exists():
            raise FileNotFoundError(f"object_urdf not found: {urdf_paths[0]}")
        object_scales_normalized = [tuple(assets_cfg.object_scale)]
        # A caller-supplied URDF is not generated from parameters we hold, so it
        # cannot be authored -- only converted.
        object_params = None
    else:
        urdf_paths, object_scales_normalized, object_params = generate_handle_head_urdfs(
            handle_head_types=tuple(assets_cfg.handle_head_types),
            num_per_type=assets_cfg.num_assets_per_type,
            out_dir=env._tmp_asset_dir,
            shuffle=assets_cfg.shuffle_assets,
            seed=assets_cfg.object_seed,
            density_scale=assets_cfg.object_density_scale,
        )
    if not urdf_paths:
        raise ValueError(
            "No procedural object URDFs were generated. "
            "Check cfg.assets.handle_head_types and num_assets_per_type."
        )
    env._object_urdf_paths = [str(path) for path in urdf_paths]
    _log_scene_step(setup_t0, f"generated {len(urdf_paths)} object URDFs")

    # 2. Convert URDFs -> raw USDs -> role-specific baked USDs.
    usd_work_dir = Path(env._tmp_asset_dir) / "usd"
    bake_root = Path(env._tmp_asset_dir) / "baked_usd"
    usd_work_dir.mkdir(parents=True, exist_ok=True)

    # Objects can be AUTHORED instead of converted. Each is one rigid body with
    # two analytic shapes and a mass/inertia generate_objects already computes in
    # closed form, so the URDF round-trip through Kit's importer recovers numbers
    # we started with -- measured at 0.2 s per object, ~240 s for the 1200-object
    # pool. Authoring is ~5 prims each.
    #
    # Verified equivalent before being offered: mass, inertia and centre of mass
    # agree to ~1e-8 and dropped-object resting poses to ~2e-7 m
    # (genmech.tools.compare_object_assets, compare_object_physics). The capsule
    # detail matters and is handled in author_objects: this conversion passes
    # replace_cylinders_with_capsules=True, so a URDF cylinder becomes a CAPSULE
    # whose height is the cylindrical section.
    # "object" / "goalviz" author just one of the pair; used to bisect which
    # asset carries a behavioural difference rather than guessing attributes.
    author_objects = bool(getattr(assets_cfg, "author_object_usds", False))
    _which_cfg = str(getattr(assets_cfg, "author_which", "both")).lower()
    _author_which = ({"object", "goalviz"} if _which_cfg == "both"
                     else {_which_cfg}) if author_objects else set()
    if author_objects and object_params is None:
        raise ValueError(
            "assets.author_object_usds=True requires the procedural object pool; "
            "assets.object_urdf is set, and a caller-supplied URDF carries no "
            "parameters to author from."
        )

    object_raw_usds = [
        _convert_urdf_to_usd(
            str(urdf), usd_work_dir, fix_base=False, replace_cylinders_with_capsules=True,
        )
        for urdf in urdf_paths
    ] if not (author_objects and _author_which == {"object", "goalviz"}) else []
    object_usd_paths = [
        _bake_usd(usd, bake_root, "object", props=dict(
            kinematic_enabled=False, disable_gravity=False,
            max_depenetration_velocity=1000.0, articulation_enabled=False,
        ))
        for usd in object_raw_usds
    ]
    goalviz_usd_paths = [
        _bake_usd(usd, bake_root, "goalviz", props=dict(
            kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
        ), collision_enabled=False)
        for usd in object_raw_usds
    ]

    spec = env.robot_spec

    def _prepare_robot_usd(design_spec, tag: str) -> str:
        """URDF -> USD -> self-collision filters -> baked USD, for one design."""
        converted = _convert_urdf_to_usd(
            assets_cfg.robot_urdf or design_spec.urdf_path, usd_work_dir,
            fix_base=True, self_collision=True,
            joint_drive=_robot_joint_drive_cfg(),
            replace_cylinders_with_capsules=design_spec.replace_cylinders_with_capsules,
        )
        # Isaac Gym enables all robot self-collisions then masks adjacent links;
        # mirror that by authoring FilteredPairsAPI for the spec's adjacency
        # pairs before the bake (PhysX additionally auto-filters directly-jointed
        # parent/child links). The adjacency is PER DESIGN -- geometry-less
        # ghosted links are transparent, so which pairs need filtering differs.
        _apply_self_collision_filters(converted, design_spec.adjacent_links)
        return _bake_usd(
            converted, bake_root, tag,
            props=dict(
                disable_gravity=True, max_depenetration_velocity=1000.0,
                enabled_self_collisions=True,
                solver_position_iterations=8, solver_velocity_iterations=0,
            ),
            apply_physx_articulation=True,
        )

    # Resolved by the env before super().__init__ (the observation spaces are
    # derived from it); re-resolve only when setup_scene is driven directly, as
    # the benchmarking tools do.
    def _author_population_usds(specs) -> list[str]:
        """Author one USD per design, with the arm converted ONCE and referenced.

        Measured against the converter on this cluster: 1.17 s per design (~8 h
        for 24,576) becomes ~29 ms (~12 min), because ~90% of a conversion is
        re-importing the SAME arm's 16 STL meshes.

        The authored asset is verified equivalent to the converted one by
        genmech.tools.compare_authored_robot: masses, inertias, joint limits,
        colliders and every actuation property agree exactly, and driven
        identically the fingertips land within 0.0013 mm.
        """
        import numpy as np

        from isaacsimenvs.authoring.author_robot import (
            arm_only_urdf, author_robot_usd, flatten_arm_usd,
        )
        from hand_sampler.population import load_population, load_population_any
        from pxr import Sdf, Usd, UsdGeom

        arm_dir = Path(env._tmp_asset_dir) / "arm"
        arm_urdf = arm_only_urdf(arm_dir / "iiwa14_arm_only.urdf")
        arm_raw = _convert_urdf_to_usd(
            str(arm_urdf), arm_dir, fix_base=True, self_collision=True,
            joint_drive=_robot_joint_drive_cfg())
        # Flatten: the converter's output references configuration/*_base.usd,
        # and those do NOT resolve through a second level of nesting -- a
        # referenced arm otherwise composes with NO collision geometry.
        arm_usd = flatten_arm_usd(arm_raw, arm_dir / "arm_flat.usd")
        arm_stage = Usd.Stage.Open(arm_usd)
        arm_root = str(next(c for c in arm_stage.GetPseudoRoot().GetChildren()).GetPath())
        link7_world = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(
            arm_stage.GetPrimAtPath(f"{arm_root}/iiwa14_link_7"))).T
        _log_scene_step(setup_t0, "converted the shared arm once")

        layer_for_envs = get_current_stage().GetRootLayer()
        hands = load_population_any(
            getattr(assets_cfg, "robot_population_seed", None),
            getattr(assets_cfg, "robot_population_path", None))
        count = int(getattr(assets_cfg, "robot_population_count", 0) or 0)
        if count:
            hands = hands[:count]
        if len(hands) != len(specs):
            raise RuntimeError(
                f"population/spec mismatch: {len(hands)} hands, {len(specs)} specs")

        if bool(getattr(assets_cfg, "author_robots_into_envs", True)):
            # DIRECTLY INTO THE ENV PRIMS, no files and no MultiUsdFileCfg.
            #
            # Writing a USD per design and spawning it makes Isaac Lab compose
            # each design into each env, which is the 0.00018*k*n term in
            # docs/multi_embodiment.md 3 -- about 30 h at k = n = 24,576, and
            # the reason a 24k run sat in the scene build for hours AFTER
            # authoring completed in 28 min. Authoring into the env prims is
            # what bench_authored_sps measured at 362 s for k = n = 24,576, and
            # is exactly the pattern _author_objects_into_envs already uses.
            from isaacsimenvs.authoring.author_robot import author_robot_prims
            from isaacsimenvs.authoring.author_usd import _set_xform

            env._robot_design_index_per_env = torch.tensor(
                [i % len(specs) for i in range(env.num_envs)],
                device=env.device, dtype=torch.long)
            env._robot_collider_links = {}
            # The same pose build_robot_articulation_usd_cfg puts in init_state,
            # applied to the prim instead because a fixed base ignores the other.
            base_pos = env.robot_spec.base_pos
            base_rot = env.robot_spec.base_rot
            t_auth = time.perf_counter()
            with Sdf.ChangeBlock():
                for env_path in sorted(env.scene.env_prim_paths, key=_env_id_of):
                    idx = _env_id_of(env_path) % len(specs)
                    _, _cl = author_robot_prims(
                        layer_for_envs, f"{env_path}/Robot", hands[idx],
                        specs[idx], arm_usd=arm_usd, arm_root_prim=arm_root,
                        link7_world=link7_world,
                        adjacency=specs[idx].adjacent_links,
                        in_change_block=True)
                    # Recorded HERE, where the geometry is created, so the
                    # friction pass does not have to rediscover it per link.
                    env._robot_collider_links[idx] = _cl
                    # PLACE THE BASE. The arm is FIXED-BASE: its root_joint pins
                    # it to the world wherever the prim sits, so init_state.pos
                    # cannot move it afterwards -- writing a root pose to a
                    # fixed articulation does nothing. Isaac Lab's spawner
                    # normally applies translation/orientation when it CREATES
                    # the prim, but spawn=None means nothing creates it, so
                    # nothing applied it: the robot sat at the env origin, on
                    # top of the table instead of 0.8 m behind it, with link_7
                    # reaching to y = -0.575 instead of +0.181.
                    _root_spec = layer_for_envs.GetPrimAtPath(f"{env_path}/Robot")
                    _set_xform(_root_spec, tuple(float(v) for v in base_pos),
                               tuple(float(v) for v in base_rot))
            _log_scene_step(
                setup_t0,
                f"authored {env.num_envs} robots into env prims from "
                f"{len(specs)} designs "
                f"({(time.perf_counter() - t_auth) / max(env.num_envs, 1) * 1000:.2f} ms each)")
            return None          # signals: nothing to spawn

        out = []
        design_dir = Path(env._tmp_asset_dir) / "designs"
        for i, (hand, design_spec) in enumerate(zip(hands, specs)):
            # ONE DIRECTORY PER DESIGN. _bake_usd copies the source file's
            # sibling directories (shutil.copytree over the parent's children),
            # so putting every authored design in one folder makes baking design
            # i copy the i-1 designs before it -- O(n^2), and measured at 39
            # designs/min falling, i.e. SLOWER than the 51/min converter this
            # path exists to beat.
            raw = author_robot_usd(
                hand, design_spec, design_dir / hand.name / f"{hand.name}.usd",
                arm_usd=arm_usd, arm_root_prim=arm_root, link7_world=link7_world)
            # The same finish the converted path gets, so self-collision
            # filtering and the articulation/solver properties are identical.
            _apply_self_collision_filters(raw, design_spec.adjacent_links)
            out.append(_bake_usd(
                raw, bake_root, f"robot_{i:05d}",
                props=dict(disable_gravity=True, max_depenetration_velocity=1000.0,
                           enabled_self_collisions=True,
                           solver_position_iterations=8, solver_velocity_iterations=0),
                apply_physx_articulation=True))
        return out

    population_specs = getattr(env, "_robot_population_specs", None)
    if population_specs is None and getattr(
            assets_cfg, "robot_population_seed", None) is not None:
        population_specs = _resolve_robot_population(assets_cfg)
    if population_specs is None:
        robot_usd_arg: str | list[str] = _prepare_robot_usd(spec, "robot")
        env._robot_population_specs = None
    else:
        # One asset per DESIGN, not per env: n envs share k designs, and
        # preparing per env is what turned a 24,576-env build into hours
        # (docs/multi_embodiment.md §3).
        if bool(getattr(assets_cfg, "author_robot_usds", False)):
            robot_usd_arg = _author_population_usds(population_specs)
            # None means the prims are already on the stage: spawn nothing.
            if robot_usd_arg is None:
                robot_usd_arg = "__authored_in_place__"
        else:
            robot_usd_arg = [
                _prepare_robot_usd(s, f"robot_{i:05d}")
                for i, s in enumerate(population_specs)
            ]
        env._robot_population_specs = population_specs
        if isinstance(robot_usd_arg, str):
            # The sentinel, not a path list. len() on it reported "21 distinct
            # robot USDs" -- the length of "__authored_in_place__" -- for a run
            # that wrote no USDs at all.
            _log_scene_step(setup_t0, "robot prims already on the stage, "
                                      "nothing to spawn")
        else:
            _log_scene_step(
                setup_t0, f"prepared {len(robot_usd_arg)} distinct robot USDs")
    robot_usd_path = robot_usd_arg
    # Table USD(s). When table_scale_range_x/y are non-trivial and
    # table_scale_num_variants > 1, pre-bake N scaled URDF variants and pass
    # them as a list to RigidObject — Isaac Lab's MultiUsdFileCfg cycles
    # through the list, giving each env one of the variants. Z scale is held
    # at 1.0 so the table surface height matches the policy's expectation.
    scale_range_x = tuple(float(v) for v in getattr(assets_cfg, "table_scale_range_x", (1.0, 1.0)))
    scale_range_y = tuple(float(v) for v in getattr(assets_cfg, "table_scale_range_y", (1.0, 1.0)))
    n_table_variants = int(getattr(assets_cfg, "table_scale_num_variants", 1))
    table_scale_is_trivial = (
        scale_range_x == (1.0, 1.0) and scale_range_y == (1.0, 1.0)
    ) or n_table_variants <= 1
    if table_scale_is_trivial:
        table_usd_paths = [_bake_usd(
            _convert_urdf_to_usd(assets_cfg.table_urdf, usd_work_dir, fix_base=False),
            bake_root, "table",
            props=dict(
                kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
            ),
        )]
        # Single (sx, sy) = (1.0, 1.0) for downstream consumers (eval viz).
        env._table_variant_scales = [(1.0, 1.0)]
    else:
        variant_urdf_dir = Path(env._tmp_asset_dir) / "table_variants"
        variant_urdf_paths, variant_scales = _generate_scaled_table_urdfs(
            base_urdf_path=assets_cfg.table_urdf,
            num_variants=n_table_variants,
            scale_range_x=scale_range_x,
            scale_range_y=scale_range_y,
            out_dir=variant_urdf_dir,
            # Deterministic across runs so the on-disk variants are stable.
            # The env-level seed governs which variant lands in which env via
            # Isaac Lab's round-robin spawn ordering.
            seed=0,
        )
        env._table_variant_scales = list(variant_scales)
        table_usd_paths = [
            _bake_usd(
                _convert_urdf_to_usd(p, usd_work_dir, fix_base=False),
                bake_root, f"table_variant_{idx:03d}",
                props=dict(
                    kinematic_enabled=True, disable_gravity=True, articulation_enabled=False,
                ),
            )
            for idx, p in enumerate(variant_urdf_paths)
        ]
        # variant_scales already stashed above for downstream consumers.
        _log_scene_step(
            setup_t0,
            f"baked {len(table_usd_paths)} scaled table USD variants "
            f"x_range={scale_range_x} y_range={scale_range_y}",
        )
    _log_scene_step(setup_t0, "resolved baked USDs")

    # 3. Pre-create env roots so regex spawns resolve to every env.
    _materialize_env_prims(env)
    # These three steps were a single silent block, and at 24,576 authored
    # robots (~5.8M prims on the stage) it stayed silent for over 20 minutes
    # with no way to tell which step was responsible -- or whether it was
    # progressing at all. Anything that traverses or recomposes the stage per
    # env costs O(n x stage) here, which is the shape of every scaling bug found
    # on this path so far, so each step now reports its own time.
    _log_scene_step(setup_t0, "materialized env prims")

    # 4. Spawn assets.
    env.robot = Articulation(build_robot_articulation_usd_cfg(
        robot_usd_path,
        env.robot_spec,
        start_arm_higher=getattr(env.cfg.reset, "start_arm_higher", False),
    ))
    _log_scene_step(setup_t0, "built robot articulation")
    env.table = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/Table", table_usd_paths))
    _log_scene_step(setup_t0, "spawned table")
    if author_objects:
        # Author one object and one goal marker INTO each env, cycling the pool
        # the same way MultiUsdFileCfg would. This replaces both the per-variant
        # conversion (~0.2 s x 1200) and the proto/copy spawn machinery: there is
        # no USD file, no template prim, and no Sdf.CopySpec.
        n_pool = len(object_params)
        _author_objects_into_envs(env, object_params, n_pool, which=_author_which)
        env.object = (RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object", spawn=None))
            if "object" in _author_which
            else RigidObject(build_rigid_object_cfg(
                "/World/envs/env_.*/Object", object_usd_paths)))
        env.goal_viz = (RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/GoalViz", spawn=None))
            if "goalviz" in _author_which
            else RigidObject(build_rigid_object_cfg(
                "/World/envs/env_.*/GoalViz", goalviz_usd_paths)))
    else:
        env.object = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/Object", object_usd_paths))
        env.goal_viz = RigidObject(build_rigid_object_cfg("/World/envs/env_.*/GoalViz", goalviz_usd_paths))
    _log_scene_step(setup_t0, "spawned robot/table/object/goalviz")

    # 5. Ground plane + dome light (global, outside env_*).
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 6. Per-env scale tensor for spawned Objects.
    _build_object_scale_tensor(
        env, object_scales_normalized,
        len(object_params) if author_objects else len(object_usd_paths))
    if env._robot_population_specs is not None:
        _build_robot_design_tensor(env, len(env._robot_population_specs))

    # 7. Register with scene so DirectRLEnv refreshes their tensors each step.
    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    _log_scene_step(setup_t0, "registered assets with scene")
