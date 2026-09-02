"""URDF to USD for mesh assets: rewrite a URDF into what Kit's converter
accepts, convert it, and apply the SDF collision markers and self-collision
filters it declares. PhysX properties are applied by the spawner, not here.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from hand_sampler.paths import resolve as resolve_repo_path




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
