"""Procedural handle-head URDF generation for PoseReach.

Ported from isaacgymenvs' simtoolreal generate_objects: a single cuboid or
cylinder handle, or a handle + head composite as one link with variable
densities and parallel-axis-adjusted inertia. Pools are drawn under
``np.random.seed`` in a fixed order, so a seed yields the same pool everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from hand_sampler.design_space import compute_mass_and_inertia

from .object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS, Scale, Scale3

_SEED = 42
_NUM_OBJECTS_PER_TYPE_DEFAULT = 100
_OBJECT_BASE_SIZE = 0.04  # reward-space normalisation, the env's object_base_size
# One link name for every object: RigidObject derives its view regex from env_0.
OBJECT_ROOT_LINK = "object_root"
_BROWN = '<material name="brown"><color rgba="0.55 0.27 0.07 1.0"/></material>'
_GRAY = '<material name="gray"><color rgba="0.5 0.5 0.5 1.0"/></material>'
# A CUBE'S ORIENTATION IS INVISIBLE without marked faces, which is the whole
# difficulty of watching a reorientation policy: the object looks identical in 24
# distinct orientations, so "is it at the goal" cannot be judged by eye. Standard
# Rubik colours, opposite faces paired the way a real cube pairs them.
_RUBIK_FACES = (
    #  axis, sign, name,     rgb
    (0, +1, "red",    "0.72 0.07 0.20"),
    (0, -1, "orange", "0.93 0.45 0.13"),
    (1, +1, "blue",   "0.05 0.28 0.75"),
    (1, -1, "green",  "0.11 0.55 0.24"),
    (2, +1, "white",  "0.95 0.95 0.95"),
    (2, -1, "yellow", "0.98 0.85 0.10"),
)
_BLACK = '<material name="black"><color rgba="0.08 0.08 0.09 1.0"/></material>'
# Sticker footprint as a fraction of the face, and thickness as a fraction of the
# edge. The inset leaves the black body showing as a border, which is what makes
# the faces readable at the size the viewer draws them.
_STICKER_INSET = 0.82
_STICKER_THICKNESS = 0.02


def _rubik_stickers(scale) -> list[tuple[str, str]]:
    """Six visual-only face plates for a cube, sitting just proud of each face.

    Proud rather than flush: a plate coplanar with the body's own visual z-fights
    with it and flickers as the camera moves.
    """
    edge = float(min(scale))
    t = _STICKER_THICKNESS * edge
    face = _STICKER_INSET * edge
    out = []
    for axis, sign, name, rgb in _RUBIK_FACES:
        size = [face, face, face]
        size[axis] = t
        xyz = [0.0, 0.0, 0.0]
        xyz[axis] = sign * (0.5 * edge + 0.5 * t)
        geom = _box_geom(size, xyz=f"{xyz[0]} {xyz[1]} {xyz[2]}")
        out.append((geom, f'<material name="{name}"><color rgba="{rgb} 1.0"/></material>'))
    return out


def _is_cube(scale) -> bool:
    return len(scale) == 3 and max(scale) - min(scale) < 1e-9 * max(scale) + 1e-6


_AXIS_X = "0 -1.5707963267948966 0"  # rpy putting a cylinder's axis along link x
_AXIS_Y = "-1.5707963267948966 0 0"


# --- URDF emitters ------------------------------------------------------------------

def _box_geom(scale, xyz: str = "0 0 0", rpy: str = "0 0 0") -> str:
    lx, ly, lz = scale
    return f'<origin xyz="{xyz}" rpy="{rpy}"/>\n      <geometry><box size="{lx} {ly} {lz}"/></geometry>'


def _cylinder_geom(height, radius, xyz: str = "0 0 0", rpy: str = "0 0 0") -> str:
    return (f'<origin xyz="{xyz}" rpy="{rpy}"/>\n'
            f'      <geometry><cylinder length="{height}" radius="{radius}"/></geometry>')


def _write_urdf(path: Path, name: str, parts, mass, ixx, iyy, izz, inertial_origin: str,
                visual_only=()) -> Path:
    """One-link URDF with explicit <mass>/<inertia>: Isaac Sim's importer has no
    <density> fallback and would use 1 kg.

    ``visual_only`` parts get a <visual> and NO <collision>. That distinction is
    load-bearing: the Rubik face stickers are decoration, and emitting a collider
    for each of them would change the contact geometry of every object that has
    them, i.e. silently change the physics of the task they were added to explain.
    """
    body = "".join(
        f"    <visual>\n      {geom}\n      {material}\n    </visual>\n"
        f"    <collision>\n      {geom}\n    </collision>\n"
        for geom, material in parts)
    body += "".join(
        f"    <visual>\n      {geom}\n      {material}\n    </visual>\n"
        for geom, material in visual_only)
    path.write_text(
        f'<?xml version="1.0"?>\n<robot name="{name}">\n  <link name="{OBJECT_ROOT_LINK}">\n'
        f"{body}    <inertial>\n      <origin {inertial_origin}/>\n      <mass value=\"{mass}\"/>\n"
        f'      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>\n'
        f"    </inertial>\n  </link>\n</robot>\n")
    return path


def generate_handle_urdf(path: Path, handle_scale: Scale, handle_density: float = 400.0) -> Path:
    """A single cuboid (3-tuple) or cylinder (2-tuple: height, diameter) handle."""
    if len(handle_scale) == 3:
        m, ixx, iyy, izz = compute_mass_and_inertia(handle_scale, handle_density)
        # A cube gets Rubik faces and a black body; every other cuboid keeps brown.
        # Gated on the SHAPE rather than on a flag because a cube is exactly the
        # case whose orientation is unreadable without them -- an elongated handle
        # already shows how it is turned.
        if _is_cube(handle_scale):
            return _write_urdf(path, "cuboid", [(_box_geom(handle_scale), _BLACK)],
                               m, ixx, iyy, izz, 'xyz="0 0 0" rpy="0 0 0"',
                               visual_only=_rubik_stickers(handle_scale))
        return _write_urdf(path, "cuboid", [(_box_geom(handle_scale), _BROWN)],
                           m, ixx, iyy, izz, 'xyz="0 0 0" rpy="0 0 0"')
    if len(handle_scale) == 2:
        h, d = handle_scale
        # Inertia is in the geometry frame (axis z); the same rpy on the inertial origin keeps it right.
        m, ixx, iyy, izz = compute_mass_and_inertia((h, d), handle_density)
        return _write_urdf(path, "cylinder", [(_cylinder_geom(h, d / 2, rpy=_AXIS_X), _BROWN)],
                           m, ixx, iyy, izz, f'xyz="0 0 0" rpy="{_AXIS_X}"')
    raise ValueError(f"Invalid handle_scale: {handle_scale}")


def _handle_head_urdf(path: Path, handle_scale: Scale, head_scale: Scale,
                      handle_density: float, head_density: float) -> Path:
    """Handle + head as one link, inertia shifted to the composite COM."""
    if len(handle_scale) == 3:
        handle_geom = _box_geom(handle_scale)
        handle_mass, handle_ixx, handle_iyy, handle_izz = compute_mass_and_inertia(
            handle_scale, handle_density)
    else:
        h, d = handle_scale
        handle_geom = _cylinder_geom(h, d / 2, rpy=_AXIS_X)
        handle_mass, handle_izz, handle_iyy, handle_ixx = compute_mass_and_inertia(
            handle_scale, handle_density)  # axis along +x: ixx <-> izz
    if len(head_scale) == 3:
        x_offset = handle_scale[0] / 2 + head_scale[0] / 2
        head_geom = _box_geom(head_scale, xyz=f"{x_offset} 0 0")
        head_mass, head_ixx, head_iyy, head_izz = compute_mass_and_inertia(
            head_scale, head_density)
    else:
        hh, hd = head_scale
        x_offset = handle_scale[0] / 2 + hd / 2
        head_geom = _cylinder_geom(hh, hd / 2, xyz=f"{x_offset} 0 0", rpy=_AXIS_Y)
        head_mass, head_ixx, head_izz, head_iyy = compute_mass_and_inertia(
            head_scale, head_density)  # axis along +y: iyy <-> izz

    total_mass = handle_mass + head_mass
    com_x = (handle_mass * 0.0 + head_mass * x_offset) / total_mass
    d_handle, d_head = -com_x, x_offset - com_x
    ixx = handle_ixx + head_ixx
    iyy = (handle_iyy + handle_mass * d_handle * d_handle) + (head_iyy + head_mass * d_head * d_head)
    izz = (handle_izz + handle_mass * d_handle * d_handle) + (head_izz + head_mass * d_head * d_head)
    return _write_urdf(path, "handle_head", [(handle_geom, _BROWN), (head_geom, _GRAY)],
                       total_mass, ixx, iyy, izz, f'xyz="{com_x} 0 0" rpy="0 0 0"')


def generate_handle_head_urdf(path: Path, handle_scale: Scale, head_scale: Scale | None,
                              handle_density: float = 400.0,
                              head_density: float | None = 800.0) -> Path:
    """A handle-only object (head scale and density both None) or a handle+head composite."""
    if head_scale is None and head_density is None:
        return generate_handle_urdf(path, handle_scale, handle_density)
    if head_scale is not None and head_density is not None:
        return _handle_head_urdf(path, handle_scale, head_scale, handle_density, head_density)
    raise ValueError(f"head_scale and head_density must both be set or both None "
                     f"(got {head_scale} and {head_density})")


# --- the pool ------------------------------------------------------------------------

def _scale_to_3d(scale) -> Scale3:
    """Cylinder (h, d) -> (h, d, d), so every returned scale is a 3-tuple."""
    if len(scale) == 3:
        return float(scale[0]), float(scale[1]), float(scale[2])
    if len(scale) == 2:
        return float(scale[0]), float(scale[1]), float(scale[1])
    raise ValueError(f"Invalid scale: {scale}")


def matching_distributions(handle_head_types: tuple[str, ...]):
    """The ObjectSizeDistributions a pool of these types draws from, in draw order.

    A type can have several shape variants, so the pool size is
    ``num_per_type * len(matching_distributions(types))``, not ``* len(types)``.
    """
    type_set = set(handle_head_types)
    matching = [d for d in OBJECT_SIZE_DISTRIBUTIONS if d.type in type_set]
    if not matching:
        raise ValueError(
            f"No matching ObjectSizeDistribution for handle_head_types={handle_head_types}. "
            f"Valid types: {sorted({d.type for d in OBJECT_SIZE_DISTRIBUTIONS})}")
    return matching


def sample_pool_params(
    handle_head_types: tuple[str, ...],
    num_per_type: int = _NUM_OBJECTS_PER_TYPE_DEFAULT,
    object_base_size: float = _OBJECT_BASE_SIZE,
    seed: int = _SEED,
    shuffle: bool = True,
    density_scale: float = 1.0,
) -> tuple[list[dict], list[int]]:
    """Replay a pool's RNG draws without writing anything.

    All randomness lives here, so a run's ``(types, num_per_type, seed,
    shuffle, density_scale)`` reconstructs its pool exactly. Returns the
    entries in pre-shuffle order and ``permutation``, where ``permutation[k]``
    is the pre-shuffle index of final entry ``k``.
    """
    np.random.seed(seed)
    entries: list[dict] = []
    for dist_index, dist in enumerate(matching_distributions(handle_head_types)):
        # Draw order is fixed: densities first, then scales.
        handle_densities = dist.sample_handle_densities(num_per_type)
        head_densities = dist.sample_head_densities(num_per_type)
        handle_scales = dist.sample_handle_scales(num_per_type)
        head_scales = dist.sample_head_scales(num_per_type)
        for idx in range(num_per_type):
            h_scale = tuple(float(x) for x in handle_scales[idx])
            head = tuple(float(x) for x in head_scales[idx]) if head_scales is not None else None
            # density_scale is applied after sampling so it changes mass, not geometry.
            h_d = float(handle_densities[idx]) * density_scale
            head_d = float(head_densities[idx]) * density_scale if head_densities is not None else None
            x, y, z = _scale_to_3d(h_scale)
            entries.append({
                "type": dist.type,
                "shape": dist.shape,
                "distribution_index": dist_index,
                "sample_index": idx,
                "handle_scale": h_scale,
                "head_scale": head,
                "handle_density": h_d,
                "head_density": head_d,
                "scale_normalized": (x / object_base_size, y / object_base_size, z / object_base_size),
            })
    if shuffle:  # so type order does not bias env i -> asset i % N
        indices = np.arange(len(entries))
        np.random.shuffle(indices)
        permutation = [int(i) for i in indices]
    else:
        permutation = list(range(len(entries)))
    return entries, permutation


def pool_urdf_filename(entry: dict) -> str:
    """The URDF filename of a pool entry, a pure function of its parameters."""
    return (f"{entry['sample_index']:03d}_{entry['type']}"
            f"_handle_{entry['handle_scale']}_head_{entry['head_scale']}"
            f"_d{entry['handle_density']:.1f}_{entry['head_density']}".replace(".", "-") + ".urdf")


def generate_handle_head_urdfs(
    handle_head_types: tuple[str, ...],
    num_per_type: int = _NUM_OBJECTS_PER_TYPE_DEFAULT,
    out_dir: str | Path = "/tmp/genmech_assets",
    object_base_size: float = _OBJECT_BASE_SIZE,
    seed: int = _SEED,
    shuffle: bool = True,
    density_scale: float = 1.0,
    curated: str | None = None,
) -> tuple[list[str], list[Scale3], list[tuple]]:
    """Write a pool of URDFs and return ``(paths, scales_normalized, params)``
    in final (shuffled) order, so env i takes entry ``i % len(pool)``.

    ``params`` are the ``(handle_scale, head_scale, handle_density, head_density)``
    each URDF was written from, for authoring the same object directly.

    ``curated`` names a hand-picked pool in ``curated_pools`` instead of
    sampling; ``num_per_type``, ``seed`` and ``shuffle`` are then unused and
    the pool keeps its listed order.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.suffix == ".urdf":
                p.unlink()
    else:
        os.makedirs(out_dir)

    if curated:
        from .curated_pools import curated_entries
        entries = curated_entries(curated, object_base_size, density_scale)
        permutation = list(range(len(entries)))
    else:
        entries, permutation = sample_pool_params(
            handle_head_types=handle_head_types, num_per_type=num_per_type,
            object_base_size=object_base_size, seed=seed, shuffle=shuffle,
            density_scale=density_scale)
    paths, scales_norm, params = [], [], []
    for entry in entries:
        urdf_path = out_dir / pool_urdf_filename(entry)
        generate_handle_head_urdf(
            path=urdf_path, handle_scale=entry["handle_scale"], head_scale=entry["head_scale"],
            handle_density=entry["handle_density"], head_density=entry["head_density"])
        paths.append(str(urdf_path))
        scales_norm.append(entry["scale_normalized"])
        params.append((entry["handle_scale"], entry["head_scale"],
                       entry["handle_density"], entry["head_density"]))
    return ([paths[i] for i in permutation], [scales_norm[i] for i in permutation],
            [params[i] for i in permutation])


__all__ = [
    "OBJECT_ROOT_LINK",
    "generate_handle_head_urdfs",
    "matching_distributions",
    "sample_pool_params",
    "pool_urdf_filename",
]
