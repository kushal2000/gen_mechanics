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

from hand_sampler.inertia import compute_mass_and_inertia

from .object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS, Scale, Scale3

_SEED = 42
_NUM_OBJECTS_PER_TYPE_DEFAULT = 100
_OBJECT_BASE_SIZE = 0.04  # reward-space normalisation, the env's object_base_size
# One link name for every object: RigidObject derives its view regex from env_0.
OBJECT_ROOT_LINK = "object_root"
_BROWN = '<material name="brown"><color rgba="0.55 0.27 0.07 1.0"/></material>'
_GRAY = '<material name="gray"><color rgba="0.5 0.5 0.5 1.0"/></material>'
_AXIS_X = "0 -1.5707963267948966 0"  # rpy putting a cylinder's axis along link x
_AXIS_Y = "-1.5707963267948966 0 0"


# --- URDF emitters ------------------------------------------------------------------

def _box_geom(scale, xyz: str = "0 0 0", rpy: str = "0 0 0") -> str:
    lx, ly, lz = scale
    return f'<origin xyz="{xyz}" rpy="{rpy}"/>\n      <geometry><box size="{lx} {ly} {lz}"/></geometry>'


def _cylinder_geom(height, radius, xyz: str = "0 0 0", rpy: str = "0 0 0") -> str:
    return (f'<origin xyz="{xyz}" rpy="{rpy}"/>\n'
            f'      <geometry><cylinder length="{height}" radius="{radius}"/></geometry>')


def _write_urdf(path: Path, name: str, parts, mass, ixx, iyy, izz, inertial_origin: str) -> Path:
    """One-link URDF with explicit <mass>/<inertia>: Isaac Sim's importer has no
    <density> fallback and would use 1 kg."""
    body = "".join(
        f"    <visual>\n      {geom}\n      {material}\n    </visual>\n"
        f"    <collision>\n      {geom}\n    </collision>\n"
        for geom, material in parts)
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
) -> tuple[list[str], list[Scale3], list[tuple]]:
    """Write a pool of URDFs and return ``(paths, scales_normalized, params)``
    in final (shuffled) order, so env i takes entry ``i % len(pool)``.

    ``params`` are the ``(handle_scale, head_scale, handle_density, head_density)``
    each URDF was written from, for authoring the same object directly.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.suffix == ".urdf":
                p.unlink()
    else:
        os.makedirs(out_dir)

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
