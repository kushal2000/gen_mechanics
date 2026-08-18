"""Procedural handle-head URDF generation for PoseReach.

Clean port of the minimal subset of isaacgymenvs/tasks/simtoolreal/generate_objects.py
needed for the base goal-pose-reaching env: `generate_handle_urdf` (single cuboid
or cylinder) and `generate_handle_head_urdf` (handle + head composite with
variable densities + parallel-axis-adjusted inertia). No trimesh / vhacd code.

The top-level entry `generate_handle_head_urdfs(types, num_per_type, out_dir)`
mirrors legacy `env.py:_handle_head_primitives`: per-type sampling with
`np.random.seed(42)`, pool shuffle, and scales normalized by `object_base_size`.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS, Scale2, Scale3

Scale = Union[Scale2, Scale3]

# Matches legacy env.py:1732.
_SEED = 42
# Matches legacy env.py:1731; kept module-level so call sites can override for tests.
_NUM_OBJECTS_PER_TYPE_DEFAULT = 100
# Reward-space normalization constant (same value the env uses when rescaling).
_OBJECT_BASE_SIZE = 0.04


# ----------------------------------------------------------------------------
# Primitive URDF emitters
# ----------------------------------------------------------------------------


# All procedural object URDFs use this same link name so MultiUsdFileCfg-spawned
# prims have an identical internal structure across envs (env_*/Object/<LINK>).
# RigidObject's view regex is derived from env_0's structure and applied to all
# envs; if the link name varies, only matching envs are registered and
# set_transforms overflows the partial view.
_OBJECT_ROOT_LINK = "object_root"


def _cuboid_urdf_constant_density(path: Path, scale: Scale3, density: float) -> Path:
    lx, ly, lz = scale
    # Standard URDF requires <mass> + <inertia>. Legacy isaacgym's URDF
    # importer accepts <density> as a Gazebo extension and back-computes mass
    # from density × volume; Isaac Sim's importer follows strict URDF spec
    # and falls back to 1 kg when <mass> is missing — that breaks erasers and
    # any other handle-only types. Emit explicit <mass> + <inertia> so both
    # backends see byte-identical physics.
    m, ixx, iyy, izz = _compute_mass_and_inertia(scale, density)
    urdf = f"""<?xml version="1.0"?>
<robot name="cuboid">
  <link name="{_OBJECT_ROOT_LINK}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{lx} {ly} {lz}"/></geometry>
      <material name="brown"><color rgba="0.55 0.27 0.07 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{lx} {ly} {lz}"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{m}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""
    path.write_text(urdf)
    return path


def _cylinder_urdf_constant_density(
    path: Path, height: float, diameter: float, density: float
) -> Path:
    radius = diameter / 2
    # Same fix as the cuboid branch — emit explicit <mass> + <inertia> so
    # Isaac Sim's URDF importer doesn't fall back to a default 1 kg mass.
    # The geometry is rotated by rpy="0 -π/2 0" so the cylinder's length axis
    # aligns with link x. _compute_mass_and_inertia returns inertia in the
    # geometry-local frame (axis = z); we apply the same rpy on the inertial
    # origin so the values stay correct after rotation.
    m, ixx, iyy, izz = _compute_mass_and_inertia((height, diameter), density)
    urdf = f"""<?xml version="1.0"?>
<robot name="cylinder">
  <link name="{_OBJECT_ROOT_LINK}">
    <visual>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <geometry><cylinder length="{height}" radius="{radius}"/></geometry>
      <material name="brown"><color rgba="0.55 0.27 0.07 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <geometry><cylinder length="{height}" radius="{radius}"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <mass value="{m}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""
    path.write_text(urdf)
    return path


def generate_handle_urdf(
    path: Path, handle_scale: Scale, handle_density: float = 400.0
) -> Path:
    """Emit a URDF for a single cuboid (len(scale)==3) or cylinder (len(scale)==2) handle."""
    if len(handle_scale) == 3:
        return _cuboid_urdf_constant_density(path, handle_scale, handle_density)
    if len(handle_scale) == 2:
        return _cylinder_urdf_constant_density(
            path, handle_scale[0], handle_scale[1], handle_density
        )
    raise ValueError(f"Invalid handle_scale: {handle_scale}")


# ----------------------------------------------------------------------------
# Handle + head composite (variable density, single-link, with adjusted inertia)
# ----------------------------------------------------------------------------


def _compute_mass_and_inertia(scale: Scale, density: float):
    """Capsule-approximation for cylinders; exact for cuboids.

    Returns (m, ixx, iyy, izz) with scale-axis = z (caller flips if needed).
    """
    if len(scale) == 3:
        lx, ly, lz = scale
        v = lx * ly * lz
        m = v * density
        ixx = (1 / 12) * m * (ly * ly + lz * lz)
        iyy = (1 / 12) * m * (lx * lx + lz * lz)
        izz = (1 / 12) * m * (lx * lx + ly * ly)
        return m, ixx, iyy, izz
    if len(scale) == 2:
        h, d = scale[0], scale[1]
        r = d / 2
        # Capsule mass = cylinder + two hemispheres.
        m_c = density * math.pi * r * r * h
        m_h = density * (2 / 3) * math.pi * r ** 3
        m = m_c + 2 * m_h
        # Cylinder inertia about centroid (axis = z).
        i_c_axis = 0.5 * m_c * r * r
        i_c_perp = (1 / 12) * m_c * (3 * r * r + h * h)
        # Hemisphere inertia about its own centroid.
        i_h_axis = (2 / 5) * m_h * r * r
        i_h_perp = (83 / 320) * m_h * r * r
        d_com = (h / 2) + (3 * r / 8)
        izz = i_c_axis + 2 * i_h_axis
        ixx = iyy = i_c_perp + 2 * (i_h_perp + m_h * d_com * d_com)
        return m, ixx, iyy, izz
    raise ValueError(f"Invalid scale: {scale}")


def _handle_head_urdf_variable_density(
    path: Path,
    handle_scale: Scale,
    head_scale: Scale,
    handle_density: float,
    head_density: float,
) -> Path:
    # Handle geometry + inertia (axis along +x; for cylinders we rotate -pi/2 about y).
    if len(handle_scale) == 3:
        lx, ly, lz = handle_scale
        handle_geom = (
            f'<origin xyz="0 0 0" rpy="0 0 0"/>\n'
            f'        <geometry><box size="{lx} {ly} {lz}"/></geometry>'
        )
        handle_mass, handle_ixx, handle_iyy, handle_izz = _compute_mass_and_inertia(
            handle_scale, handle_density
        )
    else:
        h, d = handle_scale
        r = d / 2
        handle_geom = (
            f'<origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>\n'
            f'        <geometry><cylinder length="{h}" radius="{r}"/></geometry>'
        )
        # Rotated so handle axis is along +x; flip ixx ↔ izz.
        handle_mass, handle_izz, handle_iyy, handle_ixx = _compute_mass_and_inertia(
            handle_scale, handle_density
        )

    # Head geometry + inertia.
    if len(head_scale) == 3:
        hlx, hly, hlz = head_scale
        x_offset = handle_scale[0] / 2 + hlx / 2
        head_geom = (
            f'<origin xyz="{x_offset} 0 0" rpy="0 0 0"/>\n'
            f'        <geometry><box size="{hlx} {hly} {hlz}"/></geometry>'
        )
        head_mass, head_ixx, head_iyy, head_izz = _compute_mass_and_inertia(
            head_scale, head_density
        )
    else:
        hh, hd = head_scale
        hr = hd / 2
        x_offset = handle_scale[0] / 2 + hr
        head_geom = (
            f'<origin xyz="{x_offset} 0 0" rpy="-1.5707963267948966 0 0"/>\n'
            f'        <geometry><cylinder length="{hh}" radius="{hr}"/></geometry>'
        )
        # Rotated so head axis is along +y; flip iyy ↔ izz.
        head_mass, head_ixx, head_izz, head_iyy = _compute_mass_and_inertia(
            head_scale, head_density
        )

    # Parallel-axis shift to composite COM (handle at origin, head at +x_offset).
    total_mass = handle_mass + head_mass
    com_x = (handle_mass * 0.0 + head_mass * x_offset) / total_mass
    d_handle = -com_x
    d_head = x_offset - com_x

    ixx = handle_ixx + head_ixx
    iyy = (handle_iyy + handle_mass * d_handle * d_handle) + (
        head_iyy + head_mass * d_head * d_head
    )
    izz = (handle_izz + handle_mass * d_handle * d_handle) + (
        head_izz + head_mass * d_head * d_head
    )

    urdf = f"""<?xml version="1.0"?>
<robot name="handle_head">
  <link name="{_OBJECT_ROOT_LINK}">
    <visual>
      {handle_geom}
      <material name="brown"><color rgba="0.55 0.27 0.07 1.0"/></material>
    </visual>
    <collision>
      {handle_geom}
    </collision>
    <visual>
      {head_geom}
      <material name="gray"><color rgba="0.5 0.5 0.5 1.0"/></material>
    </visual>
    <collision>
      {head_geom}
    </collision>
    <inertial>
      <origin xyz="{com_x} 0 0" rpy="0 0 0"/>
      <mass value="{total_mass}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""
    path.write_text(urdf)
    return path


def generate_handle_head_urdf(
    path: Path,
    handle_scale: Scale,
    head_scale: Optional[Scale],
    handle_density: float = 400.0,
    head_density: Optional[float] = 800.0,
) -> Path:
    """Emit a URDF for a handle (no head) or a handle+head composite.

    The handle-only branch matches legacy `generate_handle_urdf`; the composite
    branch matches `generate_handle_head_urdf_variable_density` (single-link
    with parallel-axis-adjusted inertia — the 2-link variant was unstable per
    legacy env.py:1723-1724 and is not ported).
    """
    if head_scale is None and head_density is None:
        return generate_handle_urdf(path, handle_scale, handle_density)
    if head_scale is not None and head_density is not None:
        return _handle_head_urdf_variable_density(
            path, handle_scale, head_scale, handle_density, head_density
        )
    raise ValueError(
        f"head_scale and head_density must both be set or both None (got "
        f"{head_scale} and {head_density})"
    )


# ----------------------------------------------------------------------------
# Top-level: generate a pool of N×types URDFs for the env
# ----------------------------------------------------------------------------


def _scale_to_3d(scale: np.ndarray) -> tuple[float, float, float]:
    """Convert cylinder (h, d) → (h, d, d) so all returned scales are 3-tuples."""
    if len(scale) == 3:
        return (float(scale[0]), float(scale[1]), float(scale[2]))
    if len(scale) == 2:
        return (float(scale[0]), float(scale[1]), float(scale[1]))
    raise ValueError(f"Invalid scale shape: {scale.shape}")


def matching_distributions(handle_head_types: tuple[str, ...]):
    """The ObjectSizeDistributions a pool of these types draws from, in draw order.

    There are more distributions than types -- twelve for the six handle-head
    types -- because a type can have several shape variants (hammer and
    screwdriver have cuboid and cylinder handles, brush has two head shapes).
    Pool size is therefore ``num_per_type * len(matching_distributions(types))``,
    not ``num_per_type * len(types)``, which is the arithmetic every caller
    sizing a pool against an env count gets wrong first.
    """
    type_set = set(handle_head_types)
    matching = [d for d in OBJECT_SIZE_DISTRIBUTIONS if d.type in type_set]
    if not matching:
        raise ValueError(
            f"No matching ObjectSizeDistribution for handle_head_types={handle_head_types}. "
            f"Valid types: {sorted({d.type for d in OBJECT_SIZE_DISTRIBUTIONS})}"
        )
    return matching


def sample_pool_params(
    handle_head_types: tuple[str, ...],
    num_per_type: int = _NUM_OBJECTS_PER_TYPE_DEFAULT,
    object_base_size: float = _OBJECT_BASE_SIZE,
    seed: int = _SEED,
    shuffle: bool = True,
    density_scale: float = 1.0,
) -> tuple[list[dict], list[int]]:
    """Replay a pool's RNG draws without writing anything to disk.

    This is the sampling half of :func:`generate_handle_head_urdfs`, split out so
    that reconstructing a past run's pool -- which needs the parameters but not
    the URDFs -- runs the SAME draws rather than a copy of them. A copy is how
    the two silently drift, and the copy would be the one nobody notices is
    wrong: it produces a plausible pool of the right size for the wrong run.

    Everything that consumes randomness lives here (``np.random.seed``, the four
    ``sample_*`` uniforms per distribution, the final shuffle); writing a URDF
    draws nothing. So a caller that replays this function with a run's
    ``(handle_head_types, num_per_type, seed, shuffle, density_scale)`` gets that
    run's pool exactly.

    Returns
    -------
    entries : list[dict]
        One dict per pool member in **pre-shuffle** order.
    permutation : list[int]
        ``permutation[k]`` is the pre-shuffle index of final pool entry ``k``.
        Identity when ``shuffle`` is False.
    """
    np.random.seed(seed)
    matching = matching_distributions(handle_head_types)

    entries: list[dict] = []
    for dist_index, dist in enumerate(matching):
        # Sample order MUST match legacy env.py:1758-1772 (densities first,
        # scales second) -- the two pools share seed=42 so first-asset URDFs
        # come out byte-identical across backends only when the np.random
        # draws line up step for step.
        handle_densities = dist.sample_handle_densities(num_per_type)
        head_densities = dist.sample_head_densities(num_per_type)
        handle_scales = dist.sample_handle_scales(num_per_type)
        head_scales = dist.sample_head_scales(num_per_type)

        for idx in range(num_per_type):
            h_scale = tuple(float(x) for x in handle_scales[idx])
            head = tuple(float(x) for x in head_scales[idx]) if head_scales is not None else None
            # density_scale multiplies mass and inertia uniformly. Applied
            # AFTER sampling so the draw sequence -- and therefore the pool's
            # geometry -- is unchanged: a mass-only held-out condition must not
            # also perturb the shapes.
            h_d = float(handle_densities[idx]) * density_scale
            head_d = (
                float(head_densities[idx]) * density_scale
                if head_densities is not None else None
            )
            # Normalized by object_base_size to match env.py:1814-1821; the
            # reward and the object_scales observation read these directly.
            x, y, z = _scale_to_3d(np.asarray(h_scale))
            entries.append({
                "type": dist.type,
                "shape": dist.shape,
                "distribution_index": dist_index,
                "sample_index": idx,
                "handle_scale": h_scale,
                "head_scale": head,
                "handle_density": h_d,
                "head_density": head_d,
                "scale_normalized": (
                    x / object_base_size, y / object_base_size, z / object_base_size,
                ),
            })

    # Shuffle so the per-type ordering doesn't bias env-i-gets-asset-i%N.
    # Matches legacy env.py:1824-1830. Disabled for debug-parity runs
    # (debug_differences/policy_rollout_isaacsim.py) where we want
    # pool[0] = first matching ObjectSizeDistribution.
    if shuffle:
        indices = np.arange(len(entries))
        np.random.shuffle(indices)
        permutation = [int(i) for i in indices]
    else:
        permutation = list(range(len(entries)))

    return entries, permutation


def pool_urdf_filename(entry: dict) -> str:
    """The URDF filename a pool entry is written to.

    Derived from the sampled parameters alone, so it is stable regardless of the
    order entries are written in.
    """
    return (
        f"{entry['sample_index']:03d}_{entry['type']}"
        f"_handle_{entry['handle_scale']}_head_{entry['head_scale']}"
        f"_d{entry['handle_density']:.1f}_{entry['head_density']}"
        .replace(".", "-")
        + ".urdf"
    )


def generate_handle_head_urdfs(
    handle_head_types: tuple[str, ...],
    num_per_type: int = _NUM_OBJECTS_PER_TYPE_DEFAULT,
    out_dir: Union[str, Path] = "/tmp/genmech_assets",
    object_base_size: float = _OBJECT_BASE_SIZE,
    seed: int = _SEED,
    shuffle: bool = True,
    density_scale: float = 1.0,
) -> tuple[list[str], list[tuple[float, float, float]], list[tuple]]:
    """Generate a pool of URDFs across the requested handle-head types.

    Mirrors legacy ``_handle_head_primitives`` at
    isaacgymenvs/tasks/simtoolreal/env.py:1706-1840:
      1. Seed NumPy with ``seed`` (deterministic pool per run).
      2. For each matching ``ObjectSizeDistribution``, sample ``num_per_type``
         (handle_scale, head_scale, handle_density, head_density) tuples.
      3. Emit one URDF per sample; return file paths paired with object scales
         normalized by ``object_base_size``.
      4. Shuffle (paths, scales) in lockstep so env ``i`` gets pool entry
         ``i % len(pool)`` with uniform coverage over types.

    Steps 1, 2 and the step-4 permutation live in :func:`sample_pool_params`;
    this function only writes the URDFs and applies the permutation.

    Returns
    -------
    urdf_paths : list[str]
        Absolute paths to generated URDFs. Length = ``num_per_type × len(matching distributions)``.
    object_scales_normalized : list[tuple[float, float, float]]
        Per-pool-entry 3-tuple scales, normalized by ``object_base_size`` (reward
        math uses these directly; see env.py:1814-1821).
    object_params : list[tuple]
        Per-pool-entry ``(handle_scale, head_scale, handle_density, head_density)``.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.suffix == ".urdf":
                p.unlink()
    else:
        os.makedirs(out_dir)

    entries, permutation = sample_pool_params(
        handle_head_types=handle_head_types,
        num_per_type=num_per_type,
        object_base_size=object_base_size,
        seed=seed,
        shuffle=shuffle,
        density_scale=density_scale,
    )

    paths: list[str] = []
    scales_norm: list[tuple[float, float, float]] = []
    params_raw: list[tuple] = []
    # Written in pre-shuffle order, exactly as before the split: the filename is
    # a pure function of the parameters, so write order does not affect the
    # result, but keeping it makes this a behaviour-preserving refactor.
    for entry in entries:
        urdf_path = out_dir / pool_urdf_filename(entry)
        generate_handle_head_urdf(
            path=urdf_path,
            handle_scale=entry["handle_scale"],
            head_scale=entry["head_scale"],
            handle_density=entry["handle_density"],
            head_density=entry["head_density"],
        )
        paths.append(str(urdf_path))
        scales_norm.append(entry["scale_normalized"])
        # The exact arguments this URDF was written from. Direct USD
        # authoring needs them, and recovering them by parsing the URDF back
        # would be a second source of truth that could drift from this one.
        params_raw.append((
            entry["handle_scale"], entry["head_scale"],
            entry["handle_density"], entry["head_density"],
        ))

    paths = [paths[i] for i in permutation]
    scales_norm = [scales_norm[i] for i in permutation]
    # params must ride the SAME permutation: env i takes pool entry i, and
    # a params list left in the original order would author a different
    # object than the one whose scale the reward and observations use.
    params_raw = [params_raw[i] for i in permutation]

    # Returned as a third element rather than a new function: any caller that
    # unpacks two values keeps working, and the params cannot drift out of step
    # with the paths because they are built in the same loop.
    return paths, scales_norm, params_raw


__all__ = [
    "generate_handle_urdf",
    "generate_handle_head_urdf",
    "generate_handle_head_urdfs",
    "matching_distributions",
    "sample_pool_params",
    "pool_urdf_filename",
]
