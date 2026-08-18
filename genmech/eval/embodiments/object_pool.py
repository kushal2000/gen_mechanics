"""Reconstruct a training run's object pool without booting Isaac Sim.

A run's object pool is never written anywhere durable: ``setup_scene`` builds it
into ``tempfile.mkdtemp(prefix="genmech_assets_")`` and it dies with the process
(``genmech/tasks/pose_reach/utils/scene_utils.py``). So after the fact there is
no file that says which object environment 4,211 was holding.

It is recoverable anyway, because the pool is a pure function of five config
fields. Everything that consumes randomness on that path lives in
``generate_objects.sample_pool_params`` -- one ``np.random.seed(object_seed)``,
four ``np.random.uniform`` draws per distribution, one final shuffle -- and
writing a URDF draws nothing. Replaying that function with a run's
``(handle_head_types, num_assets_per_type, object_seed, shuffle_assets,
object_density_scale)`` therefore reproduces that run's pool exactly, in order.

This module calls the real sampler rather than reimplementing it. That matters
more than it looks: a reimplementation that drifts does not fail, it silently
returns a plausible pool of the right size describing the wrong run.

**Pool size is not ``num_assets_per_type * len(types)``.** Six handle-head types
map to twelve ``ObjectSizeDistribution`` entries, because several types have
shape variants (hammer and screwdriver have cuboid and cylinder handles, brush
has two head shapes). With all six types, ``num_assets_per_type=100`` gives 1,200
objects, not 600. Use :func:`expected_pool_size` rather than multiplying by hand.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from genmech.tasks.pose_reach.utils.generate_objects import (
    matching_distributions,
    pool_urdf_filename,
    sample_pool_params,
)

# generate_objects' own defaults, repeated here so a caller reconstructing a run
# that did not override them does not have to reach into a private name.
DEFAULT_OBJECT_BASE_SIZE = 0.04
DEFAULT_OBJECT_SEED = 42
DEFAULT_NUM_ASSETS_PER_TYPE = 100

ALL_CATEGORIES = ("hammer", "screwdriver", "marker", "spatula", "eraser", "brush")


@dataclass(frozen=True)
class PoolEntry:
    """One object in a pool, at its final (post-shuffle) index.

    ``index`` is the number that matters: environment ``i`` holds pool entry
    ``i % pool_size``, so this is the identifier an assignment record points at.
    """

    index: int
    type: str
    shape: str  # "cuboid" | "cylinder"
    distribution_index: int
    sample_index: int
    handle_scale: tuple[float, ...]
    head_scale: Optional[tuple[float, ...]]
    handle_density: float
    head_density: Optional[float]
    scale_normalized: tuple[float, float, float]
    urdf_filename: str

    @property
    def has_head(self) -> bool:
        return self.head_scale is not None

    def label(self) -> str:
        """A short one-line description, for viewer dropdowns and log lines."""
        dims = " x ".join(f"{100 * v:.1f}" for v in self.handle_scale)
        head = "" if self.head_scale is None else (
            " + head " + " x ".join(f"{100 * v:.1f}" for v in self.head_scale)
        )
        return f"#{self.index} {self.type} ({self.shape}) handle {dims} cm{head}"

    def to_json(self) -> dict:
        d = asdict(self)
        # JSON has no tuples; be explicit rather than relying on the encoder.
        d["handle_scale"] = list(self.handle_scale)
        d["head_scale"] = None if self.head_scale is None else list(self.head_scale)
        d["scale_normalized"] = list(self.scale_normalized)
        return d


def expected_pool_size(handle_head_types: Sequence[str], num_assets_per_type: int) -> int:
    """Pool size for these types, without running the sampler.

    Cheap enough to assert against an env count before booting anything.
    """
    return num_assets_per_type * len(matching_distributions(tuple(handle_head_types)))


def reconstruct_pool(
    handle_head_types: Sequence[str],
    num_assets_per_type: int = DEFAULT_NUM_ASSETS_PER_TYPE,
    object_seed: int = DEFAULT_OBJECT_SEED,
    shuffle: bool = True,
    density_scale: float = 1.0,
    object_base_size: float = DEFAULT_OBJECT_BASE_SIZE,
) -> list[PoolEntry]:
    """The pool a run with these settings built, in final pool order.

    Argument names follow the *config* fields (``num_assets_per_type``,
    ``object_seed``) rather than the generator's shorter parameter names, since
    every caller here is reading them out of a ``.hydra/config.yaml``.
    """
    entries, permutation = sample_pool_params(
        handle_head_types=tuple(handle_head_types),
        num_per_type=num_assets_per_type,
        object_base_size=object_base_size,
        seed=object_seed,
        shuffle=shuffle,
        density_scale=density_scale,
    )
    pool: list[PoolEntry] = []
    for final_index, pre_shuffle_index in enumerate(permutation):
        e = entries[pre_shuffle_index]
        pool.append(PoolEntry(
            index=final_index,
            type=e["type"],
            shape=e["shape"],
            distribution_index=e["distribution_index"],
            sample_index=e["sample_index"],
            handle_scale=e["handle_scale"],
            head_scale=e["head_scale"],
            handle_density=e["handle_density"],
            head_density=e["head_density"],
            scale_normalized=e["scale_normalized"],
            urdf_filename=pool_urdf_filename(e),
        ))
    return pool


def type_counts(pool: Sequence[PoolEntry]) -> dict[str, int]:
    """How many pool entries each handle-head type contributes.

    Not uniform across types: with all six categories, brush gets four
    distributions and eraser one, so a "balanced" pool is 4x brush-heavy.
    """
    counts: dict[str, int] = {}
    for entry in pool:
        counts[entry.type] = counts.get(entry.type, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def pool_index_for_env(env_id: int, pool_size: int) -> int:
    """Which pool entry environment ``env_id`` holds.

    Mirrors ``scene_utils._author_objects_into_envs`` (``_authored_asset_index[i]
    = i % n_pool``) and the converted path's ``source_idx % num_object_usds``.
    Both paths agree; if that ever stops being true, this is the single place to
    fix, and ``dump_assignment --sanity_check`` is what will notice.
    """
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    return env_id % pool_size


def design_index_for_env(env_id: int, population_count: int) -> int:
    """Which population design environment ``env_id`` holds.

    Mirrors ``scene_utils._build_robot_design_tensor``. When
    ``population_count == num_envs`` -- the k = n operating point the population
    runs use -- this is the identity.
    """
    if population_count <= 0:
        raise ValueError(f"population_count must be positive, got {population_count}")
    return env_id % population_count


__all__ = [
    "ALL_CATEGORIES",
    "DEFAULT_NUM_ASSETS_PER_TYPE",
    "DEFAULT_OBJECT_BASE_SIZE",
    "DEFAULT_OBJECT_SEED",
    "PoolEntry",
    "design_index_for_env",
    "expected_pool_size",
    "pool_index_for_env",
    "reconstruct_pool",
    "type_counts",
]
