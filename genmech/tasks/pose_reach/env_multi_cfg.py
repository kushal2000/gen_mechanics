"""Config for the multi-embodiment PoseReach env.

Subclasses the single-embodiment config rather than copying it: the task,
reward, reset, DR and object pipeline are identical, and the two must not drift.
Only what genuinely differs is declared here -- which hands, and the morphology
observation they require.
"""

from __future__ import annotations

from dataclasses import field

from isaaclab.utils import configclass

from .env_cfg import AssetsCfg, ObsCfg, PoseReachEnvCfg
from .utils.morphology import DESCRIPTOR_DIM  # noqa: F401  (documented width)

# The single-embodiment defaults, to append to rather than restate. Restating
# them would let the two observation layouts drift silently.
_BASE_OBS = ObsCfg()


@configclass
class MultiAssetsCfg(AssetsCfg):
    """Assets, plus the hand population."""

    # Which cached population to draw from
    # (genmech.robots.generated.population.build_population).
    #
    # Every env gets its own design, round-robin: env i holds design
    # i % len(population), the same rule the object pool uses. All of them run in
    # ONE articulation view because ghosting pads each design to the same
    # 37-joint template -- view count tracks joint COUNT, not population size.
    robot_population_seed: int | None = 0
    robot_population_count: int = 0
    """0 means the whole cached population for that seed."""

    # robot_spec is inherited but NOT used to build the scene: the population
    # supplies the specs. It stays for tooling that expects the field.


@configclass
class MultiObsCfg(ObsCfg):
    """Observation lists, plus the morphology descriptor.

    ``fingertip_pos_rel_palm`` and every other field keep their meaning and
    position; ``morphology`` is appended. It is constant per env, so it costs
    nothing to recompute and gives the policy the mechanism it is commanding --
    mount poses, link lengths and radii, which joints exist, and the abduction
    range that joint_pos normalization otherwise hides.
    """

    # Read off an INSTANCE, not the class: @configclass turns these into
    # dataclass fields, so ObsCfg.state_list is not a plain class attribute and
    # raises AttributeError at import time.
    state_list: tuple[str, ...] = _BASE_OBS.state_list + ("morphology",)
    obs_list: tuple[str, ...] = _BASE_OBS.obs_list + ("morphology",)


@configclass
class PoseReachMultiEnvCfg(PoseReachEnvCfg):
    assets: MultiAssetsCfg = field(default_factory=MultiAssetsCfg)
    obs: MultiObsCfg = field(default_factory=MultiObsCfg)

    log_morphology_layout: bool = False
    """Print the descriptor's field map at startup. Useful when debugging an
    observation vector; noisy otherwise."""


__all__ = ["PoseReachMultiEnvCfg", "MultiAssetsCfg", "MultiObsCfg"]
