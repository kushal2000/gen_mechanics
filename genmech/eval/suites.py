"""Frozen held-out evaluation sets.

Each :class:`EvalCondition` is a named delta from the training distribution,
expressed as dotted config overrides. Conditions are literal Python and
committed, so a suite is a versioned artifact rather than something regenerated
per run.

Two rules make cross-hand comparison valid, and both are enforced by
``validate_suite``:

1. **No condition may contain a hand-specific field.** The same suite object is
   run for every robot; anything under ``assets.robot_spec`` or a spec field
   would make the comparison meaningless.
2. **Every randomized quantity resolves to a value or an explicit seed.** No
   condition may depend on wall-clock time, the run index, or an ambient RNG.

Every suite includes a ``nominal`` condition — the training distribution — so
each hand is scored against its own baseline. The headline metric is retention,
``goal_pct(condition) / goal_pct(nominal)``, which is invariant to hands having
different absolute performance (docs/methodology.md §4).

Importable without Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Goals are sampled LIVE, with exactly the sampler and parameters training used:
# a bounded random walk of 0.1 m / 90 deg steps inside the workspace box, with
# the first goal drawn uniformly at episode reset. This keeps the nominal
# condition a faithful replication of the training setup rather than a
# reconstruction of it.
#
# The cost is that the goal sequences are not identical across hands -- two
# robots do not consume the torch RNG identically. That is sampling noise, not
# bias: both hands draw from the same distribution, and it shrinks as
# 1/sqrt(num_envs). Run with enough envs that the reported SEM is small relative
# to the effect being measured, and compare `goal_pct_sem` across conditions
# before trusting a difference.
#
# genmech/eval/make_trajectory_pool.py can freeze a distribution into a
# replayable pool if a condition ever needs the variance eliminated; it is not
# used by default.
TRAIN_GOAL_MODE = "delta"
TRAIN_DELTA_DISTANCE = 0.1
TRAIN_DELTA_ROTATION_DEG = 90.0
TRAIN_VOLUME_MINS = (-0.35, -0.2, 0.6)
TRAIN_VOLUME_MAXS = (0.35, 0.2, 0.95)

# Goals an episode must reach before it ends.
GOALS_PER_TRAJECTORY = 10


@dataclass(frozen=True)
class EvalCondition:
    """One named evaluation condition."""

    axis: str
    """nominal | object_physics | object_geom | dr | goals"""
    name: str
    """Unique within the axis; ``axis/name`` identifies a condition."""
    overrides: dict[str, Any]
    """Dotted config paths -> values, applied to a fresh PoseReachEnvCfg."""
    seed: int
    """Frozen. Never derived from time or run index."""
    note: str = field(default="", compare=False)
    """What this condition is probing, and why it counts as held-out."""

    @property
    def id(self) -> str:
        return f"{self.axis}/{self.name}"


# ---------------------------------------------------------------------------
# The training distribution.
#
# Evaluation must pin these explicitly rather than inherit the configclass
# defaults: several DR knobs ship as no-ops -- object_friction_scale_range and
# object_scale_noise_multiplier_range both default to (1.0, 1.0) -- so "the
# defaults" and "what training actually randomized" are not the same statement.
# Anything a condition does not override keeps the config default.
# ---------------------------------------------------------------------------

_EVAL_PROTOCOL: dict[str, Any] = {
    # Deterministic initial state, so a condition's effect is not buried in
    # reset noise. This mirrors the DexToolBench eval setup.
    "reset.reset_position_noise_x": 0.0,
    "reset.reset_position_noise_y": 0.0,
    "reset.reset_position_noise_z": 0.0,
    "reset.reset_dof_pos_random_interval_arm": 0.0,
    "reset.reset_dof_pos_random_interval_fingers": 0.0,
    "reset.reset_dof_vel_random_interval": 0.0,
    "reset.table_reset_z_range": 0.0,
    # Live goal sampling, identical to training.
    "reset.fixed_trajectory_file": "",
    "reset.goal_sampling_type": TRAIN_GOAL_MODE,
    "reset.delta_goal_distance": TRAIN_DELTA_DISTANCE,
    "reset.delta_rotation_degrees": TRAIN_DELTA_ROTATION_DEG,
    "reset.target_volume_mins": TRAIN_VOLUME_MINS,
    "reset.target_volume_maxs": TRAIN_VOLUME_MAXS,
    # Pin the success criterion. The tolerance curriculum ends wherever a run
    # happened to reach, so scoring at the live tolerance would let a run that
    # trained further look better for free.
    "termination.eval_success_tolerance": 0.01,
    "termination.success_steps": 1,
    "termination.max_consecutive_successes": GOALS_PER_TRAJECTORY,
}

# DR profiles. `train` reproduces what the training runs randomize; `off` is the
# clean-room control; `hard` is the held-out robustness condition.
DR_PROFILES: dict[str, dict[str, Any]] = {
    "off": {
        "domain_randomization.use_obs_delay": False,
        "domain_randomization.use_action_delay": False,
        "domain_randomization.use_object_state_delay_noise": False,
        "domain_randomization.joint_velocity_obs_noise_std": 0.0,
        "domain_randomization.force_scale": 0.0,
        "domain_randomization.torque_scale": 0.0,
    },
    "train": {
        "domain_randomization.use_obs_delay": True,
        "domain_randomization.obs_delay_max": 3,
        "domain_randomization.use_action_delay": True,
        "domain_randomization.action_delay_max": 3,
        "domain_randomization.use_object_state_delay_noise": True,
        "domain_randomization.object_state_delay_max": 10,
        "domain_randomization.object_state_xyz_noise_std": 0.01,
        "domain_randomization.object_state_rotation_noise_degrees": 5.0,
        "domain_randomization.joint_velocity_obs_noise_std": 0.1,
        "domain_randomization.force_scale": 20.0,
        "domain_randomization.torque_scale": 2.0,
    },
    "hard": {
        "domain_randomization.use_obs_delay": True,
        "domain_randomization.obs_delay_max": 6,
        "domain_randomization.use_action_delay": True,
        "domain_randomization.action_delay_max": 6,
        "domain_randomization.use_object_state_delay_noise": True,
        "domain_randomization.object_state_delay_max": 20,
        "domain_randomization.object_state_xyz_noise_std": 0.02,
        "domain_randomization.object_state_rotation_noise_degrees": 10.0,
        "domain_randomization.joint_velocity_obs_noise_std": 0.2,
        "domain_randomization.force_scale": 40.0,
        "domain_randomization.torque_scale": 4.0,
    },
}

# The six procedural object families. Training uses all six; the object-geometry
# axis holds one out at a time.
ALL_CATEGORIES = ("hammer", "screwdriver", "marker", "spatula", "eraser", "brush")

SEED = 20260811  # one frozen seed for the whole suite


def _cond(axis: str, name: str, note: str, **overrides: Any) -> EvalCondition:
    return EvalCondition(axis=axis, name=name, overrides=dict(overrides),
                         seed=SEED, note=note)


NOMINAL = EvalCondition(
    axis="nominal",
    name="nominal",
    overrides={**DR_PROFILES["train"]},
    seed=SEED,
    note=(
        "The training distribution. Every other condition is scored relative to "
        "this one, per hand."
    ),
)


OBJECT_PHYSICS: list[EvalCondition] = [
    # Mass, via baked density. Runtime mass randomization is blocked in this
    # Isaac Lab build (set_masses raises), but density is baked into the
    # generated URDF, so the held-out mass condition needs no runtime API.
    _cond("object_physics", "density_0.5x",
          "Half-mass objects. Same geometry, same pool seed -- only inertia changes.",
          **DR_PROFILES["train"], **{"assets.object_density_scale": 0.5}),
    _cond("object_physics", "density_2x",
          "Double-mass objects; the policy must hold more inertia with the same grip.",
          **DR_PROFILES["train"], **{"assets.object_density_scale": 2.0}),
    _cond("object_physics", "density_4x",
          "Quadruple mass. Expected to be hard for both hands; included to see "
          "which degrades first.",
          **DR_PROFILES["train"], **{"assets.object_density_scale": 4.0}),
    # Friction. Applied through the init-only bucketed path; a degenerate range
    # sets one fixed value for every env.
    _cond("object_physics", "friction_0.25x",
          "Slippery objects -- the grasp-stability stress case.",
          **DR_PROFILES["train"],
          **{"domain_randomization.object_friction_scale_range": (0.25, 0.25)}),
    _cond("object_physics", "friction_0.5x", "Moderately slippery objects.",
          **DR_PROFILES["train"],
          **{"domain_randomization.object_friction_scale_range": (0.5, 0.5)}),
    _cond("object_physics", "friction_2x", "Grippier objects.",
          **DR_PROFILES["train"],
          **{"domain_randomization.object_friction_scale_range": (2.0, 2.0)}),
    # Restitution. Training is fully inelastic, so any value here is held out.
    _cond("object_physics", "restitution_0.3", "Bouncy contacts.",
          **DR_PROFILES["train"], **{"assets.object_restitution": 0.3}),
    _cond("object_physics", "restitution_0.6", "Very bouncy contacts.",
          **DR_PROFILES["train"], **{"assets.object_restitution": 0.6}),
]


OBJECT_GEOM: list[EvalCondition] = [
    # Unseen draws from the *training* size distributions. This isolates
    # "memorized this pool" from "learned the family".
    _cond("object_geom", "unseen_pool_seed",
          "Fresh draws from the same size distributions. In-distribution but "
          "unseen instances; separates memorization from generalization.",
          **DR_PROFILES["train"], **{"assets.object_seed": 20260811}),
] + [
    # One held-out family at a time. Training must use the complementary five
    # for this to be a genuine held-out set -- see note.
    _cond("object_geom", f"heldout_category_{cat}",
          f"Only {cat} objects. A true held-out set ONLY for a policy trained "
          f"on the other five (assets.handle_head_types). Against a policy "
          f"trained on all six this is a per-family breakdown, not a "
          f"generalization test -- report it as such.",
          **DR_PROFILES["train"],
          **{"assets.handle_head_types": (cat,), "assets.object_seed": 20260811})
    for cat in ALL_CATEGORIES
]


GOALS: list[EvalCondition] = [
    # Training's goal distribution is a bounded random walk (0.1 m / 90 deg
    # steps). Held-out means a wider walk, no walk at all, or a shifted volume.
    _cond("goals", "delta_2x",
          "Goal-to-goal steps twice as far and twice as much rotation. Requires "
          "in-hand regrasping the training walk rarely demanded.",
          **DR_PROFILES["train"],
          **{"reset.delta_goal_distance": 0.2,
             "reset.delta_rotation_degrees": 180.0}),
    _cond("goals", "absolute",
          "Goals drawn independently across the whole workspace instead of as a "
          "random walk, so consecutive goals can be maximally far apart.",
          **DR_PROFILES["train"],
          **{"reset.goal_sampling_type": "absolute"}),
    _cond("goals", "volume_high",
          "Goal volume shifted 15 cm up, out of the trained z range [0.6, 0.95].",
          **DR_PROFILES["train"],
          **{"reset.goal_sampling_type": "absolute",
             "reset.target_volume_mins": (-0.35, -0.2, 0.75),
             "reset.target_volume_maxs": (0.35, 0.2, 1.10)}),
]


DR_AXIS: list[EvalCondition] = [
    _cond("dr", "off",
          "No perturbations. Upper bound on nominal competence; the gap to "
          "nominal measures how much DR itself costs.",
          **DR_PROFILES["off"]),
    _cond("dr", "hard",
          "Every DR magnitude doubled -- latency, sensor noise, and disturbance "
          "wrenches. Measures robustness margin beyond what training exercised.",
          **DR_PROFILES["hard"]),
]


def full_suite() -> list[EvalCondition]:
    """Every condition, nominal first."""
    return [NOMINAL, *DR_AXIS, *OBJECT_PHYSICS, *OBJECT_GEOM, *GOALS]


def smoke_suite() -> list[EvalCondition]:
    """A four-condition subset for validating the pipeline end to end."""
    by_id = {c.id: c for c in full_suite()}
    return [
        NOMINAL,
        by_id["dr/off"],
        by_id["object_physics/density_2x"],
        by_id["goals/delta_2x"],
    ]


SUITES = {"full": full_suite, "smoke": smoke_suite}


def get_suite(name: str) -> list[EvalCondition]:
    try:
        conditions = SUITES[name]()
    except KeyError:
        raise KeyError(f"unknown suite {name!r}; available: {sorted(SUITES)}") from None
    validate_suite(conditions)
    return conditions


def validate_suite(conditions: list[EvalCondition]) -> None:
    """Reject a suite that cannot support a fair cross-hand comparison."""
    seen = set()
    for c in conditions:
        if c.id in seen:
            raise ValueError(f"duplicate condition id {c.id!r}")
        seen.add(c.id)

        # Rule 1: nothing hand-specific. The same suite runs for every robot.
        for key in c.overrides:
            if "robot" in key:
                raise ValueError(
                    f"{c.id}: override {key!r} is robot-specific. Conditions must "
                    f"be identical across hands or the comparison is void."
                )

    if not any(c.axis == "nominal" for c in conditions):
        raise ValueError(
            "suite has no 'nominal' condition; retention is undefined without a "
            "per-hand baseline"
        )


def resolve_overrides(condition: EvalCondition) -> dict[str, Any]:
    """Protocol settings plus this condition's overrides (condition wins)."""
    return {**_EVAL_PROTOCOL, **condition.overrides}


__all__ = [
    "EvalCondition",
    "NOMINAL",
    "DR_PROFILES",
    "ALL_CATEGORIES",
    "full_suite",
    "smoke_suite",
    "get_suite",
    "validate_suite",
    "resolve_overrides",
    "SEED",
    "GOALS_PER_TRAJECTORY",
]
