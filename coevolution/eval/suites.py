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
# coevolution/eval/make_trajectory_pool.py can freeze a distribution into a
# replayable pool if a condition ever needs the variance eliminated; it is not
# used by default.
TRAIN_GOAL_MODE = "delta"
TRAIN_DELTA_DISTANCE = 0.1
TRAIN_DELTA_ROTATION_DEG = 90.0
TRAIN_VOLUME_MINS = (-0.35, -0.2, 0.6)
TRAIN_VOLUME_MAXS = (0.35, 0.2, 0.95)

# Goals an episode must reach before it ends. Training uses 50; the eval uses a
# shorter chain so completion rate stays off the floor and can discriminate.
# Calibrate this once against a reference policy so nominal lands near 50%
# completion -- maximum sensitivity to shifts in either direction -- then freeze
# it for every hand.
GOALS_PER_TRAJECTORY = 10

# Consecutive steps within tolerance that count as reaching a goal. Matches
# training (TerminationCfg.success_steps).
TRAIN_SUCCESS_STEPS = 10


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
    # Reset noise zeroed -- but note this does NOT give a deterministic initial
    # state, and it would be wrong to describe it that way.
    #
    # The object still spawns at a uniformly random orientation
    # (reset_utils calls random_orientation unconditionally; it is not governed
    # by any noise scale) and free-falls table_object_z_offset = 0.25 m before
    # settling. Every episode therefore begins with a random tumble whose
    # outcome depends on the object's shape.
    #
    # That variation is deliberate: it is closer to deployment than a pinned
    # pose, and a hand that copes better with awkward settles is genuinely
    # better hardware. Both hands draw from the same distribution, so it is
    # noise in the comparison rather than bias -- but it is a large source of
    # per-episode variance, and it plausibly drives part of the bimodal outcome
    # (an object that settles ungraspably yields a zero-goal episode for reasons
    # unrelated to the policy). Run enough envs that the reported SEM is small,
    # and do not read a small cross-hand difference as real.
    #
    # To pin it for a diagnostic, override reset.fixed_start_pose.
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
    # Pin the tolerance. The curriculum ends wherever a run happened to reach,
    # so scoring at the live tolerance would let a run that trained further look
    # better for free.
    "termination.eval_success_tolerance": 0.01,
    # Dwell requirement, matching training. This was 1 (instant credit on first
    # touch) for sweep throughput, which was a mistake: holding a pose steady for
    # 10 steps plausibly favours hands with more stable multi-contact grasps, so
    # removing it risks suppressing exactly the hardware difference this project
    # exists to measure. A protocol no policy trains under is not a neutral
    # measuring stick. Costs roughly 4x per condition; worth it.
    "termination.success_steps": TRAIN_SUCCESS_STEPS,
    # Chain length stays short. Unlike dwell, this is just "how many goals we
    # ask for" and is unlikely to interact with hand morphology, and the full
    # training value of 50 puts completion near the floor (5.5% for the
    # reference checkpoint), where it cannot discriminate between conditions.
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


# ---------------------------------------------------------------------------
# SINGLE-KNOB DR CONDITIONS (24k population study, 2026-08-27)
#
# The three DR_PROFILES above move seven-plus knobs at once, so they can say
# THAT a design is fragile but never TO WHAT. These conditions move exactly one
# knob each, which is what an interaction question ("do some designs generalize
# better?") actually needs.
#
# EVERY DR FIELD IS PINNED EXPLICITLY. RUN_FIELDS carries no
# domain_randomization.*, so anything a condition leaves unset silently takes
# the configclass default rather than the training run's value -- and the
# defaults are NOT no-ops: object_state_xyz_noise_std defaults to 0.01,
# force_prob_range to (0.001, 0.1). Enabling a flag without setting its group
# would therefore import magnitudes nobody chose.
#
# EVERY RANGE IS DEGENERATE. Each knob is sampled at a different cadence --
# friction once at scene init (scene_utils.py:1727), the wrench probability once
# per episode (reset_utils.py:530), object-state noise and the action-delay
# index every step (obs_utils.py:260, action_utils.py:44). A degenerate (x, x)
# makes all of them deterministic and identical across envs, so no design's
# score carries its own DR lottery and the cadence stops mattering. That is the
# difference between measuring morphology and measuring luck.
# ---------------------------------------------------------------------------

# The population runs' own DR: everything off. Behaviourally identical to
# DR_PROFILES["off"], but written out in full so no field is left to a default.
_DR_OFF_EXPLICIT: dict[str, Any] = {
    "domain_randomization.use_obs_delay": False,
    "domain_randomization.obs_delay_max": 3,
    "domain_randomization.use_action_delay": False,
    "domain_randomization.action_delay_max": 3,
    "domain_randomization.use_object_state_delay_noise": False,
    "domain_randomization.object_state_delay_max": 10,
    "domain_randomization.object_state_xyz_noise_std": 0.0,
    "domain_randomization.object_state_rotation_noise_degrees": 0.0,
    "domain_randomization.object_scale_noise_multiplier_range": (1.0, 1.0),
    "domain_randomization.joint_velocity_obs_noise_std": 0.0,
    "domain_randomization.force_scale": 0.0,
    "domain_randomization.force_prob_range": (0.0, 0.0),
    "domain_randomization.force_decay": 0.0,
    "domain_randomization.force_decay_interval": 0.08,
    "domain_randomization.force_only_when_lifted": True,
    "domain_randomization.torque_scale": 0.0,
    "domain_randomization.torque_prob_range": (0.0, 0.0),
    "domain_randomization.torque_decay": 0.0,
    "domain_randomization.torque_decay_interval": 0.08,
    "domain_randomization.torque_only_when_lifted": True,
    "domain_randomization.object_friction_scale_range": (1.0, 1.0),
    "domain_randomization.fingertip_friction_scale_range": (1.0, 1.0),
    "domain_randomization.friction_n_buckets": 16,
}

# The friction ranges are MULTIPLIERS on AssetsCfg base values, not absolute
# frictions. The population runs use finger_tip_friction=1.5 and
# object_friction=0.5, so these scales resolve to the absolute mu named.
# scene_utils.py builds `linspace(lo, hi, n_buckets) * base`, so a degenerate
# range gives every env exactly `lo * base`.
POP_FINGERTIP_FRICTION_BASE = 1.5
POP_OBJECT_FRICTION_BASE = 0.5
_FINGERTIP_FRICTION_TARGET = 1.0   # mu 1.5 -> 1.0, a less sticky robot
_OBJECT_FRICTION_TARGET = 0.3      # mu 0.5 -> 0.3, a less sticky object
_FT_SCALE = _FINGERTIP_FRICTION_TARGET / POP_FINGERTIP_FRICTION_BASE
_OBJ_SCALE = _OBJECT_FRICTION_TARGET / POP_OBJECT_FRICTION_BASE

# Wrench magnitudes are DR_PROFILES["train"]'s. Only the probability differs:
# train leaves it at the (0.001, 0.1) default, drawn log-uniform PER EPISODE
# across a 100x span, so under that profile a design's exposure is mostly its
# own draw. Pinned here so every env is struck at the same rate.
_WRENCH_PROB = 0.01

DR_KNOBS: dict[str, dict[str, Any]] = {
    "fingertip_friction": {
        "domain_randomization.fingertip_friction_scale_range": (_FT_SCALE, _FT_SCALE),
    },
    "object_friction": {
        "domain_randomization.object_friction_scale_range": (_OBJ_SCALE, _OBJ_SCALE),
    },
    "wrench": {
        "domain_randomization.force_scale": 20.0,
        "domain_randomization.force_prob_range": (_WRENCH_PROB, _WRENCH_PROB),
        "domain_randomization.torque_scale": 2.0,
        "domain_randomization.torque_prob_range": (_WRENCH_PROB, _WRENCH_PROB),
    },
    "object_state_noise": {
        "domain_randomization.use_object_state_delay_noise": True,
        "domain_randomization.object_state_delay_max": 10,
        "domain_randomization.object_state_xyz_noise_std": 0.01,
        "domain_randomization.object_state_rotation_noise_degrees": 5.0,
    },
    "action_delay": {
        "domain_randomization.use_action_delay": True,
        "domain_randomization.action_delay_max": 3,
    },
}

_DR_KNOB_NOTES: dict[str, str] = {
    "fingertip_friction":
        "Fingertip mu 1.5 -> 1.0 on every env. A less sticky robot. Tests whether "
        "a design's grasp survives on geometry rather than on skin friction.",
    "object_friction":
        "Object mu 0.5 -> 0.3 on every env. A less sticky object. The force-closure "
        "stress case: predicts the mount-separation optimum sharpens.",
    "wrench":
        "Disturbance force 20 N / torque 2 Nm on the object, applied only once "
        "lifted, at a pinned 1% per-step probability. Tests grasp RETENTION, not "
        "acquisition; predicts finger count and separation buy moment arm.",
    "object_state_noise":
        "Observed object pose delayed up to 10 steps and corrupted by 1 cm / 5 deg. "
        "A perception-error proxy; predicts tight pinches suffer more than wraps.",
    "action_delay":
        "Actions drawn uniformly from the last 3 policy steps, re-drawn every step "
        "(~0-50 ms at 60 Hz). Tests tolerance to actuation latency.",
}

# Nominal for THIS axis is DR-off, not DR_PROFILES["train"]. The 24k population
# runs trained with every DR knob disabled, so the suite's NOMINAL would itself
# be a held-out shift for them and retention against it would be meaningless.
NOMINAL_DR_OFF = EvalCondition(
    axis="nominal", name="dr_off", overrides=dict(_DR_OFF_EXPLICIT), seed=SEED,
    note=("The 24k population runs' own training distribution: all DR disabled. "
          "The baseline every dr_knob condition is scored against."),
)

DR_KNOB_AXIS: list[EvalCondition] = [
    EvalCondition(axis="dr_knob", name=name,
                  overrides={**_DR_OFF_EXPLICIT, **knob},
                  seed=SEED, note=_DR_KNOB_NOTES[name])
    for name, knob in DR_KNOBS.items()
] + [
    # All five at once. Not a substitute for the singles: it says whether the
    # damage compounds, and the singles say what it is made of.
    EvalCondition(
        axis="dr_knob", name="all_knobs",
        overrides={**_DR_OFF_EXPLICIT,
                   **{k: v for knob in DR_KNOBS.values() for k, v in knob.items()}},
        seed=SEED,
        note="All five single knobs simultaneously, at the same magnitudes.",
    )
]


# ---------------------------------------------------------------------------
# MAGNITUDE SWEEP (2026-08-27, second round)
#
# Round 1 measured: the three contact/physics knobs moved performance by ~1% and
# did not re-rank designs at all (disattenuated r vs nominal 0.965-0.977), while
# object-state noise took the population to -78% with 9.6% of designs at zero.
# Both ends are useless as a fitness signal. Target band, taking action_delay
# (-41%, 0.8% zeros, reliability 0.57) as the reference: a 30-50% mean drop with
# under 2% of designs at zero.
#
# WHY THE PHYSICS KNOBS READ NULL. Nothing in the repo authors a friction
# combine mode, so PhysX's default -- average -- applies, and effective
# fingertip-object friction is (mu_ft + mu_obj)/2 = (1.5 + 0.5)/2 = 1.0. Round
# 1 therefore moved effective friction by 25% (fingertip) and 10% (object), not
# the 33% and 40% the scales suggest. Scaling BOTH surfaces by k gives
# mu_eff = k exactly, which is why the ladder below is expressed that way.
#
# Note object friction alone is capped: even at mu_obj = 0 it only reaches
# mu_eff = 0.75. There is no magnitude that makes it a strong knob, which is why
# it is not swept on its own.
#
# WHY THE WRENCH LADDER MOVES PROBABILITY, NOT SCALE. force_scale is an
# ACCELERATION -- action_utils computes `randn * mass * force_scale`, so mass
# cancels -- and force_decay = 0 zeroes the wrench every step, so each fire is a
# one-step impulse of force_scale/60 ~ 0.33 m/s per axis. Fires are independent,
# so sustained disturbance amplitude grows as sqrt(rate): 0.01 -> 1.0 is 100x the
# events and ~10x the amplitude. At prob 1.0 the perturbation stops being
# discrete kicks and becomes white-noise buffeting -- same energy scale,
# different phenomenon, and the ceiling of the knob.
# ---------------------------------------------------------------------------

# delay_max = 0 gives a length-1 queue (allocated as max(1, delay_max)), so
# _sample_delay's randint(0, 1) always returns the value pushed this step: zero
# delay, noise intact. The only gate on the block is use_object_state_delay_noise,
# so the flag must stay True or the noise goes too.
_OBJ_NOISE_NO_DELAY = {
    "domain_randomization.use_object_state_delay_noise": True,
    "domain_randomization.object_state_delay_max": 0,
}


def _friction_k(k: float) -> dict[str, Any]:
    """Scale BOTH contact surfaces by k, so mu_eff = k under average combine."""
    return {
        "domain_randomization.fingertip_friction_scale_range": (k, k),
        "domain_randomization.object_friction_scale_range": (k, k),
    }


def _wrench_at(prob: float) -> dict[str, Any]:
    """Round 1's magnitudes; only the fire rate moves."""
    return {
        "domain_randomization.force_scale": 20.0,
        "domain_randomization.force_prob_range": (prob, prob),
        "domain_randomization.torque_scale": 2.0,
        "domain_randomization.torque_prob_range": (prob, prob),
    }


DR_SWEEP: dict[str, tuple[dict[str, Any], str]] = {
    "obj_noise_1cm": (
        {**_OBJ_NOISE_NO_DELAY,
         "domain_randomization.object_state_xyz_noise_std": 0.01,
         "domain_randomization.object_state_rotation_noise_degrees": 5.0},
        "Round 1's noise magnitude with the 10-step (167 ms) delay removed. "
        "Perturbs observed object POSE only -- velocities are read from the "
        "delayed state and never get noise added.",
    ),
    "obj_noise_0.5cm": (
        {**_OBJ_NOISE_NO_DELAY,
         "domain_randomization.object_state_xyz_noise_std": 0.005,
         "domain_randomization.object_state_rotation_noise_degrees": 2.5},
        "Half magnitude. Brackets the target band from below, since removing "
        "the delay could leave the damage anywhere from -70% to -20%.",
    ),
    "friction_k0.5":   (_friction_k(0.5),   "mu_ft 0.75, mu_obj 0.25 -> mu_eff 0.50."),
    "friction_k0.25":  (_friction_k(0.25),  "mu_ft 0.375, mu_obj 0.125 -> mu_eff 0.25."),
    "friction_k0.125": (_friction_k(0.125), "mu_ft 0.1875, mu_obj 0.0625 -> mu_eff 0.125."),
    "friction_ft_zero": (
        {"domain_randomization.fingertip_friction_scale_range": (0.0, 0.0),
         "domain_randomization.object_friction_scale_range": (1.0, 1.0)},
        "Frictionless fingertips, object untouched: mu_eff = (0 + 0.5)/2 = 0.25, "
        "the SAME effective friction as friction_k0.25 but split differently "
        "between the surfaces. If the two score alike the average-combine model "
        "holds and friction is one knob; if not, the combine mode is not average "
        "and every friction number needs rereading.",
    ),
    "wrench_p0.025": (_wrench_at(0.025),
                      "1.5 fires/s, ~1.6x round 1's disturbance amplitude. Fills "
                      "the gap between the measured null at 0.01 and 0.1."),
    "wrench_p0.05": (_wrench_at(0.05), "3 fires/s, ~2.2x."),
    "wrench_p0.1": (_wrench_at(0.1), "6 fires/s, ~3.2x round 1's disturbance amplitude."),
    "wrench_p0.3": (_wrench_at(0.3), "18 fires/s, ~5.5x."),
    "wrench_p1.0": (_wrench_at(1.0),
                    "A fresh wrench every step: 60 fires/s, ~10x, and the "
                    "ceiling of this knob at these scales. If this does not "
                    "reach the target band, the wrench axis is dead at "
                    "force_scale 20 and no intermediate point matters."),
}

DR_SWEEP_AXIS: list[EvalCondition] = [
    EvalCondition(axis="dr_sweep", name=name,
                  overrides={**_DR_OFF_EXPLICIT, **knob}, seed=SEED, note=note)
    for name, (knob, note) in DR_SWEEP.items()
]


def dr_sweep_suite() -> list[EvalCondition]:
    """DR-off baseline plus the round-2 magnitude sweep."""
    return [NOMINAL_DR_OFF, *DR_SWEEP_AXIS]


def dr_knobs_suite() -> list[EvalCondition]:
    """DR-off baseline plus one condition per knob, plus all knobs together."""
    return [NOMINAL_DR_OFF, *DR_KNOB_AXIS]


def condition_by_id(cond_id: str) -> EvalCondition:
    """Look up any committed condition by ``axis/name``."""
    known = {c.id: c for c in (*full_suite(), *dr_knobs_suite(), *dr_sweep_suite())}
    try:
        return known[cond_id]
    except KeyError:
        raise KeyError(
            f"unknown condition {cond_id!r}; available: {sorted(known)}") from None


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


SUITES = {"full": full_suite, "smoke": smoke_suite, "dr_knobs": dr_knobs_suite,
          "dr_sweep": dr_sweep_suite}


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
    "dr_knobs_suite",
    "dr_sweep_suite",
    "DR_SWEEP",
    "condition_by_id",
    "DR_KNOBS",
    "NOMINAL_DR_OFF",
    "get_suite",
    "validate_suite",
    "resolve_overrides",
    "SEED",
    "GOALS_PER_TRAJECTORY",
]
