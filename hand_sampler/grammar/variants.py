"""Named ``Distribution`` variants and operator mixtures for the E0
evolvability experiments. Every variant below is built from EXISTING
``Distribution`` fields (no new field was needed for these six), via
``dataclasses.replace`` on ``DEFAULT_DISTRIBUTION`` -- so each differs from
``G_FULL`` in exactly the field(s) named in its own docstring, everything
else identical to current default sampling behaviour.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import Tuple

from .derive import OPERATORS, SMALL_STEP_OPERATORS
from .distributions import DEFAULT_DISTRIBUTION, Distribution

# G_FULL is intentionally the *same object* as DEFAULT_DISTRIBUTION (not a
# copy): every other module's DEFAULT_DISTRIBUTION-keyed behaviour (replay
# hashes, existing tests) is automatically also "G_FULL" behaviour.
G_FULL: Distribution = DEFAULT_DISTRIBUTION

# No palm joints: every additional palm body is a fixed joint.
G_NOPALMJOINT: Distribution = replace(DEFAULT_DISTRIBUTION, palm_joint_probability=0.0)

# No in-digit branching.
G_NOBRANCH: Distribution = replace(DEFAULT_DISTRIBUTION, branch_probability=0.0)

# No coupled modules: renormalize module_probabilities with "Coupled"'s
# weight zeroed (the other weights are unchanged, so their *relative*
# proportions among R/C/P are identical to G_FULL's).
G_NOCOUPLE: Distribution = replace(
    DEFAULT_DISTRIBUTION,
    module_probabilities=tuple(
        (kind, 0.0 if kind == "Coupled" else weight) for kind, weight in DEFAULT_DISTRIBUTION.module_probabilities
    ),
)

# Fully serial hands: no additional palm bodies (root is still always the
# one palm body -- see rules.py), no branching, no palm joints (moot once
# palm_body_count_range is (0, 0), but set for documentation/robustness).
G_SERIAL: Distribution = replace(
    DEFAULT_DISTRIBUTION,
    palm_body_count_range=(0, 0),
    branch_probability=0.0,
    palm_joint_probability=0.0,
)

# I14 fix 5: an "insertion" sub-Distribution (see distributions.Distribution
# .insertion / derive.py's growth operators) that growth operators draw NEW
# digit/phalanx material from: phalanges 1..3 (no growth op should ever
# insert a full-size, up-to-6-phalanx digit/subtree in one step), no
# branching. Everything else (palm-body-related fields included -- "palm
# bodies as before") is identical to DEFAULT_DISTRIBUTION.
_INSERTION_DIST: Distribution = replace(
    DEFAULT_DISTRIBUTION, phalanx_count_range=(1, 3), branch_probability=0.0,
)

# G_FULL, but every growth operator inserts new material from
# ``_INSERTION_DIST`` instead of the (otherwise identical) outer
# distribution.
G_FULL_INS: Distribution = replace(DEFAULT_DISTRIBUTION, insertion=_INSERTION_DIST)

# G_NOBRANCH, with the same insertion sub-distribution wired in.
G_NOBRANCH_INS: Distribution = replace(G_NOBRANCH, insertion=_INSERTION_DIST)

# G_FULL_SMALL: the SAME sampling distribution as G_FULL, paired with the
# small-step operator mixture (see derive.py's SMALL_STEP_OPERATORS) for use
# with vary(..., operators=...) rather than the coarse-grained OPERATORS
# default. A distinct object from a bare Distribution: (dist, operators).
G_FULL_SMALL: Tuple[Distribution, Tuple[str, ...]] = (G_FULL, SMALL_STEP_OPERATORS)

# Grammar 0.5 (I16 support-widening, priority 1): the rest-bend primitive,
# a 15-degree/5-mm grid (each rpy component in {-30,-15,0,15,30} deg, each
# offset component in {-5,0,5} mm, full Cartesian product -- see
# distributions.Distribution.bend_probability's docstring), probability 0.3.
_BEND_DEG_GRID: Tuple[float, ...] = (-30.0, -15.0, 0.0, 15.0, 30.0)
_BEND_OFFSET_MM_GRID: Tuple[float, ...] = (-5.0, 0.0, 5.0)

_BEND_RPY_CHOICES_RAD: Tuple[Tuple[float, float, float], ...] = tuple(
    tuple(v * math.pi / 180.0 for v in combo) for combo in itertools.product(_BEND_DEG_GRID, repeat=3)
)
_BEND_OFFSET_CHOICES_M: Tuple[Tuple[float, float], ...] = tuple(
    tuple(v / 1000.0 for v in combo) for combo in itertools.product(_BEND_OFFSET_MM_GRID, repeat=2)
)

G_BEND: Distribution = replace(
    DEFAULT_DISTRIBUTION,
    bend_rpy_choices_rad=_BEND_RPY_CHOICES_RAD,
    bend_offset_choices_m=_BEND_OFFSET_CHOICES_M,
    bend_probability=0.3,
)

# Grammar 0.5 (I16 support-widening, priority 2): continuous, sign-normalised
# revolute limits, range widened to (-180, 180) deg. NOTE (E11): generated
# designs sampled under G_CONT may exceed the grammar's usual (-45, 110) deg
# support range -- that is this variant's entire point (support widening),
# not a defect.
G_CONT: Distribution = replace(
    DEFAULT_DISTRIBUTION,
    limits_continuous=True,
    revolute_limit_range_deg=(-180.0, 180.0),
)

# --------------------------------------------------------------------------
# G0 CPU grammar screen (plan revision 2026-09-27, step 2; project-notes/
# grammar/experiments/g0/report.md). V0 is G_SERIAL (above): the
# old-sampler-like baseline. V1-V3 are built one generative rule at a time,
# each strictly on top of the last, so the screen can attribute any change
# in viability/diversity to the rule that variant adds. V4 (palm support
# under the spawn point) is deferred -- it needs geometry.py's palm-cell
# convex-hull machinery, out of scope for this CPU screen's time box; see
# the report's caveats.
# --------------------------------------------------------------------------

# V1: DEFAULT, constrained to what the viability oracle
# (isaacsimenvs.inhand_reorient.scene.grammar_envelope.admit) can even
# structurally admit -- revolute-only (every other module weight zeroed
# and renormalized: no couplings, no continuous/prismatic joints), no
# in-digit branching, at most 2 additional palm bodies, 1-5 digits.
# Without this, most of a raw G_FULL/G_SERIAL sample is thrown out by the
# STRUCTURAL gate alone, before viability (overlap, reach) is even
# measured -- V1 is the fair "old-sampler-like, but envelope-shaped"
# starting point the later rules build on.
G_V1: Distribution = replace(
    DEFAULT_DISTRIBUTION,
    module_probabilities=(("R", 1.0), ("C", 0.0), ("P", 0.0), ("Coupled", 0.0)),
    branch_probability=0.0,
    palm_body_count_range=(0, 2),
    digit_count_range=(1, 5),
)

# V2: V1 + the I29 finger-mount spacing rule (``sample_derivation``'s
# ``derive._plan_top_level_mounts``, gated by ``mount_min_separation_m``):
# top-level digit mounts are placed (never rejected/resampled) to spread
# across the available palm/root hosts and, within one host, across
# ``mount_frac_choices``, targeting this minimum physical separation
# between two same-host mounts. The target is a FIXED, conservative
# constant -- 2x the LARGEST sampled capsule radius (0.012 m; see
# ``Distribution.capsule_radius_choices_m``) plus a 5 mm margin (the same
# grid unit ``link_length_grid_m`` uses elsewhere) -- so every sampled
# radius clears it, not just the modal one.
G_V2: Distribution = replace(G_V1, mount_min_separation_m=2.0 * 0.012 + 0.005)

# V3: V2 + the I30 curl and opposition priors.
#
# Curl axis prior: ``digit_axis_elevation_band_deg=(60, 120)`` keeps every
# digit phalanx's revolute axis within 30 degrees of the horizontal (see
# ``distributions.sample_axis``'s ``elevation_band_deg``), i.e. hinge-like
# and roughly transverse to the segment's own forward direction, rather
# than a fully isotropic 3D axis. Opus review of G0 (item 5): this band
# restricts the axis's ELEVATION only -- its AZIMUTH is still drawn i.i.d.
# per phalanx, so it does NOT, on its own, put successive phalanges' axes
# in a shared hinge plane (measured: only 8% of successive axis pairs land
# within 20 deg of each other under V3). That stronger claim is not made
# here; V3/V3s only restrict each axis individually to the transverse band.
#
# Rest bend ("curl toward the palm normal"): reuses the EXISTING rest-bend
# primitive (``Distribution.bend_probability``/``bend_rpy_choices_rad`` --
# see ``G_BEND`` above for the symmetric-grid precedent) rather than a new
# mechanism, with a choice set of ONLY forward (positive-pitch) values and
# probability 1.0: every phalanx after a digit's first is a CONTINUATION
# joint, whose bend composes directly as its own origin orientation (no
# existing rotation to compose with -- see ``derive._compose_bend_rpy``),
# so a positive-pitch-only bend makes every phalanx curl the SAME
# consistent way relative to the previous phalanx's own heading, like a
# closing finger, instead of ``G_BEND``'s symmetric (any-direction, only
# 30% of the time) grid.
#
# Opposition prior: ``opposition_prior=True`` makes the LAST-sampled
# top-level digit (whenever there are >= 2) oppose the mean mounting
# direction of the earlier digits (``derive._best_opposing_rpy``) -- a
# thumb-like layout -- instead of an independently random mount
# orientation.
_CURL_BEND_RPY_CHOICES_RAD: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 15.0 * math.pi / 180.0, 0.0),
    (0.0, 30.0 * math.pi / 180.0, 0.0),
    (0.0, 45.0 * math.pi / 180.0, 0.0),
)
G_V3: Distribution = replace(
    G_V2,
    digit_axis_elevation_band_deg=(60.0, 120.0),
    opposition_prior=True,
    bend_rpy_choices_rad=_CURL_BEND_RPY_CHOICES_RAD,
    bend_offset_choices_m=((0.0, 0.0),),
    bend_probability=1.0,
)

# --------------------------------------------------------------------------
# Opus review of G0 (opus-review-g0.md, I39) -- V1/V2/V3 above are KEPT
# byte-identical, for reference/comparison; V1s/V2s/V3s below are the FIXED
# variants the rerun screen actually uses. Each is built the same
# one-rule-at-a-time way as V1-V3, on the new off-by-default fields
# ``distributions.Distribution`` gained for this review (surface mounting,
# curl-skip-first-phalanx, opposition-uses-host-frame) -- none of them
# changes any EXISTING named ``Distribution``'s sampling.
# --------------------------------------------------------------------------

# The same fixed, conservative separation target V2 already uses -- 2x the
# largest sampled capsule radius (0.012 m) plus a 5 mm margin -- reused
# here so V1s/V2s/V3s are directly attributable to the SAME "2*radius +
# margin" target the review asked for, not a different number.
_MOUNT_SEP_TARGET_M = 2.0 * 0.012 + 0.005

# V1s: V1 + surface mounting (review item 2) -- a digit's mount origin sits
# on its host's own surface (radial offset = the hand's one
# ``capsule_radius_m``), azimuth drawn i.i.d. from the same 15-degree grid
# ``mount_rpy`` uses (no deliberate spacing plan yet -- see V2s). This alone
# already turns same-host mount collisions from "guaranteed at low
# root/palm length" into "usually apart", since two i.i.d. azimuths at the
# same axial frac are no longer coincident.
G_V1S: Distribution = replace(G_V1, mount_on_host_surface=True)

# V2s: V1s + cross-host spacing (review item 2) -- ``mount_min_separation_m``
# now (via ``mount_on_host_surface=True``) dispatches to
# ``derive._plan_top_level_mounts_surface``: a greedy furthest-point search
# over every (host, frac, azimuth) grid point's ACTUAL 3-D position (root
# frame, via a scratch FK pass over the sampled root/palm bodies), so
# mounts on DIFFERENT hosts are spaced apart too, and a short root
# (20-80 mm) can still place 4-5 digits by spreading them around the
# surface, not just along its length. Sibling of V3s (both built on V1s,
# NOT nested under each other): V2s isolates the spacing rule's own effect,
# V3s isolates the curl/opposition fix's own effect, so the rerun can
# attribute either gain without the other confounding it.
G_V2S: Distribution = replace(G_V1S, mount_min_separation_m=_MOUNT_SEP_TARGET_M)

# V3s: V1s + the FIXED curl and opposition priors (review item 5) -- a
# sibling of V2s (see its own comment above), not built on top of it, so
# V3s isolates the curl/opposition fix from the spacing rule.
#
# ``curl_skip_first_phalanx=True``: phalanx 0 (whose origin composes with
# the digit's own ``mount_rpy`` -- see ``derive._compose_bend_rpy``) never
# receives a rest-bend, so the mount frame the opposition prior committed
# to is the digit's TRUE final rest-pose orientation, not one silently
# tilted 15-45 deg afterward (V3's bug). Every phalanx after the first
# still curls, exactly as V3 intends.
#
# ``opposition_use_host_frame=True``: the mean-forward direction (and the
# last digit's own opposing target) are computed in the ROOT frame, each
# earlier digit's ``mount_rpy`` rotated by ITS OWN host's accumulated
# rotation first (``derive._host_transforms_from_steps``) -- fixes V3's
# "105 deg in mixed-host hands" (root + palm-body digits averaged as if
# they shared one frame).
G_V3S: Distribution = replace(
    G_V1S,
    digit_axis_elevation_band_deg=(60.0, 120.0),
    opposition_prior=True,
    opposition_use_host_frame=True,
    bend_rpy_choices_rad=_CURL_BEND_RPY_CHOICES_RAD,
    bend_offset_choices_m=((0.0, 0.0),),
    bend_probability=1.0,
    curl_skip_first_phalanx=True,
)

# --------------------------------------------------------------------------
# G_WIDE (2026-10-06): G_FULL with its RULES widened just enough to express the
# commercial hands of grammar_bench/manifest.json (see
# adapters/conform.py's rule-conflict report): lateral finger mounts across
# the palm (the new, off-by-default ``mount_lateral_grid_m``: 5 mm grid,
# up to 65 mm), a rest bend at any joint on the full 15 degree rotation grid
# (roll and yaw all the way round, pitch -90..90; drawn with probability 0.2
# when sampling), continuous joint limits anywhere in +/-180 deg on the
# 15 degree step, and mount fractions in 5% steps. A reference for conforming
# real hands and for weighing those extensions, not a proposal for sampling
# (its random hands are bent at arbitrary angles).
# --------------------------------------------------------------------------

_WIDE_BEND_RPY_CHOICES_RAD: Tuple[Tuple[float, float, float], ...] = tuple(
    (r * math.pi / 180.0, pch * math.pi / 180.0, y * math.pi / 180.0)
    for r in range(-180, 180, 15) for pch in range(-90, 91, 15) for y in range(-180, 180, 15)
)

G_WIDE: Distribution = replace(
    DEFAULT_DISTRIBUTION,
    mount_lateral_grid_m=0.005,
    mount_lateral_max_m=0.065,
    bend_rpy_choices_rad=_WIDE_BEND_RPY_CHOICES_RAD,
    bend_offset_choices_m=((0.0, 0.0),),
    bend_probability=0.2,
    limits_continuous=True,
    revolute_limit_range_deg=(-180.0, 180.0),
    mount_frac_choices=tuple(round(k * 0.05, 10) for k in range(21)),
)

# --------------------------------------------------------------------------
# THE grammar (2026-10-06): one base distribution plus three on/off rules.
#
# Under the three-layer design most named variants above are redundant:
# what V1/SERIAL/NOPALMJOINT/NOBRANCH/NOCOUPLE restrict is a generation LIMIT
# (limits.py), not a grammar. They stay unchanged in NAMED_DISTRIBUTIONS for
# reproducing past experiments; new work uses ``build_distribution``.
#
# ``GRAMMAR_BASE``: the default grammar (G_FULL's productions and sampling
# priors) with capability grids wide enough to contain the commercial hands
# (adapters/conform.py): mount fractions in 5% steps, a 5 mm lateral mount
# grid up to 65 mm, a rest bend at any joint on the 15 degree rotation grid,
# and joint ranges anywhere in +/-180 deg. The last three are support only:
# mutation reaches them and a conformed real hand may use them, but random
# sampling never draws them (distributions.py's support fields).
# --------------------------------------------------------------------------

GRAMMAR_BASE: Distribution = replace(
    DEFAULT_DISTRIBUTION,
    mount_frac_choices=tuple(round(k * 0.05, 10) for k in range(21)),
    mount_lateral_grid_m=0.005,
    mount_lateral_max_m=0.065,
    mount_lateral_sampled=False,
    bend_support_rpy_choices_rad=_WIDE_BEND_RPY_CHOICES_RAD,
    limits_support_continuous=True,
    revolute_limit_range_deg=(-180.0, 180.0),
)

RULES: Tuple[str, ...] = ("surface", "spacing", "curl_opposition")
RULE_TEXT = {
    "surface": "fingers sit on the palm surface",
    "spacing": "fingers spaced apart",
    "curl_opposition": "fingers curl and oppose",
}


def build_distribution(surface: bool = True, spacing: bool = True, curl_opposition: bool = True) -> Distribution:
    """The one grammar with its three generation rules switched on or off.

    - ``surface`` (from V1s): a finger's mount sits on its host's capsule
      surface at a 15 degree azimuth, not on the host's axis;
    - ``spacing`` (from V2/V2s): finger mounts are planned at least 29 mm
      apart (2 x the largest radius + 5 mm), across all hosts in 3-D when
      ``surface`` is on, along each host otherwise;
    - ``curl_opposition`` (the V3s bundle): digit hinge axes within 30 deg of
      transverse, every bone after a finger's first curled 15-45 deg toward
      the palm, and the last finger turned to oppose the others (in the root
      frame).

    All three on is V1s + spacing + the V3s curl/opposition on the wide
    grids of ``GRAMMAR_BASE``; it is not byte-identical to ``G_V3S`` (whose
    base is G_V1's restricted counts and 5 mount fractions, without spacing).
    Restrictions such as hinge-only joints, no branches or at most 5 fingers
    are generation limits (``limits.SIMULATOR``), not rules."""
    d = GRAMMAR_BASE
    if surface:
        d = replace(d, mount_on_host_surface=True)
    if spacing:
        d = replace(d, mount_min_separation_m=_MOUNT_SEP_TARGET_M)
    if curl_opposition:
        d = replace(
            d,
            digit_axis_elevation_band_deg=(60.0, 120.0),
            opposition_prior=True,
            opposition_use_host_frame=True,
            bend_rpy_choices_rad=_CURL_BEND_RPY_CHOICES_RAD,
            bend_offset_choices_m=((0.0, 0.0),),
            bend_probability=1.0,
            curl_skip_first_phalanx=True,
        )
    return d


# Every named Distribution variant above, for iteration by experiment code.
NAMED_DISTRIBUTIONS = {
    "G_FULL": G_FULL,
    "G_NOPALMJOINT": G_NOPALMJOINT,
    "G_NOBRANCH": G_NOBRANCH,
    "G_NOCOUPLE": G_NOCOUPLE,
    "G_SERIAL": G_SERIAL,
    "G_FULL_INS": G_FULL_INS,
    "G_NOBRANCH_INS": G_NOBRANCH_INS,
    "G_BEND": G_BEND,
    "G_CONT": G_CONT,
    "G_V1": G_V1,
    "G_V2": G_V2,
    "G_V3": G_V3,
    "G_V1S": G_V1S,
    "G_V2S": G_V2S,
    "G_V3S": G_V3S,
    "G_WIDE": G_WIDE,
}

# The G0 screen's own variant roster, named per the plan (V0 = G_SERIAL).
# V1/V2/V3 kept for reference; V1s/V2s/V3s are the fixed variants the rerun
# (opus-review-g0.md) uses.
G0_SCREEN_VARIANTS = {
    "V0": G_SERIAL,
    "V1": G_V1,
    "V2": G_V2,
    "V3": G_V3,
    "V1s": G_V1S,
    "V2s": G_V2S,
    "V3s": G_V3S,
}

__all__ = [
    "G_FULL",
    "G_NOPALMJOINT",
    "G_NOBRANCH",
    "G_NOCOUPLE",
    "G_SERIAL",
    "G_FULL_INS",
    "G_NOBRANCH_INS",
    "G_FULL_SMALL",
    "G_BEND",
    "G_CONT",
    "G_V1",
    "G_V2",
    "G_V3",
    "G_V1S",
    "G_V2S",
    "G_V3S",
    "G_WIDE",
    "GRAMMAR_BASE",
    "RULES",
    "RULE_TEXT",
    "build_distribution",
    "NAMED_DISTRIBUTIONS",
    "G0_SCREEN_VARIANTS",
    "OPERATORS",
    "SMALL_STEP_OPERATORS",
]
