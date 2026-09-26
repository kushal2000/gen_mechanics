"""E13: representation-check at 5 mm / 10 degrees, plus the real-hand
parameter atlas (representation-check plan, item 3).

For every available hand in ``grammar_bench/manifest.json`` (``dev``,
``heldout``, ``articulated_palm`` splits; ``excluded`` hands are always
skipped) plus the analytic fixture ``fixtures/analytic/coupled_finger.urdf``:

    ``load_urdf(hand_root)`` -> ``project_to_derivation(palm_joints,
    tip_frames)`` -> ``derive`` -> compare with the original model's own
    forward kinematics over zero plus ``N_RANDOM_CONFIGS`` random
    configurations (seed ``SEED``). Unlike
    ``grammar_bench/tests/test_representation_projection.py`` (item 2,
    which holds the projection to numerical precision but only samples the
    ORIGINAL model's *independent* joints, leaving every mimic/coupled
    joint at its default 0), every configuration here samples ALL of the
    original model's movable joints independently within their own
    declared limits (continuous joints in ``[-pi, pi]``) -- including
    mimic/coupled joints, which take an arbitrary value of their own,
    ignoring the coupling relationship. This is deliberate: ``fk.py``'s
    ``forward_kinematics`` never reads ``model.couplings`` at all (a
    coupling is bookkeeping metadata consumed elsewhere, e.g.
    ``coords.q_from_u``); it drives every joint literally from whatever
    ``q`` says, defaulting missing entries to 0. So the honest test that
    the projection has transcribed every joint's own geometry (origin,
    axis) correctly -- independent of whether its coupling relationship
    also survived the projection -- is to feed the SAME arbitrary value to
    both the original and the projected joint (via ``name_map``) and check
    the resulting poses agree, for every joint, not just the independent
    ones. PASS iff max joint position error <= 5 mm, max joint axis error
    <= 10 deg, max fingertip error <= 5 mm, and the movable-joint count is
    conserved.

The atlas pools, per hand and overall, the continuous parameter values the
projected ``Derivation`` actually holds (palm-body axes/mounts, digit/
phalanx counts, link lengths, rest bends, axis-to-link angles, joint limits,
couplings kept/dropped), and compares each against
``distributions.DEFAULT_DISTRIBUTION``'s sampled range or choice set -- input
for later grammar tuning; this script changes nothing in ``rules.py`` or
``distributions.py``.

Stdlib + numpy + this project's own ``hand_sampler`` package only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..adapters.projection import ProjectionFailure, project_to_derivation
from ..adapters.urdf import load_urdf
from ..derive import derive, validate_derivation
from ..distributions import DEFAULT_DISTRIBUTION
from ..fk import forward_kinematics, pose_to_matrix, rotation_error, rpy_to_matrix
from ..kinematics import MOVABLE_TYPES
from .runner import register, run_experiment

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = REPO_ROOT / "hand_sampler" / "grammar_bench"
MANIFEST_PATH = BENCH_DIR / "manifest.json"
ANALYTIC_FIXTURE = BENCH_DIR / "fixtures" / "analytic" / "coupled_finger.urdf"

SEED = 20260926
N_RANDOM_CONFIGS = 64
POS_TOL_M = 0.005
AXIS_TOL_DEG = 10.0
AXIS_TOL_RAD = AXIS_TOL_DEG * math.pi / 180.0

# Splits scored by this experiment. "excluded" is always skipped (a local
# entry -- svh_right, shadow_right_local, arms_skel -- replaces it where the
# plan calls for one; the *_local ids are "articulated_palm").
SCORED_SPLITS = ("dev", "heldout", "articulated_palm")


# --------------------------------------------------------------------------
# Hand resolution -- mirrors grammar_bench/evaluate.py's own ``_resolve_hand``
# exactly (same pattern ``e11_support_widening.py`` follows: this
# ``hand_sampler.grammar`` package never imports the benchmarking package
# ``hand_sampler.grammar_bench``, only mirrors its logic). Its absolute-path
# branch (new for this experiment: ``svh_right``/``shadow_right_local``/
# ``arms_skel`` all give an absolute local ``source_path``) needs no special
# case: ``Path(source_root) / "/abs/path"`` returns the absolute path
# unchanged (a documented ``pathlib`` behaviour -- the right operand being
# absolute overrides the left; verified directly against this repo's
# manifest before writing this module), so it falls straight out of the
# existing ``source_root``-join branch below.
# --------------------------------------------------------------------------


def _resolve_hand(hand: dict, manifest: dict) -> Tuple[Optional[Path], str, Optional[str]]:
    if hand.get("split") == "excluded":
        return None, "excluded", hand.get("notes") or "excluded split: never imported"

    fixture_path = hand.get("fixture_path")
    if fixture_path:
        p = BENCH_DIR / fixture_path
        if p.is_file():
            return p, "available", None
        return None, "unavailable", f"fixture_path listed in manifest but missing on disk: {fixture_path}"

    source_path = hand.get("source_path")
    if isinstance(source_path, str) and source_path.startswith("REPO:"):
        p = REPO_ROOT / source_path[len("REPO:"):]
        if p.is_file():
            return p, "available", None
        return None, "unavailable", f"REPO path listed in manifest but missing on disk: {source_path}"

    if isinstance(source_path, str) and source_path:
        source_root = manifest.get("source_root")
        if source_root:
            p = Path(source_root) / source_path  # absolute source_path overrides source_root (pathlib)
            if p.is_file():
                return p, "available", None
        return None, "unavailable", f"local-only:{hand['id']} source not present at manifest source_root on this machine"

    return None, "unavailable", "no fixture_path/source_path in manifest"


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def _cases(manifest: dict, hand_ids: Optional[Sequence[str]] = None) -> List[Tuple[dict, Optional[Path], str, Optional[str]]]:
    """Every scored manifest hand plus the analytic fixture, each as
    ``(hand_dict, path_or_None, availability, reason_or_None)``. ``hand_ids``
    (optional), when given, restricts to just those ids -- used by tests to
    exercise the core check/atlas plumbing on a handful of hands without
    running the whole manifest."""
    out: List[Tuple[dict, Optional[Path], str, Optional[str]]] = []
    for hand in manifest["hands"]:
        if hand.get("split") not in SCORED_SPLITS:
            continue
        if hand_ids is not None and hand["id"] not in hand_ids:
            continue
        path, availability, reason = _resolve_hand(hand, manifest)
        out.append((hand, path, availability, reason))
    analytic_hand = {
        "id": "coupled_finger", "family": "analytic_fixture", "split": "analytic",
        "hand_root": None, "palm_joints": [], "tip_frames": {},
    }
    if hand_ids is None or "coupled_finger" in hand_ids:
        if ANALYTIC_FIXTURE.is_file():
            out.append((analytic_hand, ANALYTIC_FIXTURE, "available", None))
        else:
            out.append((analytic_hand, None, "unavailable", "local-only:coupled_finger fixture missing on disk"))
    return out


# --------------------------------------------------------------------------
# Core per-hand check
# --------------------------------------------------------------------------


def _rotation_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cos_t = float(np.dot(a, b) / (na * nb))
    cos_t = max(-1.0, min(1.0, cos_t))
    return float(math.acos(cos_t))


def check_hand(hand_id: str, path: Path, hand_root: Optional[str], palm_joints: Sequence[str],
               tip_frames: Dict[str, str], seed: int = SEED, n_configs: int = N_RANDOM_CONFIGS) -> Dict[str, Any]:
    """Project ``hand_id``, derive it back, and compare forward kinematics
    against the original over zero + ``n_configs`` random configurations
    sampling EVERY original movable joint independently (see module
    docstring). Returns a JSON-serializable dict with the pass/fail
    verdict, error magnitudes, the projection report, and the atlas
    contribution for this one hand (empty atlas on failure)."""
    result: Dict[str, Any] = {
        "id": hand_id, "ok": False, "passed": False, "reason": None,
        "max_pos_mm": None, "max_axis_deg": None, "max_tip_mm": None,
        "n_movable_orig": None, "n_movable_der": None, "joint_count_conserved": None,
        "report": None, "atlas": None,
    }
    try:
        imported = load_urdf(path, hand_root=hand_root)
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"import failed: {type(exc).__name__}: {exc}"
        return result

    model = imported.model
    n_orig = sum(1 for j in model.joints if j.type in MOVABLE_TYPES)

    try:
        pr = project_to_derivation(model, palm_joints=palm_joints, tip_frames=tip_frames)
    except ProjectionFailure as exc:
        result["reason"] = f"projection failed: {exc}"
        result["report"] = exc.report
        return result

    issues = validate_derivation(pr.derivation)
    if issues:
        result["reason"] = f"validate_derivation issues: {issues}"
        result["report"] = pr.report
        return result

    derived = derive(pr.derivation)
    n_der = sum(1 for j in derived.joints if j.type in MOVABLE_TYPES)
    result["n_movable_orig"], result["n_movable_der"] = n_orig, n_der
    result["joint_count_conserved"] = bool(n_orig == n_der)
    result["ok"] = True
    result["report"] = pr.report

    movable_joints = [j for j in model.joints if j.type in MOVABLE_TYPES]
    derived_joint_by_name = {j.name: j for j in derived.joints}
    derived_frames = {f.name: f for f in derived.frames}

    rng = np.random.default_rng(seed)
    max_pos, max_axis_rad, max_tip = 0.0, 0.0, 0.0

    for trial in range(n_configs + 1):
        q_orig: Dict[str, float] = {}
        for j in movable_joints:
            if trial == 0:
                q_orig[j.name] = 0.0
            elif j.type == "continuous":
                q_orig[j.name] = float(rng.uniform(-math.pi, math.pi))
            else:
                lo, hi = j.limits
                q_orig[j.name] = float(rng.uniform(lo, hi))

        T_orig = forward_kinematics(model, q_orig)
        q_der = {pr.name_map[jn]: v for jn, v in q_orig.items() if jn in pr.name_map}
        T_der = forward_kinematics(derived, q_der)
        RT = pr.root_transform

        for j in movable_joints:
            djn = pr.name_map.get(j.name)
            if djn is None:
                continue
            derived_body = djn[:-2]  # "<body>_j" -> "<body>" (derive.py's own convention)
            if derived_body not in T_der:
                continue
            T_o, T_d = T_orig[j.child], RT @ T_der[derived_body]
            max_pos = max(max_pos, float(np.linalg.norm(T_o[:3, 3] - T_d[:3, 3])))

            dj = derived_joint_by_name[djn]
            axis_o = T_o[:3, :3] @ np.asarray(j.axis, dtype=float)
            axis_d = T_d[:3, :3] @ np.asarray(dj.axis, dtype=float)
            max_axis_rad = max(max_axis_rad, _rotation_angle_between(axis_o, axis_d))

        for orig_body, derived_name in pr.name_map.items():
            if not derived_name.endswith("_tip") or orig_body not in T_orig:
                continue
            fr = derived_frames.get(derived_name)
            if fr is None:
                continue
            T_tip_der_root = RT @ (T_der[fr.body] @ pose_to_matrix(fr.pose))
            max_tip = max(max_tip, float(np.linalg.norm(T_orig[orig_body][:3, 3] - T_tip_der_root[:3, 3])))

    result["max_pos_mm"] = max_pos * 1000.0
    result["max_axis_deg"] = max_axis_rad * 180.0 / math.pi
    result["max_tip_mm"] = max_tip * 1000.0
    result["passed"] = bool(
        max_pos <= POS_TOL_M and max_axis_rad <= AXIS_TOL_RAD and max_tip <= POS_TOL_M
        and result["joint_count_conserved"]
    )
    result["atlas"] = _hand_atlas(hand_id, pr, derived, model)
    return result


# --------------------------------------------------------------------------
# Atlas: continuous-parameter extraction from one hand's projected Derivation
# --------------------------------------------------------------------------


def _bend_angle_deg(bend_rpy: Sequence[float]) -> float:
    if all(abs(v) < 1e-12 for v in bend_rpy):
        return 0.0
    R = rpy_to_matrix(tuple(float(v) for v in bend_rpy))
    return rotation_error(np.eye(3), R) * 180.0 / math.pi


def _hand_atlas(hand_id: str, pr, derived, model) -> Dict[str, Any]:
    steps_by_prod: Dict[str, List[dict]] = {"Hand": [], "PalmBody": [], "Digit": [], "Phalanx": []}
    for s in pr.derivation.steps:
        steps_by_prod.setdefault(s.production, []).append(s.params)

    T0 = forward_kinematics(derived, {})

    hand_params = steps_by_prod["Hand"][0] if steps_by_prod["Hand"] else {}
    root_length_m = float(hand_params.get("root_length", 0.0))

    palm_bodies = []
    for p in steps_by_prod["PalmBody"]:
        name = p["name"]
        mount_offset = tuple(float(v) for v in p.get("mount_offset", (0.0, 0.0)))
        entry = {
            "name": name,
            "parent": p.get("parent"),
            "has_joint": bool(p.get("has_joint")),
            "length_m": float(p.get("length", 0.0)),
            "mount_frac": float(p.get("mount_frac", 0.0)),
            "mount_offset_mag_m": float(math.hypot(*mount_offset)),
            "direction_rpy_deg": [float(v) * 180.0 / math.pi for v in p.get("direction_rpy", (0.0, 0.0, 0.0))],
            "axis_to_root_z_deg": None,
            "limits_deg": None,
        }
        if entry["has_joint"] and name in T0:
            axis_local = np.asarray(p.get("axis", (0.0, 0.0, 1.0)), dtype=float)
            axis_root = T0[name][:3, :3] @ axis_local
            entry["axis_to_root_z_deg"] = _rotation_angle_between(axis_root, np.array([0.0, 0.0, 1.0])) * 180.0 / math.pi
            lo, hi = p.get("limits", (0.0, 0.0))
            entry["limits_deg"] = [lo * 180.0 / math.pi, hi * 180.0 / math.pi]
        palm_bodies.append(entry)

    digits_by_id = {d["digit_id"]: d for d in steps_by_prod["Digit"]}
    phalanges_per_digit: Dict[str, int] = {d["digit_id"]: int(d.get("phalanx_count", 0)) for d in steps_by_prod["Digit"]}

    phalanges = []
    for ph in steps_by_prod["Phalanx"]:
        digit_id, p_idx = ph["digit_id"], int(ph["p"])
        module = ph.get("module", {})
        axis_local = module.get("axis")
        axis_to_link_deg = None
        if axis_local is not None:
            axis_to_link_deg = _rotation_angle_between(np.asarray(axis_local, dtype=float), np.array([0.0, 0.0, 1.0])) * 180.0 / math.pi
        bend_rpy = ph.get("bend_rpy", (0.0, 0.0, 0.0))
        bend_offset = ph.get("bend_offset", (0.0, 0.0))
        entry = {
            "digit_id": digit_id, "p": p_idx, "length_m": float(ph.get("length", 0.0)),
            "module_kind": module.get("kind"), "axis_to_link_deg": axis_to_link_deg,
            "lateral_offset_mag_m": float(math.hypot(*bend_offset)),
            "bend_angle_deg": _bend_angle_deg(bend_rpy) if p_idx > 0 else 0.0,
            "mount_frac": float(digits_by_id[digit_id].get("mount_frac", 0.0)) if p_idx == 0 else None,
            "limits": None,
        }
        if module.get("kind") == "R":
            lo, hi = module.get("limits", (0.0, 0.0))
            entry["limits"] = {"unit": "deg", "lo": lo * 180.0 / math.pi, "hi": hi * 180.0 / math.pi}
        elif module.get("kind") == "P":
            lo, hi = module.get("limits", (0.0, 0.0))
            entry["limits"] = {"unit": "m", "lo": lo, "hi": hi}
        elif module.get("kind") == "Coupled":
            entry["coupling"] = {"multiplier": module.get("multiplier"), "offset_rad": module.get("offset")}
        phalanges.append(entry)

    return {
        "hand_id": hand_id,
        "root_length_m": root_length_m,
        "palm_body_count": len(palm_bodies),
        "palm_bodies": palm_bodies,
        "digit_count": len(digits_by_id),
        "phalanges_per_digit": phalanges_per_digit,
        "phalanges": phalanges,
        "couplings_kept": len(derived.couplings),
        "couplings_dropped_not_in_structure": list(pr.report.get("coupling_not_in_structure", [])),
        "fingertip_undefined": list(pr.report.get("fingertip_undefined", [])),
        "merged_fixed_joint_count": len(pr.report.get("merged_fixed_joints", [])),
        "dependent_limits_replaced": list(pr.report.get("dependent_limits_replaced", [])),
    }


# --------------------------------------------------------------------------
# Pooling across hands + the grammar-gap table
# --------------------------------------------------------------------------


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "median": None, "max": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {"min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max()), "n": len(values)}


def _frac_outside(values: Sequence[float], lo: float, hi: float) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(np.mean((arr < lo - 1e-9) | (arr > hi + 1e-9)))


def build_atlas(per_hand: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool every passing/available hand's ``_hand_atlas`` output plus the
    real-hand-vs-``DEFAULT_DISTRIBUTION`` grammar-gap table."""
    hand_atlases = [h["atlas"] for h in per_hand if h.get("atlas") is not None]

    link_lengths_m: List[float] = []
    palm_lengths_m: List[float] = []
    knuckle_under_5mm = 0
    mount_fracs: List[float] = []
    lateral_offsets_m: List[float] = []
    bend_angles_deg: List[float] = []  # continuation phalanges only (p > 0)
    axis_to_link_deg: List[float] = []
    palm_axis_to_root_z_deg: List[float] = []
    palm_mount_offset_m: List[float] = []
    revolute_lo_deg: List[float] = []
    revolute_hi_deg: List[float] = []
    prismatic_lo_m: List[float] = []
    prismatic_hi_m: List[float] = []
    coupling_multipliers: List[float] = []
    coupling_offsets_rad: List[float] = []
    digit_counts: List[float] = []
    palm_body_counts: List[float] = []
    phalanx_counts: List[float] = []
    couplings_kept_total, couplings_dropped_total = 0, 0

    for a in hand_atlases:
        digit_counts.append(a["digit_count"])
        palm_body_counts.append(a["palm_body_count"])
        for n in a["phalanges_per_digit"].values():
            phalanx_counts.append(n)
        if a["root_length_m"] > 0:
            palm_lengths_m.append(a["root_length_m"])
        for pb in a["palm_bodies"]:
            palm_lengths_m.append(pb["length_m"])
            palm_mount_offset_m.append(pb["mount_offset_mag_m"])
            mount_fracs.append(pb["mount_frac"])
            if pb["length_m"] < 0.005:
                knuckle_under_5mm += 1
            if pb["axis_to_root_z_deg"] is not None:
                palm_axis_to_root_z_deg.append(pb["axis_to_root_z_deg"])
            if pb["limits_deg"] is not None:
                revolute_lo_deg.append(pb["limits_deg"][0])
                revolute_hi_deg.append(pb["limits_deg"][1])
        for ph in a["phalanges"]:
            link_lengths_m.append(ph["length_m"])
            if ph["length_m"] < 0.005:
                knuckle_under_5mm += 1
            if ph["axis_to_link_deg"] is not None:
                axis_to_link_deg.append(ph["axis_to_link_deg"])
            if ph["p"] == 0 and ph["mount_frac"] is not None:
                mount_fracs.append(ph["mount_frac"])
                lateral_offsets_m.append(ph["lateral_offset_mag_m"])
            if ph["p"] > 0:
                bend_angles_deg.append(ph["bend_angle_deg"])
                lateral_offsets_m.append(ph["lateral_offset_mag_m"])
            if ph.get("limits") and ph["limits"]["unit"] == "deg":
                revolute_lo_deg.append(ph["limits"]["lo"])
                revolute_hi_deg.append(ph["limits"]["hi"])
            elif ph.get("limits") and ph["limits"]["unit"] == "m":
                prismatic_lo_m.append(ph["limits"]["lo"])
                prismatic_hi_m.append(ph["limits"]["hi"])
            if ph.get("coupling"):
                coupling_multipliers.append(ph["coupling"]["multiplier"])
                coupling_offsets_rad.append(ph["coupling"]["offset_rad"])
        couplings_kept_total += a["couplings_kept"]
        couplings_dropped_total += len(a["couplings_dropped_not_in_structure"])

    dist = DEFAULT_DISTRIBUTION
    rev_choice_los = [lo for lo, hi in dist.revolute_limit_choices_deg] + [lo for lo, hi in dist.palm_joint_limit_choices_deg]
    rev_choice_his = [hi for lo, hi in dist.revolute_limit_choices_deg] + [hi for lo, hi in dist.palm_joint_limit_choices_deg]
    pris_choice_los = [lo for lo, hi in dist.prismatic_limit_choices_m]
    pris_choice_his = [hi for lo, hi in dist.prismatic_limit_choices_m]

    gap_rows = [
        {"parameter": "digit_count (per hand)", "real": _stats(digit_counts),
         "grammar": f"range {dist.digit_count_range}",
         "frac_outside": _frac_outside(digit_counts, *dist.digit_count_range)},
        {"parameter": "palm_body_count (non-root, per hand)", "real": _stats(palm_body_counts),
         "grammar": f"range {dist.palm_body_count_range}",
         "frac_outside": _frac_outside(palm_body_counts, *dist.palm_body_count_range)},
        {"parameter": "phalanx_count (per digit)", "real": _stats(phalanx_counts),
         "grammar": f"range {dist.phalanx_count_range}",
         "frac_outside": _frac_outside(phalanx_counts, *dist.phalanx_count_range)},
        {"parameter": "phalanx link length (m)", "real": _stats(link_lengths_m),
         "grammar": f"range {dist.link_length_range_m}, grid {dist.link_length_grid_m} m",
         "frac_outside": _frac_outside(link_lengths_m, *dist.link_length_range_m)},
        {"parameter": "palm/root segment length (m)", "real": _stats(palm_lengths_m),
         "grammar": f"range {dist.palm_length_range_m}",
         "frac_outside": _frac_outside(palm_lengths_m, *dist.palm_length_range_m)},
        {"parameter": "mount_frac", "real": _stats(mount_fracs),
         "grammar": f"choices {dist.mount_frac_choices}",
         "frac_outside": _frac_outside(mount_fracs, min(dist.mount_frac_choices), max(dist.mount_frac_choices))},
        {"parameter": "lateral mount offset (mount_offset / bend_offset, m)", "real": _stats(lateral_offsets_m + palm_mount_offset_m),
         "grammar": f"choices {dist.bend_offset_choices_m} (mount_offset has no grammar production at all)",
         "frac_outside": _frac_outside(lateral_offsets_m + palm_mount_offset_m, 0.0, 0.0)},
        {"parameter": "rest bend angle, continuation phalanges (deg)", "real": _stats(bend_angles_deg),
         "grammar": f"choices {dist.bend_rpy_choices_rad} (bend_probability={dist.bend_probability})",
         "frac_outside": _frac_outside(bend_angles_deg, 0.0, 0.0)},
        {"parameter": "axis-to-link angle, phalanges (deg; 90=perpendicular)", "real": _stats(axis_to_link_deg),
         "grammar": "axis drawn on a 15-degree spherical grid (continuous elevation/azimuth choice set, not a perpendicularity constraint)",
         "frac_outside": None},
        {"parameter": "palm-joint axis vs root +z (deg)", "real": _stats(palm_axis_to_root_z_deg),
         "grammar": "same 15-degree spherical grid as phalanx axes",
         "frac_outside": None},
        {"parameter": "revolute/palm joint limit lo (deg)", "real": _stats(revolute_lo_deg),
         "grammar": f"choice-set range [{min(rev_choice_los):.1f}, {max(rev_choice_los):.1f}] "
                    f"or continuous range {dist.revolute_limit_range_deg} if limits_continuous",
         "frac_outside": _frac_outside(revolute_lo_deg, min(rev_choice_los), max(rev_choice_los))},
        {"parameter": "revolute/palm joint limit hi (deg)", "real": _stats(revolute_hi_deg),
         "grammar": f"choice-set range [{min(rev_choice_his):.1f}, {max(rev_choice_his):.1f}] "
                    f"or continuous range {dist.revolute_limit_range_deg} if limits_continuous",
         "frac_outside": _frac_outside(revolute_hi_deg, min(rev_choice_his), max(rev_choice_his))},
        {"parameter": "prismatic joint limit lo (m)", "real": _stats(prismatic_lo_m),
         "grammar": f"choice-set range [{min(pris_choice_los):.3f}, {max(pris_choice_los):.3f}]" if pris_choice_los else "n/a",
         "frac_outside": _frac_outside(prismatic_lo_m, min(pris_choice_los), max(pris_choice_los)) if pris_choice_los else None},
        {"parameter": "prismatic joint limit hi (m)", "real": _stats(prismatic_hi_m),
         "grammar": f"choice-set range [{min(pris_choice_his):.3f}, {max(pris_choice_his):.3f}]" if pris_choice_his else "n/a",
         "frac_outside": _frac_outside(prismatic_hi_m, min(pris_choice_his), max(pris_choice_his)) if pris_choice_his else None},
        {"parameter": "coupling multiplier", "real": _stats(coupling_multipliers),
         "grammar": f"choices {dist.coupling_multiplier_choices}",
         "frac_outside": float(np.mean([m not in dist.coupling_multiplier_choices for m in coupling_multipliers])) if coupling_multipliers else None},
        {"parameter": "coupling offset (rad)", "real": _stats(coupling_offsets_rad),
         "grammar": f"choices {dist.coupling_offset_choices_rad}",
         "frac_outside": float(np.mean([o not in dist.coupling_offset_choices_rad for o in coupling_offsets_rad])) if coupling_offsets_rad else None},
    ]

    return {
        "n_hands_pooled": len(hand_atlases),
        "knuckle_under_5mm_count": knuckle_under_5mm,
        "couplings_kept_total": couplings_kept_total,
        "couplings_dropped_not_in_structure_total": couplings_dropped_total,
        "per_hand": hand_atlases,
        "grammar_gap_table": gap_rows,
    }


# --------------------------------------------------------------------------
# Top-level driver (runner.py per-seed function)
# --------------------------------------------------------------------------


def run_e13(seed: int, n_configs: int = N_RANDOM_CONFIGS, hand_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    manifest = _load_manifest()
    per_hand: List[Dict[str, Any]] = []
    for hand, path, availability, reason in _cases(manifest, hand_ids=hand_ids):
        entry: Dict[str, Any] = {
            "id": hand["id"], "family": hand.get("family"), "split": hand.get("split"),
            "availability": availability,
        }
        if availability != "available":
            entry.update({"reason": reason, "passed": None, "atlas": None})
            per_hand.append(entry)
            continue
        check = check_hand(
            hand["id"], path, hand.get("hand_root"), hand.get("palm_joints") or (),
            hand.get("tip_frames") or {}, seed=seed, n_configs=n_configs,
        )
        entry.update(check)
        per_hand.append(entry)

    n_available = sum(1 for h in per_hand if h["availability"] == "available")
    n_pass = sum(1 for h in per_hand if h.get("passed"))
    n_fail = sum(1 for h in per_hand if h["availability"] == "available" and not h.get("passed"))
    max_pos_mm_all = max((h["max_pos_mm"] for h in per_hand if h.get("max_pos_mm") is not None), default=0.0)
    max_axis_deg_all = max((h["max_axis_deg"] for h in per_hand if h.get("max_axis_deg") is not None), default=0.0)
    max_tip_mm_all = max((h["max_tip_mm"] for h in per_hand if h.get("max_tip_mm") is not None), default=0.0)

    atlas = build_atlas(per_hand)

    return {
        "n_hands_total": len(per_hand),
        "n_available": n_available,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "max_pos_mm_all_hands": max_pos_mm_all,
        "max_axis_deg_all_hands": max_axis_deg_all,
        "max_tip_mm_all_hands": max_tip_mm_all,
        "_per_hand": per_hand,
        "_atlas": atlas,
    }


register("e13_representation", run_e13)


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def _fmt(v: Optional[float], nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}g}"


def render_markdown(per_seed_result: Dict[str, Any]) -> str:
    per_hand = per_seed_result["_per_hand"]
    atlas = per_seed_result["_atlas"]
    lines = ["# E13: representation check (5 mm / 10 deg) and real-hand atlas", ""]
    lines.append(
        f"{per_seed_result['n_available']} available hands, {per_seed_result['n_pass']} PASS, "
        f"{per_seed_result['n_fail']} FAIL. Tolerances: {POS_TOL_M * 1000:.0f} mm / {AXIS_TOL_DEG:.0f} deg. "
        f"Seed {SEED}, {N_RANDOM_CONFIGS} random configs + the zero config, every original movable joint "
        "(including mimic/coupled ones) sampled independently within its own declared limits."
    )
    lines.append("")
    lines.append("## Per-hand results")
    lines.append("")
    lines.append("| hand | split | available | pass | max pos (mm) | max axis (deg) | max tip (mm) | notes |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for h in per_hand:
        if h["availability"] != "available":
            lines.append(f"| {h['id']} | {h['split']} | {h['availability']} | - | - | - | - | {h.get('reason') or '-'} |")
            continue
        notes = []
        rep = h.get("report") or {}
        if h.get("reason"):
            notes.append(h["reason"])
        if rep.get("fingertip_undefined"):
            notes.append(f"fingertip_undefined: {rep['fingertip_undefined']}")
        if rep.get("coupling_not_in_structure"):
            notes.append(f"coupling_not_in_structure: {rep['coupling_not_in_structure']}")
        if rep.get("branch_not_projected"):
            notes.append(f"branch_not_projected: {rep['branch_not_projected']}")
        note_str = "; ".join(notes) if notes else "-"
        lines.append(
            f"| {h['id']} | {h['split']} | {h['availability']} | {'PASS' if h.get('passed') else 'FAIL'} "
            f"| {_fmt(h.get('max_pos_mm'))} | {_fmt(h.get('max_axis_deg'))} | {_fmt(h.get('max_tip_mm'))} | {note_str} |"
        )
    lines.append("")

    lines.append("## Atlas highlights (pooled over passing/scored hands)")
    lines.append("")
    lines.append(f"Hands pooled: {atlas['n_hands_pooled']}. Knuckles/palm segments under 5 mm: {atlas['knuckle_under_5mm_count']}. "
                  f"Couplings kept as `Coupled` phalanx modules: {atlas['couplings_kept_total']}. "
                  f"Couplings reported `coupling_not_in_structure`: {atlas['couplings_dropped_not_in_structure_total']}.")
    lines.append("")
    lines.append("## Grammar gap table (real-hand values vs `DEFAULT_DISTRIBUTION`)")
    lines.append("")
    lines.append("| parameter | real min | real median | real max | n | grammar range/choices | frac. outside |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in atlas["grammar_gap_table"]:
        r = row["real"]
        frac = "-" if row["frac_outside"] is None else f"{row['frac_outside']:.2f}"
        lines.append(
            f"| {row['parameter']} | {_fmt(r['min'])} | {_fmt(r['median'])} | {_fmt(r['max'])} | {r['n']} "
            f"| {row['grammar']} | {frac} |"
        )
    lines.append("")
    lines.append(
        "This table is descriptive input for later grammar tuning (widening ranges/choice sets, adding a "
        "rest-bend/lateral-offset production with nonzero default probability, etc.) -- no grammar file "
        "changed as part of this experiment."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="project-notes/grammar/experiments/E13_representation")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-configs", type=int, default=N_RANDOM_CONFIGS)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_experiment(
        "e13_representation", run_e13, params={"n_configs": args.n_configs}, seeds=[args.seed],
        out_dir=str(out_dir), allow_dirty=args.allow_dirty,
    )
    per_seed_result = result["per_seed"][0]["result"]
    (out_dir / "summary.md").write_text(render_markdown(per_seed_result))
    print(f"wrote {out_dir / 'result.json'} and {out_dir / 'summary.md'} "
          f"({per_seed_result['n_pass']}/{per_seed_result['n_available']} PASS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
