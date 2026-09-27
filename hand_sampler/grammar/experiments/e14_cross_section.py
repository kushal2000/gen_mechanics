"""E14: cross-section study (representation-check plan, item 4) -- secondary
to E13, no change to ``geometry.py`` or simulator code.

Martin's question: geometry stays capsules for now; would a single
rounded-rectangle cross-section ("one cross-section fits most") represent
real finger links better, and would it complicate RL/contact math? This
script answers the geometric half only (fit quality); the contact-cost half
is a documentation-only desk study, written up in
``project-notes/grammar/cross-section-study.md``, not run here.

For each hand with a loadable collision (or, failing that, visual) mesh:

1. Reuse ``hand_sampler.grammar.adapters.urdf.load_urdf`` +
   ``adapters.projection.project_to_derivation`` (exactly as E13) to get the
   digit/phalanx structure and ``name_map``. This module never re-derives the
   palm/digit-chain logic itself -- it only reads ``name_map`` (which joint
   became which ``d{digit}p{i}_j``/``..._tip``) to recover, per digit, the
   ORIGINAL body chain order and tip body, then computes each phalanx body's
   own canonical +z (the direction to the next joint or the fingertip) and
   its joint-to-joint length directly from the original model's zero-config
   forward kinematics -- the same arithmetic ``projection.py``'s internal
   ``frame_digit`` uses, just re-derived here from public outputs since
   ``project_to_derivation`` does not expose its internal per-body frames.
2. Only finger (digit) links with length >= 10 mm are sectioned (per the
   plan). Palm bodies and the root are never sectioned.
3. Loads that link's collision (or visual) mesh via
   ``grammar_bench.refgen.mesh_sections.load_link_mesh`` (``yourdfpy`` parse
   + ``trimesh`` load; both lazily imported -- this module imports fine on
   the system Python with neither installed, it just marks every hand
   ``mesh_unavailable``), slices it at 25/50/75% of the joint-to-joint length
   in the link's own local frame, and fits a circle (two ways) and a
   rounded rectangle to the largest closed section loop.
4. Pools every section into the template search (one normalised
   ``(h/w, r/h)`` shape, grid-searched; one size per link) and the capsule
   comparisons (per-link / per-hand / global radius).

Two Python interpreters matter here (see the module's own path handling
below, and ``project-notes/grammar/plan-representation-check.md`` item 4):
system ``python3`` has numpy but no ``trimesh``/``yourdfpy`` (still imports
and runs this module -- every hand just comes back ``mesh_unavailable``,
which is what the test suite exercises); the ``piper`` conda env has both,
and is how this experiment is actually run for real hands (its meshes live
outside the repo, at ``~/karma/...``, hence ``local-only:`` reasons when
absent).

Stdlib + numpy + this project's own ``hand_sampler`` package + (lazily)
``trimesh``/``yourdfpy``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..adapters.projection import ProjectionFailure, project_to_derivation
from ..adapters.urdf import load_urdf
from ..fk import forward_kinematics
from .runner import register, run_experiment

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = REPO_ROOT / "hand_sampler" / "grammar_bench"
MANIFEST_PATH = BENCH_DIR / "manifest.json"


def _load_manifest() -> dict:
    """Deliberately NOT imported from ``e13_representation`` (which this
    module otherwise mirrors, e.g. E13's own docstring on the same pattern):
    ``e13_representation`` imports ``.runner``, and ``runner`` imports THIS
    module for its ``register()`` side effect, so importing ``.e13_
    representation`` from here is a genuine circular import when this module
    is executed directly (``python3 -m ...e14_cross_section``) -- confirmed
    by reproducing it while writing this. A three-line duplicate is cheaper
    than restructuring either module."""
    return json.loads(MANIFEST_PATH.read_text())


MIN_LINK_LENGTH_M = 0.010
FRACTIONS = (0.25, 0.50, 0.75)
TOL_MM = (2.0, 3.0)

# The plan's target hand list for this study (manifest ids). Shadow is
# skipped: its only locally-resolvable kinematic URDF
# (``shadow_right_local``) references DAE meshes exclusively (the original
# Shadow ROS package's STL collision meshes use a different link layout that
# xacro reconciles, out of scope for this time-boxed study) -- "Skip
# DAE-only hands and say so" (plan item 4's data note).
TARGET_HAND_IDS = (
    "sharpa_left_on_iiwa14", "allegro_right", "leap_right", "inspire_right",
    "barrett_bh", "dclaw", "ability_right", "xhand_right", "wuji_right",
    "tesollo_dg5f_right", "orca_right",
)
SKIPPED_HANDS = {
    "shadow_right_local": "DAE-only: shadowhand_right_local URDF references only "
                           "package://sr_description/meshes/components/*.dae; the original "
                           "Shadow package's STL collision meshes use a different link-naming "
                           "layout (reconciled by xacro macros this study does not run). "
                           "Skipped and noted, per plan item 4.",
}


# --------------------------------------------------------------------------
# Hand source resolution: the ORIGINAL full "as downloaded" URDF (not
# manifest ``fixture_path`` copies, which omit the mesh files that sit
# alongside the real URDF on disk) -- mirrors E13's ``_resolve_hand`` minus
# the fixture_path branch, since a mesh study needs the real mesh tree.
# --------------------------------------------------------------------------


def _resolve_source(hand: dict, manifest: dict) -> Tuple[Optional[Path], str, Optional[str]]:
    source_path = hand.get("source_path")
    if isinstance(source_path, str) and source_path.startswith("REPO:"):
        p = REPO_ROOT / source_path[len("REPO:"):]
        if p.is_file():
            return p, "available", None
        return None, "unavailable", f"REPO path listed in manifest but missing on disk: {source_path}"
    if isinstance(source_path, str) and source_path:
        source_root = manifest.get("source_root")
        if source_root:
            p = Path(source_root) / source_path
            if p.is_file():
                return p, "available", None
        return None, "unavailable", f"local-only:{hand['id']} source not present at manifest source_root on this machine"
    return None, "unavailable", "no source_path in manifest"


def _cases(manifest: dict, hand_ids: Optional[Sequence[str]] = None) -> List[Tuple[dict, Optional[Path], str, Optional[str]]]:
    by_id = {h["id"]: h for h in manifest["hands"]}
    ids = list(hand_ids) if hand_ids is not None else list(TARGET_HAND_IDS)
    out = []
    for hid in ids:
        if hid in SKIPPED_HANDS:
            out.append(({"id": hid}, None, "skipped", SKIPPED_HANDS[hid]))
            continue
        hand = by_id.get(hid)
        if hand is None:
            out.append(({"id": hid}, None, "unavailable", f"{hid!r} not in manifest"))
            continue
        path, availability, reason = _resolve_source(hand, manifest)
        out.append((hand, path, availability, reason))
    return out


# --------------------------------------------------------------------------
# Recover per-digit body chains + tip bodies from project_to_derivation's
# public ``name_map`` (join name -> "d{digit}p{i+1}_j" / tip body ->
# "d{digit}p{count}_tip") -- see module docstring point 1.
# --------------------------------------------------------------------------

_PHALANX_JOINT_RE = re.compile(r"^d(\d+)p(\d+)_j$")
_TIP_RE = re.compile(r"^d(\d+)p(\d+)_tip$")


def _digit_chains(model, name_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    joint_by_name = {j.name: j for j in model.joints}
    chains: Dict[str, Dict[int, str]] = {}
    for orig_name, derived_name in name_map.items():
        m = _PHALANX_JOINT_RE.match(derived_name)
        if m:
            digit_id, p = m.group(1), int(m.group(2))
            j = joint_by_name.get(orig_name)
            if j is not None:
                chains.setdefault(digit_id, {})[p - 1] = j.child
    tips: Dict[str, str] = {}
    for orig_body, derived_name in name_map.items():
        m = _TIP_RE.match(derived_name)
        if m:
            tips[m.group(1)] = orig_body
    out = {}
    for digit_id, by_p in chains.items():
        bodies = [by_p[i] for i in sorted(by_p)]
        out[digit_id] = {"bodies": bodies, "tip": tips.get(digit_id)}
    return out


def _finger_links(model, pr) -> List[Dict[str, Any]]:
    """One entry per (digit body, length in metres, canonical +z in that
    body's OWN local frame) for every finger link with length >= 10 mm."""
    T0 = forward_kinematics(model, {})
    chains = _digit_chains(model, pr.name_map)
    out = []
    for digit_id, info in chains.items():
        bodies, tip = info["bodies"], info["tip"]
        n = len(bodies)
        for i, body in enumerate(bodies):
            if i < n - 1:
                next_pt = T0[bodies[i + 1]][:3, 3]
            elif tip is not None and tip in T0:
                next_pt = T0[tip][:3, 3]
            else:
                out.append({"digit_id": digit_id, "p": i, "body": body, "skipped": "fingertip_undefined"})
                continue
            world_vec = next_pt - T0[body][:3, 3]
            length = float(np.linalg.norm(world_vec))
            if length < MIN_LINK_LENGTH_M:
                out.append({"digit_id": digit_id, "p": i, "body": body, "length_m": length, "skipped": "length<10mm"})
                continue
            z_world = world_vec / length
            z_local = T0[body][:3, :3].T @ z_world
            out.append({
                "digit_id": digit_id, "p": i, "body": body, "length_m": length,
                "z_local": z_local.tolist(), "skipped": None,
            })
    return out


# --------------------------------------------------------------------------
# Top-level driver (runner.py per-seed function)
# --------------------------------------------------------------------------


def run_e14(seed: int, hand_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    manifest = _load_manifest()
    per_hand: List[Dict[str, Any]] = []

    # mesh_sections is imported here (not at module top) so this module
    # itself always imports cleanly on the system Python (no trimesh);
    # ``sys.path`` already has the repo root (this module is only ever run
    # via ``python3 -m hand_sampler...`` or pytest from the repo root).
    from ...grammar_bench.refgen import mesh_sections as ms

    all_sections: List[Dict[str, Any]] = []  # flat list, one per (hand, digit, p, frac)

    for hand, path, availability, reason in _cases(manifest, hand_ids=hand_ids):
        hand_id = hand["id"]
        if availability != "available":
            per_hand.append({"id": hand_id, "availability": availability, "reason": reason,
                              "n_links": 0, "n_sections": 0})
            continue

        result: Dict[str, Any] = {"id": hand_id, "availability": "available", "reason": None,
                                   "n_links": 0, "n_sections": 0, "links": []}
        try:
            imported = load_urdf(path, hand_root=hand.get("hand_root"))
        except Exception as exc:  # noqa: BLE001
            result.update(availability="unavailable", reason=f"load_urdf failed: {type(exc).__name__}: {exc}")
            per_hand.append(result)
            continue
        model = imported.model
        try:
            pr = project_to_derivation(model, palm_joints=hand.get("palm_joints") or (), tip_frames=hand.get("tip_frames") or {})
        except ProjectionFailure as exc:
            result.update(availability="unavailable", reason=f"projection failed: {exc}")
            per_hand.append(result)
            continue

        all_links = _finger_links(model, pr)
        finger_links = [fl for fl in all_links if not fl["skipped"]]
        result["n_links_considered"] = len(all_links)
        result["n_links_dropped"] = [{"body": fl["body"], "digit_id": fl["digit_id"], "p": fl["p"], "reason": fl["skipped"]}
                                      for fl in all_links if fl["skipped"]]

        try:
            import yourdfpy
        except ImportError as exc:
            result.update(availability="mesh_unavailable",
                           reason=f"local-only:{hand_id} yourdfpy/trimesh not importable on this interpreter: {exc}")
            per_hand.append(result)
            continue

        try:
            robot = yourdfpy.URDF.load(
                str(path), load_meshes=False, load_collision_meshes=False,
                build_scene_graph=False, build_collision_scene_graph=False,
            )
        except Exception as exc:  # noqa: BLE001
            result.update(availability="mesh_unavailable", reason=f"local-only:{hand_id} yourdfpy.URDF.load failed: {type(exc).__name__}: {exc}")
            per_hand.append(result)
            continue

        T0 = forward_kinematics(model, {})
        fixed_children: Dict[str, List[str]] = {}
        for j in model.joints:
            if j.type == "fixed":
                fixed_children.setdefault(j.parent, []).append(j.child)

        for fl in finger_links:
            body = fl["body"]
            link_entry: Dict[str, Any] = {"digit_id": fl["digit_id"], "p": fl["p"], "body": body,
                                           "length_mm": fl["length_m"] * 1000.0, "sections": [], "mesh_kind": None,
                                           "skipped": None}
            if body not in robot.link_map:
                link_entry["skipped"] = f"link {body!r} not in yourdfpy link_map"
                result["links"].append(link_entry)
                continue
            try:
                # ORCA-style zero-mass "*_jointbody" links carry no geometry
                # of their own; fall back through fixed-joint descendants
                # (design_space.py's ``_nearest_geometry`` pattern).
                mesh, kind, mesh_body = ms.load_mesh_with_fixed_fallback(path, robot, fixed_children, T0, body)
            except ms.MeshUnavailable as exc:
                link_entry["skipped"] = f"local-only:{hand_id}/{body} {exc}"
                result["links"].append(link_entry)
                continue
            link_entry["mesh_kind"] = kind
            link_entry["mesh_body"] = mesh_body

            z_local = np.asarray(fl["z_local"], dtype=float)
            length_m = fl["length_m"]
            for frac in FRACTIONS:
                plane_origin = frac * length_m * z_local
                loop = ms.slice_mesh(mesh, plane_origin, z_local)
                if loop is None or len(loop) < 3:
                    link_entry["sections"].append({"frac": frac, "skipped": "no_intersection"})
                    continue
                xy = ms.points_to_axis_plane(loop, plane_origin, z_local)
                angle = ms.principal_axis_angle(xy)
                xy_p = ms.rotate_xy(xy, angle)
                # convention: w along the larger-extent axis
                ext = np.abs(xy_p).max(axis=0)
                if ext[1] > ext[0]:
                    xy_p = xy_p[:, ::-1]
                circ_eq = ms.fit_circle_equal_area(xy)
                circ_95 = ms.fit_circle_p95(xy)
                rr = ms.fit_rounded_rect(xy_p)
                sec = {
                    "frac": frac, "skipped": None, "n_boundary_points": int(len(xy)),
                    "circle_equal_area": {"radius_mm": circ_eq.radius * 1000.0, "mean_err_mm": circ_eq.mean_err * 1000.0,
                                          "max_err_mm": circ_eq.max_err * 1000.0, "area_ratio": circ_eq.area_ratio},
                    "circle_p95": {"radius_mm": circ_95.radius * 1000.0, "mean_err_mm": circ_95.mean_err * 1000.0,
                                   "max_err_mm": circ_95.max_err * 1000.0, "area_ratio": circ_95.area_ratio},
                    "rounded_rect": {
                        "w_mm": rr.w * 1000.0, "h_mm": rr.h * 1000.0, "r_mm": rr.r * 1000.0,
                        "h_over_w": (rr.h / rr.w) if rr.w > 0 else None,
                        "r_over_minwh": (rr.r / min(rr.w, rr.h)) if min(rr.w, rr.h) > 0 else None,
                        "mean_err_mm": rr.mean_err * 1000.0, "max_err_mm": rr.max_err * 1000.0,
                    },
                    "_xy_p_m": xy_p,  # kept in-process only for the template search; stripped before json dump
                }
                link_entry["sections"].append(sec)
                all_sections.append({"hand_id": hand_id, "digit_id": fl["digit_id"], "p": fl["p"],
                                     "link_key": f"{hand_id}/{fl['digit_id']}/{fl['p']}", "frac": frac, "xy_p_m": xy_p})
            result["links"].append(link_entry)

        result["n_links"] = sum(1 for l in result["links"] if l["skipped"] is None)
        result["n_sections"] = sum(len([s for s in l["sections"] if not s.get("skipped")]) for l in result["links"])
        per_hand.append(result)

    template_search = _run_template_search(all_sections)

    # strip the numpy payload before returning (result.json must be JSON-serializable)
    for h in per_hand:
        for l in h.get("links", []):
            for s in l.get("sections", []):
                s.pop("_xy_p_m", None)

    n_available = sum(1 for h in per_hand if h["availability"] == "available")
    n_links_total = sum(h.get("n_links", 0) for h in per_hand)
    n_sections_total = sum(h.get("n_sections", 0) for h in per_hand)

    return {
        "n_hands_total": len(per_hand), "n_available": n_available,
        "n_links_total": n_links_total, "n_sections_total": n_sections_total,
        "_per_hand": per_hand, "_template_search": template_search,
    }


# --------------------------------------------------------------------------
# Template search + capsule comparisons (across all pooled sections)
# --------------------------------------------------------------------------


def _run_template_search(all_sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    from ...grammar_bench.refgen import mesh_sections as ms

    if not all_sections:
        return {"n_sections": 0, "note": "no sections available on this interpreter/machine"}

    by_link: Dict[str, List[np.ndarray]] = {}
    for s in all_sections:
        by_link.setdefault(s["link_key"], []).append(s["xy_p_m"])
    by_hand: Dict[str, List[np.ndarray]] = {}
    for s in all_sections:
        by_hand.setdefault(s["hand_id"], []).append(s["xy_p_m"])
    all_xy = [s["xy_p_m"] for s in all_sections]

    # ---- (a) best single template: coarse outer grid over (aspect=h/w,
    # roundedness=r/h), each shape scored by the pooled 80th-percentile max
    # error using each link's OWN best size for that shape.
    aspect_grid = np.linspace(0.3, 1.0, 5)
    round_grid = np.linspace(0.05, 0.45, 5)
    best = {"aspect": None, "roundedness": None, "p80_mm": float("inf")}
    for aspect in aspect_grid:
        for roundedness in round_grid:
            per_section_max = []
            for link_key, sections in by_link.items():
                w_opt, _ = ms.best_template_size(sections, float(aspect), float(roundedness))
                h_opt = aspect * w_opt
                r_opt = min(roundedness * h_opt, 0.5 * min(w_opt, h_opt))
                for s in sections:
                    per_section_max.append(float(np.abs(ms.sdf_rounded_rect(s, w_opt, h_opt, r_opt)).max()))
            p80 = float(np.percentile(per_section_max, 80.0)) if per_section_max else float("inf")
            if p80 < best["p80_mm"]:
                best = {"aspect": float(aspect), "roundedness": float(roundedness), "p80_mm": p80 * 1000.0}

    # Recompute the winning shape's per-link sizes + per-section errors at
    # full precision (this is cheap: one shape, not the whole grid).
    aspect, roundedness = best["aspect"], best["roundedness"]
    template_errs_mm: List[float] = []
    link_sizes = {}
    for link_key, sections in by_link.items():
        w_opt, _ = ms.best_template_size(sections, aspect, roundedness)
        h_opt, r_opt = aspect * w_opt, min(roundedness * (aspect * w_opt), 0.5 * min(w_opt, aspect * w_opt))
        link_sizes[link_key] = {"w_mm": w_opt * 1000.0, "h_mm": h_opt * 1000.0, "r_mm": r_opt * 1000.0}
        for s in sections:
            template_errs_mm.append(float(np.abs(ms.sdf_rounded_rect(s, w_opt, h_opt, r_opt)).max()) * 1000.0)

    # ---- (b) per-link-radius capsule, (c) one radius per hand, (d) one
    # global radius -- each scored the same way (pooled per-section max err).
    def _capsule_errs(groups: Dict[str, List[np.ndarray]]) -> Tuple[List[float], Dict[str, float]]:
        errs, radii = [], {}
        for key, sections in groups.items():
            r, _ = ms.best_capsule_radius(sections)
            radii[key] = r * 1000.0
            for s in sections:
                errs.append(float(np.abs(ms.sdf_circle(s, r)).max()) * 1000.0)
        return errs, radii

    per_link_errs, per_link_radii = _capsule_errs(by_link)
    per_hand_errs, per_hand_radii = _capsule_errs(by_hand)
    global_r, _ = ms.best_capsule_radius(all_xy)
    global_errs = [float(np.abs(ms.sdf_circle(s, global_r)).max()) * 1000.0 for s in all_xy]

    def _fracs(errs: List[float]) -> Dict[str, float]:
        arr = np.asarray(errs)
        return {f"within_{tol:g}mm": float(np.mean(arr <= tol)) for tol in TOL_MM}

    return {
        "n_sections": len(all_sections), "n_links": len(by_link), "n_hands": len(by_hand),
        "best_template": {"aspect_h_over_w": aspect, "roundedness_r_over_h": roundedness,
                           "pooled_p80_max_err_mm": best["p80_mm"], "link_sizes": link_sizes,
                           "fractions_within_tol": _fracs(template_errs_mm)},
        "per_link_radius_capsule": {"radii_mm": per_link_radii, "fractions_within_tol": _fracs(per_link_errs)},
        "per_hand_radius_capsule": {"radii_mm": per_hand_radii, "fractions_within_tol": _fracs(per_hand_errs)},
        "global_radius_capsule": {"radius_mm": global_r * 1000.0, "fractions_within_tol": _fracs(global_errs)},
    }


register("e14_cross_section", run_e14)


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def _fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))


def render_markdown(result: Dict[str, Any]) -> str:
    per_hand = result["_per_hand"]
    ts = result["_template_search"]
    lines = ["# E14: cross-section study (capsule vs. rounded rectangle)", ""]
    lines.append(
        f"{result['n_available']}/{result['n_hands_total']} target hands available; "
        f"{result['n_links_total']} finger links sectioned, {result['n_sections_total']} sections total "
        f"(25/50/75% of joint-to-joint length, links >= 10 mm)."
    )
    lines.append("")
    lines.append("## Per-hand availability")
    lines.append("")
    lines.append("| hand | availability | links | sections | reason |")
    lines.append("|---|---|---|---|---|")
    for h in per_hand:
        lines.append(f"| {h['id']} | {h['availability']} | {h.get('n_links', 0)} | {h.get('n_sections', 0)} | {h.get('reason') or '-'} |")
    lines.append("")

    if ts.get("n_sections", 0) == 0:
        lines.append(f"No sections available on this interpreter/machine: {ts.get('note')}")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Template search")
    lines.append("")
    bt = ts["best_template"]
    lines.append(
        f"Best single template: h/w = {bt['aspect_h_over_w']:.2f}, r/h = {bt['roundedness_r_over_h']:.2f}, "
        f"pooled 80th-percentile max boundary error = {bt['pooled_p80_max_err_mm']:.2f} mm "
        f"(one size per link, {ts['n_links']} links, {ts['n_sections']} sections, {ts['n_hands']} hands)."
    )
    lines.append("")
    lines.append("| fit | within 2 mm | within 3 mm |")
    lines.append("|---|---|---|")
    for label, block in (
        ("(a) best single template", bt),
        ("(b) per-link-radius capsule", ts["per_link_radius_capsule"]),
        ("(c) one radius per hand capsule", ts["per_hand_radius_capsule"]),
        ("(d) one radius for all hands", ts["global_radius_capsule"]),
    ):
        f = block["fractions_within_tol"]
        lines.append(f"| {label} | {f.get('within_2mm', float('nan')):.2f} | {f.get('within_3mm', float('nan')):.2f} |")
    lines.append("")
    gr = ts["global_radius_capsule"]["radius_mm"]
    lines.append(f"Global single capsule radius (all hands): {gr:.2f} mm.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="project-notes/grammar/experiments/E14_cross_section")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--hand-ids", nargs="*", default=None)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_experiment(
        "e14_cross_section", run_e14, params={"hand_ids": args.hand_ids}, seeds=[args.seed],
        out_dir=str(out_dir), allow_dirty=args.allow_dirty,
    )
    per_seed_result = result["per_seed"][0]["result"]
    (out_dir / "summary.md").write_text(render_markdown(per_seed_result))
    print(f"wrote {out_dir / 'result.json'} and {out_dir / 'summary.md'} "
          f"({per_seed_result['n_available']}/{per_seed_result['n_hands_total']} hands available, "
          f"{per_seed_result['n_sections_total']} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
