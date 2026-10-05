"""Commercial hands from `grammar_bench/manifest.json`, expressed in the grammar.

Pipeline (E13, `hand_sampler/grammar/experiments/e13_representation.py`):
`load_urdf(hand_root)` -> `project_to_derivation(palm_joints, tip_frames)` ->
`derive`. The fidelity readout replays E13's `check_hand` measurement (zero
plus 64 random configurations, seed 20260926, every original movable joint
sampled independently, joint frames compared through `name_map` and
`root_transform`) without its Pinocchio export cross-check, which spawns a
separate interpreter. `tests/test_commercial.py` checks the numbers equal
`check_hand`'s.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hand_sampler.grammar.adapters.projection import ProjectionFailure, ProjectionResult, project_to_derivation
from hand_sampler.grammar.adapters.urdf import ImportResult, load_urdf
from hand_sampler.grammar.derive import derive, validate_derivation
from hand_sampler.grammar.experiments import e13_representation as e13
from hand_sampler.grammar.fk import forward_kinematics, pose_to_matrix
from hand_sampler.grammar.kinematics import KinematicModel, MOVABLE_TYPES

from .envload import REPO_ROOT, load_env_modules

MANIFEST_PATH = REPO_ROOT / "hand_sampler" / "grammar_bench" / "manifest.json"
BENCH_DIR = MANIFEST_PATH.parent


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


@dataclass(frozen=True)
class HandEntry:
    id: str
    split: str
    family: str
    notes: str
    kin_path: Optional[Path]       # the file E13 projects (fixture copy first)
    availability: str
    reason: Optional[str]
    mesh_path: Optional[Path]      # a copy that ships meshes (source_root / REPO / absolute path)
    hand_root: Optional[str]
    palm_joints: Tuple[str, ...]
    tip_frames: Dict[str, str]
    sha256: Optional[str]

    @property
    def label(self) -> str:
        tag = "" if self.availability == "available" else f" [{self.availability}]"
        return f"{self.id} ({self.split}){tag}"


def _mesh_copy(hand: dict, manifest: dict) -> Optional[Path]:
    sp = hand.get("source_path")
    if not isinstance(sp, str) or not sp:
        return None
    if sp.startswith("REPO:"):
        p = REPO_ROOT / sp[len("REPO:"):]
    else:
        p = Path(manifest.get("source_root") or "/") / sp  # absolute sp overrides source_root
    return p if p.is_file() else None


def list_hands(include_excluded: bool = False) -> List[HandEntry]:
    manifest = load_manifest()
    out = []
    for h in manifest["hands"]:
        if h.get("split") == "excluded" and not include_excluded:
            continue
        path, availability, reason = e13._resolve_hand(h, manifest)
        out.append(HandEntry(
            id=h["id"], split=h.get("split", ""), family=h.get("family", ""), notes=h.get("notes", ""),
            kin_path=path, availability=availability, reason=reason, mesh_path=_mesh_copy(h, manifest),
            hand_root=h.get("hand_root"), palm_joints=tuple(h.get("palm_joints") or ()),
            tip_frames=dict(h.get("tip_frames") or {}), sha256=h.get("sha256"),
        ))
    return out


def hand_entry(hand_id: str) -> HandEntry:
    for h in list_hands(include_excluded=True):
        if h.id == hand_id:
            return h
    raise KeyError(f"hand {hand_id!r} not in manifest")


# --------------------------------------------------------------------------
# Fidelity (E13 check_hand's measurement, Pinocchio part omitted)
# --------------------------------------------------------------------------


def projection_fidelity(model: KinematicModel, pr: ProjectionResult, derived: KinematicModel,
                        seed: int = e13.SEED, n_configs: int = e13.N_RANDOM_CONFIGS) -> Dict[str, Any]:
    movable = [j for j in model.joints if j.type in MOVABLE_TYPES]
    derived_joint = {j.name: j for j in derived.joints}
    derived_frames = {f.name: f for f in derived.frames}
    n_orig = len(movable)
    n_der = sum(1 for j in derived.joints if j.type in MOVABLE_TYPES)
    rng = np.random.default_rng(seed)
    max_pos = max_axis = max_tip = 0.0
    n_tip = 0
    per_joint_pos: Dict[str, float] = {}
    RT = pr.root_transform
    for trial in range(n_configs + 1):
        q_orig = {}
        for j in movable:
            if trial == 0:
                q_orig[j.name] = 0.0
            elif j.type == "continuous":
                q_orig[j.name] = float(rng.uniform(-math.pi, math.pi))
            else:
                lo, hi = j.limits
                q_orig[j.name] = float(rng.uniform(lo, hi))
        T_o_all = forward_kinematics(model, q_orig)
        q_der = {pr.name_map[jn]: v for jn, v in q_orig.items() if jn in pr.name_map}
        T_d_all = forward_kinematics(derived, q_der)
        for j in movable:
            djn = pr.name_map.get(j.name)
            if djn is None:
                continue
            body = djn[:-2]
            if body not in T_d_all:
                continue
            T_o, T_d = T_o_all[j.child], RT @ T_d_all[body]
            pe = float(np.linalg.norm(T_o[:3, 3] - T_d[:3, 3]))
            max_pos = max(max_pos, pe)
            per_joint_pos[j.name] = max(per_joint_pos.get(j.name, 0.0), pe)
            ao = T_o[:3, :3] @ np.asarray(j.axis, dtype=float)
            ad = T_d[:3, :3] @ np.asarray(derived_joint[djn].axis, dtype=float)
            max_axis = max(max_axis, e13._rotation_angle_between(ao, ad))
        for orig_body, dname in pr.name_map.items():
            if not dname.endswith("_tip") or orig_body not in T_o_all:
                continue
            fr = derived_frames.get(dname)
            if fr is None:
                continue
            T_tip = RT @ (T_d_all[fr.body] @ pose_to_matrix(fr.pose))
            max_tip = max(max_tip, float(np.linalg.norm(T_o_all[orig_body][:3, 3] - T_tip[:3, 3])))
            n_tip += 1
    conserved = n_orig == n_der
    tip_ok = (max_tip <= e13.POS_TOL_M) if n_tip > 0 else True
    worst = max(per_joint_pos.items(), key=lambda kv: kv[1])[0] if per_joint_pos else None
    return {
        "max_pos_mm": max_pos * 1000.0,
        "max_axis_deg": math.degrees(max_axis),
        "max_tip_mm": (max_tip * 1000.0) if n_tip > 0 else None,
        "n_tip_compared": n_tip,
        "n_movable_orig": n_orig,
        "n_movable_der": n_der,
        "joint_count_conserved": bool(conserved),
        "n_independent_orig": n_orig - len(model.couplings),
        "n_independent_der": n_der - len(derived.couplings),
        "worst_joint": worst,
        "passed": bool(max_pos <= e13.POS_TOL_M and max_axis <= e13.AXIS_TOL_RAD and tip_ok and conserved),
        "n_configs": n_configs + 1,
        "seed": seed,
    }


# --------------------------------------------------------------------------
# A loaded commercial hand
# --------------------------------------------------------------------------


@dataclass
class CommercialHand:
    entry: HandEntry
    imported: Optional[ImportResult] = None
    projection: Optional[ProjectionResult] = None
    derived: Optional[KinematicModel] = None
    fidelity: Optional[Dict[str, Any]] = None
    fit: Optional[Tuple[bool, Tuple[str, ...]]] = None   # admit(check_overlap=False)
    error: Optional[str] = None
    sha_ok: Optional[bool] = None
    derived_to_orig_joint: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.derived is not None

    @property
    def capsule_radius_m(self) -> Optional[float]:
        if self.projection is None:
            return None
        hand = next(s for s in self.projection.derivation.steps if s.path == "hand")
        return float(hand.params["capsule_radius_m"])

    def approximations(self) -> List[str]:
        """Everything the projection dropped, merged or approximated, plus the
        importer's own losses, as readable lines."""
        lines: List[str] = []
        r = self.capsule_radius_m
        if r is not None:
            lines.append(f"capsule radius {r * 1000:.1f} mm for every link (projection constant, "
                         f"`project_to_derivation`; real link widths vary)")
        if self.projection is None or self.imported is None:
            return lines
        rep = self.projection.report
        if rep.get("merged_fixed_joints"):
            lines.append(f"{len(rep['merged_fixed_joints'])} fixed joint(s) merged into their parent: "
                         + ", ".join(rep["merged_fixed_joints"]))
        if rep.get("coupling_as_independent"):
            lines.append(f"{len(rep['coupling_as_independent'])} coupled (mimic) joint(s) made independent motors: "
                         + ", ".join(rep["coupling_as_independent"]))
        if rep.get("fingertip_undefined"):
            lines.append("no fingertip link beyond the last joint (tip length 0) for: "
                         + ", ".join(rep["fingertip_undefined"]))
        if rep.get("palm_bodies"):
            lines.append("articulated palm bodies (manifest `palm_joints`): " + ", ".join(rep["palm_bodies"]))
        loss = self.imported.losses
        if loss.dropped_above_hand_root:
            lines.append(f"{len(loss.dropped_above_hand_root)} link(s) above hand_root dropped: "
                         + ", ".join(loss.dropped_above_hand_root))
        if loss.couplings_cut:
            lines.append("couplings cut by hand_root: " + ", ".join(loss.couplings_cut))
        if loss.non_unit_axes:
            lines.append("non-unit joint axes in the URDF: " + ", ".join(loss.non_unit_axes))
        nc = len(self.imported.model.couplings)
        if nc:
            lines.append(f"URDF declares {nc} mimic coupling(s); the projection keeps none (one motor per joint)")
        lines.append("visual/collision geometry is not projected: links become capsules, palm bodies become "
                     "nearest-spine hull cells")
        return lines

    def orig_link_poses(self, u_derived: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Original URDF link poses expressed in the DERIVED root frame, with
        every original joint driven by its projected counterpart's value."""
        if self.imported is None or self.projection is None:
            return {}
        q_orig = {orig: float(u_derived.get(der, 0.0)) for der, orig in self.derived_to_orig_joint.items()}
        T = forward_kinematics(self.imported.model, q_orig)
        inv_rt = np.linalg.inv(self.projection.root_transform)
        return {name: inv_rt @ M for name, M in T.items()}


def load_commercial(hand_id: str, *, with_fidelity: bool = True) -> CommercialHand:
    entry = hand_entry(hand_id)
    ch = CommercialHand(entry=entry)
    if entry.availability != "available" or entry.kin_path is None:
        ch.error = f"{entry.availability}: {entry.reason}"
        return ch
    if entry.sha256:
        ch.sha_ok = hashlib.sha256(entry.kin_path.read_bytes()).hexdigest() == entry.sha256
    try:
        ch.imported = load_urdf(entry.kin_path, hand_root=entry.hand_root)
    except Exception as exc:  # noqa: BLE001
        ch.error = f"import failed: {type(exc).__name__}: {exc}"
        return ch
    try:
        ch.projection = project_to_derivation(ch.imported.model, palm_joints=entry.palm_joints,
                                              tip_frames=entry.tip_frames)
    except ProjectionFailure as exc:
        ch.error = f"projection failed: {exc}"
        return ch
    issues = validate_derivation(ch.projection.derivation)
    if issues:
        ch.error = f"validate_derivation issues: {issues}"
        return ch
    ch.derived = derive(ch.projection.derivation)
    ch.derived_to_orig_joint = {der: orig for orig, der in ch.projection.name_map.items()
                                if der.endswith("_j")}
    if with_fidelity:
        ch.fidelity = projection_fidelity(ch.imported.model, ch.projection, ch.derived)
    res = load_env_modules().grammar_envelope.admit(ch.derived, check_overlap=False)
    ch.fit = (bool(res.ok), tuple(res.reasons))
    return ch
