"""Stable-grasp cache for the ``anyrotate`` profile (Kit-free: numpy, torch, stdlib).

Sources:
- HORA (H. Qi et al., "In-Hand Object Rotation via Rapid Motor Adaptation",
  CoRL 2022; github.com/HaozhiQi/hora, ``hora/tasks/allegro_hand_grasp.py``,
  ``scripts/gen_grasp.sh``). Each candidate starts from the canonical grasp
  pose plus 0.25 x U(-1, 1) rad per joint, the object at a fixed point in
  the hand, and the PD targets held (zero actions) for 50 control steps at
  15 Hz with gravity on. A candidate is reset as soon as any fingertip is
  more than 0.1 m from the object, fewer than 2 fingertips touch it, or it
  falls more than 1.5 cm; the joint positions and object pose of the
  survivors at the end of the hold are saved (50k per object scale). At
  reset, a training episode copies one saved state, with no added noise
  (joint positions and PD targets both set to the saved joint positions),
  and the pose penalty is measured against that episode's initial pose.
- AnyRotate (M. Yang et al., CoRL 2024, arXiv 2405.07391v3, App. C): the
  object 13 cm above the hand base in a random orientation, the hand at a
  canonical grasp pose + U(-0.3, 0.3) rad, 120 steps (6 s) while gravity
  turns through the hand's +-x, +-y, +-z axes; saved when more than 2
  fingertips and no other hand body touch the object, the total
  fingertip-to-object distance is below 0.2 and the object stays stable
  (10000 grasps per object).

Adaptation to variable hands (documented in the evolution README):
- Canonical pose: HORA's for allegro_right (``CANONICAL_GRASP_POSES``);
  otherwise the hand's calibrated default pose (a grammar design's
  ``palm_up`` curl). A fraction of candidates instead curls every joint to
  a random fraction of its range (``sample_joint_candidates``), because a
  design has no hand-made grasp pose.
- Object placement: the hand's spawn point, or the centroid of its
  fingertips after the hand has settled at the candidate's targets.
- Acceptance (``stable_mask``): HORA's tests on the whole hold (object
  displacement, every fingertip within reach, >= 2 tip contacts) plus final
  linear and angular speed bounds; AnyRotate's no-non-tip-contact and
  total-distance tests and its gravity sweep are options.
- Stored per design: the settled joint positions, the PD targets held
  during the hold (HORA resets the targets to the settled positions, which
  drops the squeeze that held the object), and the object pose in the palm
  frame. Keyed by the design's sha256 (population) or by hand id and a
  sha256 of its calibration (single hand).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "CACHE_SCHEMA", "HORA_CANONICAL_ALLEGRO", "CANONICAL_GRASP_POSES", "GraspSet", "GraspTable",
    "StabilityThresholds", "design_key", "hand_key", "hand_calibration_sha", "sidecar_path",
    "object_signature", "signature_mismatches", "save_cache", "load_cache", "merge_sets", "prune_sets",
    "sample_joint_candidates", "random_quats", "stable_mask", "enclosure_mask", "build_table", "cache_summary", "profile_pose",
    "HORA_LIKE_PROFILE", "HORA_THUMB_PROFILE", "opposition_poses",
]

CACHE_SCHEMA = "anyrotate_grasp_cache/0.1"

# HORA's canonical grasp pose (hora/tasks/allegro_hand_grasp.py), listed in
# HORA's Allegro joint order: index 0-3, thumb 4-7, middle 8-11, ring 12-15
# (the thumb's second joint, 1.163, sits on the Allegro thumb_joint_1
# upper limit). Mapped here onto the drake allegro_right URDF's names
# (joint_0-3 index, joint_4-7 middle, joint_8-11 ring, joint_12-15 thumb).
_HORA_POSE = (0.082, 1.244, 0.265, 0.298, 1.104, 1.163, 0.953, -0.138,
              0.005, 1.096, 0.080, 0.150, 0.029, 1.337, 0.285, 0.317)
_HORA_TO_DRAKE = {"index": (0, 0), "thumb": (4, 12), "middle": (8, 4), "ring": (12, 8)}
HORA_CANONICAL_ALLEGRO: Dict[str, float] = {
    f"joint_{drake + i}": _HORA_POSE[hora + i] for hora, drake in _HORA_TO_DRAKE.values() for i in range(4)}
CANONICAL_GRASP_POSES: Dict[str, Dict[str, float]] = {"allegro_right": HORA_CANONICAL_ALLEGRO}

# HORA's allegro pose as fractions of each joint's range, along a finger:
# index (0.59, 0.80, 0.24, 0.29), middle (0.51, 0.71, 0.13, 0.20), ring
# (0.53, 0.85, 0.24, 0.30). A generic "cage" for any design's fingers
# (joints past the fourth keep 0.25).
HORA_LIKE_PROFILE: Tuple[float, ...] = (0.6, 0.8, 0.25, 0.25, 0.25, 0.25)
# HORA's allegro thumb as range fractions: (1.104, 1.163, 0.953, -0.138) in
# limits (0.263-1.396, -0.105-1.163, -0.189-1.644, -0.162-1.719).
HORA_THUMB_PROFILE: Tuple[float, ...] = (0.742, 1.0, 0.623, 0.013, 0.25, 0.25)


def opposition_poses(joint_valid: np.ndarray, limits: np.ndarray, default: np.ndarray,
                     n_fingers: int = 5, per_finger: int = 6) -> Dict[int, np.ndarray]:
    """``{finger: (32,) pose}``: for every finger with a real chain, the
    design's pose with that finger as the opposing digit (HORA's thumb
    fractions) and every other finger at HORA's finger fractions."""
    out: Dict[int, np.ndarray] = {}
    base = profile_pose(joint_valid, limits, default, HORA_LIKE_PROFILE, n_fingers, per_finger)
    for f in range(n_fingers):
        slots = [f * per_finger + k for k in range(per_finger) if joint_valid[f * per_finger + k]]
        if not slots:
            continue
        q = base.copy()
        for d, s in enumerate(slots):
            lo, hi = limits[s]
            q[s] = lo + HORA_THUMB_PROFILE[min(d, len(HORA_THUMB_PROFILE) - 1)] * (hi - lo)
        out[f] = q
    return out


def profile_pose(joint_valid: np.ndarray, limits: np.ndarray, default: np.ndarray,
                 profile: Sequence[float] = HORA_LIKE_PROFILE, n_fingers: int = 5, per_finger: int = 6
                 ) -> np.ndarray:
    """``(32,)`` pose in envelope-slot order: the d-th real joint of each
    finger at ``lo + profile[d] (hi - lo)``; carrier and ghost slots keep
    ``default``."""
    q = np.array(default, dtype=float, copy=True)
    for f in range(n_fingers):
        d = 0
        for k in range(per_finger):
            s = f * per_finger + k
            if not joint_valid[s]:
                continue
            frac = profile[min(d, len(profile) - 1)]
            lo, hi = limits[s]
            q[s] = lo + frac * (hi - lo)
            d += 1
    return q


# --------------------------------------------------------------------------
# Keys and provenance
# --------------------------------------------------------------------------


def _canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def design_key(sha256: str) -> str:
    """Population design: its entry sha256 (``population_file``)."""
    if not sha256:
        raise ValueError("a population design needs its sha256 to be cached")
    return f"design:{sha256}"


def hand_key(hand_id: str, calibration_sha: str) -> str:
    """Single hand: hand id and the sha256 of its calibration (16 hex)."""
    return f"hand:{hand_id}:{calibration_sha[:16]}"


def hand_calibration_sha(hand_id: str, calibration: Optional[Mapping], pose: Optional[Mapping],
                         collision_from_visuals: bool) -> str:
    """sha256 over everything that places a single hand and shapes its
    colliders: its palm-up calibration entry, its pose-file entry and the
    collider source."""
    payload = {"hand_id": hand_id, "calibration": calibration or {}, "pose": pose or {},
               "collision_from_visuals": bool(collision_from_visuals)}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def sidecar_path(population_path) -> Path:
    """``<dir>/<stem>.grasps.npz`` next to a population file."""
    p = Path(population_path)
    return p.with_name(p.stem + ".grasps.npz")


_SIGNATURE_FIELDS = (
    "object_shape", "box_size", "capsule_radius", "capsule_width", "static_friction", "dynamic_friction",
    "sim_dt", "decimation", "hand_disable_gravity", "hand_stiffness", "hand_damping", "hand_effort_limit",
    "object_contact_offset", "object_rest_offset",
)


def object_signature(a) -> dict:
    """The physical settings a cached grasp depends on, from an
    ``AnyRotateCfg`` (or any object with those attributes)."""
    out = {}
    for name in _SIGNATURE_FIELDS:
        v = getattr(a, name)
        out[name] = round(float(v), 9) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
    if getattr(a, "population_hand_actuator", False):
        out["population_hand_actuator"] = True  # absent when off: older caches keep matching
    if getattr(a, "population_palm_collider", "capsule") != "capsule":
        out["population_palm_collider"] = a.population_palm_collider
    if float(getattr(a, "population_capsule_radius", -1.0)) > 0:
        out["population_capsule_radius"] = round(float(a.population_capsule_radius), 9)
    if getattr(a, "population_projected_pose", False):
        out["population_projected_pose"] = True
    return out


def signature_mismatches(stored: Mapping, current: Mapping) -> List[str]:
    out = []
    for k in sorted(set(stored) | set(current)):
        if stored.get(k) != current.get(k):
            out.append(f"{k}: cache {stored.get(k)!r} vs env {current.get(k)!r}")
    return out


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


INFO_COLUMNS = ("tip_contacts", "nontip_contacts", "object_mass", "max_disp_m")


@dataclass
class GraspSet:
    """One design's (or hand's) stable grasps. Joint arrays are in
    ``joint_names`` order; the object pose is (x, y, z, qw, qx, qy, qz) in
    the palm frame."""

    q: np.ndarray
    q_target: np.ndarray
    obj_pose: np.ndarray
    info: np.ndarray
    joint_names: Tuple[str, ...]
    source: str = ""
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=np.float32).reshape(-1, len(self.joint_names))
        self.q_target = np.asarray(self.q_target, dtype=np.float32).reshape(-1, len(self.joint_names))
        self.obj_pose = np.asarray(self.obj_pose, dtype=np.float32).reshape(-1, 7)
        self.info = np.asarray(self.info, dtype=np.float32).reshape(-1, len(INFO_COLUMNS))
        self.joint_names = tuple(self.joint_names)
        n = self.q.shape[0]
        if not (self.q_target.shape[0] == self.obj_pose.shape[0] == self.info.shape[0] == n):
            raise ValueError("GraspSet arrays disagree on the number of grasps")

    @property
    def n(self) -> int:
        return int(self.q.shape[0])

    @classmethod
    def empty(cls, joint_names: Sequence[str], source: str = "", stats: Optional[dict] = None) -> "GraspSet":
        j = len(joint_names)
        return cls(np.zeros((0, j)), np.zeros((0, j)), np.zeros((0, 7)), np.zeros((0, len(INFO_COLUMNS))),
                   tuple(joint_names), source, dict(stats or {}))

    def truncated(self, k: int) -> "GraspSet":
        return GraspSet(self.q[:k], self.q_target[:k], self.obj_pose[:k], self.info[:k], self.joint_names,
                        self.source, dict(self.stats))


def save_cache(path, sets: Mapping[str, GraspSet], meta: Optional[Mapping] = None) -> Path:
    """Atomic write of ``sets`` (key -> GraspSet) and ``meta`` to an .npz."""
    path = Path(path)
    keys = list(sets)
    doc = dict(meta or {})
    doc["schema"] = CACHE_SCHEMA
    doc["keys"] = keys
    doc["entries"] = {k: {"source": s.source, "joint_names": list(s.joint_names), "n": s.n, "stats": s.stats}
                      for k, s in sets.items()}
    arrays = {"meta": np.array(json.dumps(doc, default=_json_default))}
    for i, k in enumerate(keys):
        s = sets[k]
        arrays[f"k{i}_q"] = s.q
        arrays[f"k{i}_qt"] = s.q_target
        arrays[f"k{i}_obj"] = s.obj_pose
        arrays[f"k{i}_info"] = s.info
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    return path


def load_cache(path) -> Tuple[Dict[str, GraspSet], dict]:
    with np.load(Path(path), allow_pickle=False) as z:
        doc = json.loads(str(z["meta"]))
        if doc.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"{path}: grasp cache schema {doc.get('schema')!r}, expected {CACHE_SCHEMA!r}")
        sets = {}
        for i, k in enumerate(doc["keys"]):
            e = doc["entries"][k]
            sets[k] = GraspSet(z[f"k{i}_q"], z[f"k{i}_qt"], z[f"k{i}_obj"], z[f"k{i}_info"],
                               tuple(e["joint_names"]), e.get("source", ""), e.get("stats", {}))
    return sets, doc


def merge_sets(base: Mapping[str, GraspSet], new: Mapping[str, GraspSet], overwrite: bool = False
               ) -> Dict[str, GraspSet]:
    out = dict(base)
    for k, s in new.items():
        if overwrite or k not in out:
            out[k] = s
    return out


def prune_sets(sets: Mapping[str, GraspSet], keep: Sequence[str]) -> Dict[str, GraspSet]:
    """Only the entries named in ``keep`` (in the cache's own order)."""
    keep = set(keep)
    return {k: s for k, s in sets.items() if k in keep}


def cache_summary(sets: Mapping[str, GraspSet]) -> dict:
    counts = [s.n for s in sets.values()]
    viable = [c for c in counts if c > 0]
    return {
        "n_keys": len(counts), "n_viable": len(viable),
        "viable_frac": (len(viable) / len(counts)) if counts else 0.0,
        "grasps_total": int(sum(counts)),
        "grasps_per_viable_median": float(np.median(viable)) if viable else 0.0,
        "grasps_per_viable_min": int(min(viable)) if viable else 0,
    }


# --------------------------------------------------------------------------
# Candidate sampling (torch)
# --------------------------------------------------------------------------


def sample_joint_candidates(canonical: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor,
                            valid: Optional[torch.Tensor], *, noise: float, curl_frac: float,
                            curl_range: Tuple[float, float] = (0.15, 0.85), curl_noise: float = 0.1,
                            generator: Optional[torch.Generator] = None, return_curl: bool = False):
    """PD targets for ``n`` candidates, ``(n, J)``, inside the joint limits
    (and, with ``return_curl``, which candidates are curl-family, ``(n,)``).

    A ``1 - curl_frac`` share: ``canonical + U(-noise, noise)`` (HORA:
    0.25, AnyRotate: 0.3 rad). A ``curl_frac`` share: every joint at the
    same random fraction c ~ U(curl_range) of its range, plus
    ``U(-curl_noise, curl_noise)`` (adapt.: grammar designs have no
    hand-made grasp pose). Joints outside ``valid`` keep ``canonical``."""
    n, j = canonical.shape
    dev = canonical.device

    def u(*shape):
        return torch.rand(*shape, device=dev, generator=generator)

    q_canon = canonical + (u(n, j) * 2.0 - 1.0) * noise
    c = u(n, 1) * (curl_range[1] - curl_range[0]) + curl_range[0]
    q_curl = lower + c * (upper - lower) + (u(n, j) * 2.0 - 1.0) * curl_noise
    use_curl = (u(n, 1) < curl_frac)
    q = torch.where(use_curl, q_curl, q_canon)
    q = torch.max(torch.min(q, upper), lower)
    if valid is not None:
        q = torch.where(valid, q, canonical)
    return (q, use_curl.squeeze(-1)) if return_curl else q


def random_quats(n: int, device=None, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Uniform random unit quaternions (w, x, y, z), w >= 0."""
    q = torch.randn(n, 4, device=device, generator=generator)
    q = torch.nn.functional.normalize(q, dim=-1)
    return torch.where(q[:, :1] < 0, -q, q)


@dataclass
class StabilityThresholds:
    """Acceptance tests of a held candidate (see the module docstring)."""

    max_disp_m: float = 0.02
    """Largest object displacement from its start (palm frame) at any
    control step of the hold. HORA: a fall of 1.5 cm (start z 0.66 m,
    threshold 0.645 m)."""
    max_lin_speed: float = 0.05  # m/s at the end of the hold
    max_ang_speed: float = 0.5  # rad/s at the end of the hold
    min_tip_contacts: int = 2  # HORA: >= 2; AnyRotate: "greater than 2"
    max_nontip_contacts: int = -1  # -1: no test (HORA); AnyRotate: 0
    max_tip_dist_m: float = 0.1  # HORA: every fingertip within 0.1 m of the object (whole hold)
    max_mean_tip_dist_m: float = -1.0  # AnyRotate: total < 0.2 over 4 tips = mean 0.05; -1: no test
    max_joint_speed: float = -1.0
    """Peak real-joint speed (rad/s) over the hold; -1: no test. A hand that
    vibrates while holding is not at rest (2026-10-02: two grammar designs
    dominated population training with per-step work penalties of -24 to
    -125)."""


def stable_mask(*, max_disp: torch.Tensor, lin_speed: torch.Tensor, ang_speed: torch.Tensor,
                tip_contacts: torch.Tensor, nontip_contacts: torch.Tensor, max_tip_dist: torch.Tensor,
                mean_tip_dist: torch.Tensor, finite: Optional[torch.Tensor] = None,
                joint_speed: Optional[torch.Tensor] = None,
                th: StabilityThresholds = StabilityThresholds()) -> torch.Tensor:
    """``(n,)`` bool: which candidates pass every test. ``max_tip_dist`` is
    the largest valid-fingertip distance to the object over the hold,
    ``mean_tip_dist`` the mean over valid fingertips at its end."""
    ok = (max_disp <= th.max_disp_m) & (lin_speed <= th.max_lin_speed) & (ang_speed <= th.max_ang_speed)
    ok = ok & (tip_contacts >= th.min_tip_contacts) & (max_tip_dist <= th.max_tip_dist_m)
    if th.max_nontip_contacts >= 0:
        ok = ok & (nontip_contacts <= th.max_nontip_contacts)
    if th.max_mean_tip_dist_m > 0:
        ok = ok & (mean_tip_dist <= th.max_mean_tip_dist_m)
    if th.max_joint_speed > 0 and joint_speed is not None:
        ok = ok & (joint_speed <= th.max_joint_speed)
    if finite is not None:
        ok = ok & finite
    return ok


def enclosure_mask(rel: torch.Tensor, contact: torch.Tensor, opposition_cos: float = -1.0,
                   max_mean_norm: float = -1.0) -> torch.Tensor:
    """``(n,)`` bool: the contacting fingertips enclose the object, as
    HORA's allegro pose does (a thumb against three fingers). ``rel``
    ``(n, k, 3)``: fingertip minus object centre; ``contact`` ``(n, k)``.

    - Opposition (``opposition_cos`` > -1): some contacting digit points
      against the mean direction of the other contacting digits, with a
      cosine <= -``opposition_cos`` (0.5: at least 120 degrees apart).
    - Span (``max_mean_norm`` > 0): the mean of the contacting digits' unit
      directions is at most this long (directions spread around the object).
    """
    c = contact.to(rel.dtype).unsqueeze(-1)
    u = rel / rel.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    total = (u * c).sum(dim=1)  # (n, 3)
    count = c.sum(dim=1).squeeze(-1)  # (n,)
    ok = count >= 1
    if opposition_cos > -1.0:
        others = total.unsqueeze(1) - u * c  # (n, k, 3)
        n_others = count.unsqueeze(1) - c.squeeze(-1)
        cos = (u * (others / others.norm(dim=-1, keepdim=True).clamp(min=1e-9))).sum(dim=-1)
        ok = ok & (contact & (n_others >= 1) & (cos <= -float(opposition_cos))).any(dim=1)
    if max_mean_norm > 0:
        ok = ok & ((total / count.clamp(min=1).unsqueeze(-1)).norm(dim=-1) <= float(max_mean_norm))
    return ok


# --------------------------------------------------------------------------
# Reset-time table
# --------------------------------------------------------------------------


@dataclass
class GraspTable:
    """All designs' grasps on the env's device, in the env's joint order.
    Design d owns rows ``offsets[d] : offsets[d] + counts[d]``."""

    q: torch.Tensor
    q_target: torch.Tensor
    obj_pose: torch.Tensor
    offsets: torch.Tensor
    counts: torch.Tensor

    @property
    def has_grasp(self) -> torch.Tensor:
        return self.counts > 0

    @property
    def total(self) -> int:
        return int(self.q.shape[0])

    def sample(self, design_idx: torch.Tensor, generator: Optional[torch.Generator] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(rows, ok)``: a uniformly drawn grasp row per env (HORA:
        ``np.random.randint`` over the cache), and whether its design has
        any (rows of the others are 0 and must not be used)."""
        counts = self.counts[design_idx]
        ok = counts > 0
        u = torch.rand(design_idx.shape[0], device=design_idx.device, generator=generator)
        local = torch.minimum((u * counts.clamp(min=1).float()).long(), (counts - 1).clamp(min=0))
        rows = torch.where(ok, self.offsets[design_idx] + local, torch.zeros_like(local))
        return rows, ok


def build_table(sets: Sequence[Optional[GraspSet]], env_joint_names: Sequence[str], device="cpu",
                max_per_design: int = 0) -> GraspTable:
    """Stack per-design sets (index = design index; None or empty = no
    stable grasp) into a ``GraspTable``, reordering joints by name."""
    env_joint_names = list(env_joint_names)
    qs, qts, objs, counts = [], [], [], []
    for s in sets:
        if s is None or s.n == 0:
            counts.append(0)
            continue
        missing = [n for n in env_joint_names if n not in s.joint_names]
        if missing:
            raise ValueError(f"grasp set {s.source!r} lacks joints {missing}")
        cols = [s.joint_names.index(n) for n in env_joint_names]
        k = s.n if max_per_design <= 0 else min(s.n, max_per_design)
        qs.append(s.q[:k][:, cols])
        qts.append(s.q_target[:k][:, cols])
        objs.append(s.obj_pose[:k])
        counts.append(k)
    j = len(env_joint_names)
    cat = (lambda xs, w: np.concatenate(xs, 0) if xs else np.zeros((0, w), dtype=np.float32))
    counts_t = torch.as_tensor(counts, dtype=torch.long, device=device)
    offsets = torch.cumsum(counts_t, 0) - counts_t
    return GraspTable(
        q=torch.as_tensor(cat(qs, j), dtype=torch.float32, device=device),
        q_target=torch.as_tensor(cat(qts, j), dtype=torch.float32, device=device),
        obj_pose=torch.as_tensor(cat(objs, 7), dtype=torch.float32, device=device),
        offsets=offsets, counts=counts_t)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
