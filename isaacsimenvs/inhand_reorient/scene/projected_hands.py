"""Projected commercial hands in the grammar envelope, related back to their
URDFs; and a palm collider for grammar designs. Numpy + ``hand_sampler``
only (no Kit), like ``grammar_envelope.py``.

- ``joint_slot_map``: URDF joint name -> envelope slot, through the
  projection's own ``name_map`` (URDF joint -> grammar joint) and the
  design's ``slot_joint_name``.
- ``urdf_equivalent_placement``: the pose, spawn point and default joints
  that put a projected hand where its single-hand URDF asset stands
  (``repose_hand_poses.json``). The projection expresses the hand in a root
  frame rotated from the URDF root by ``root_transform`` (p_urdf = R
  p_grammar, same origin), so the grammar root gets base_rot = R_urdf R and
  the spawn point R^T s_urdf.
- ``palm_hull_points``: a convex palm for a grammar design, spanning the
  root capsule and every finger mount, thickened by the capsule radius
  except past the mounts along the root axis (where the second phalanges
  start). The grammar's own palm cells (``grammar.geometry``) cover only
  palm BODIES, so a one-body palm (every projected four-finger hand) gets
  just its 1 cm root capsule; the object can then rest only on the fingers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from . import grammar_envelope as ge

__all__ = ["hand_id_of", "projection", "joint_slot_map", "slot_values", "urdf_equivalent_placement",
           "palm_hull_points", "palm_filter_slots"]


def palm_filter_slots(design: "ge.EnvelopeDesign", links_per_finger: int = 2) -> list:
    """Envelope slots the palm hull is collision-filtered against: each
    finger's first ``links_per_finger`` real links (at rest they start
    inside or next to the hull)."""
    out = []
    for f in range(ge.N_FINGERS):
        real = [ge.finger_slot(f, d) for d in range(ge.N_JOINTS_PER_FINGER) if design.slot_valid[ge.finger_slot(f, d)]]
        out += real[:links_per_finger]
    return sorted(out)

PROJECTED_PREFIX = "projected:"


def hand_id_of(source: str) -> Optional[str]:
    return source[len(PROJECTED_PREFIX):] if source.startswith(PROJECTED_PREFIX) else None


@lru_cache(maxsize=None)
def projection(hand_id: str) -> Tuple[Dict[str, str], np.ndarray]:
    """``(name_map, root_transform)`` of the hand's projection (the same
    pipeline as ``population_file.projected_entry``)."""
    from hand_sampler.grammar.adapters.projection import project_to_derivation
    from hand_sampler.grammar.adapters.urdf import load_urdf

    from . import population_file as pf

    manifest = pf._load_manifest()
    hand = next(h for h in manifest["hands"] if h["id"] == hand_id)
    path, availability, reason = pf._resolve_hand(hand, manifest)
    if availability != "available":
        raise FileNotFoundError(f"{hand_id}: {reason}")
    imported = load_urdf(path, hand_root=hand.get("hand_root"))
    pr = project_to_derivation(imported.model, palm_joints=hand.get("palm_joints") or (),
                               tip_frames=hand.get("tip_frames") or {})
    return dict(pr.name_map), np.asarray(pr.root_transform, dtype=float)


def joint_slot_map(design: "ge.EnvelopeDesign", hand_id: str) -> Dict[str, int]:
    """URDF joint name -> envelope slot index for the design's real joints."""
    name_map, _ = projection(hand_id)
    slot_of = {n: i for i, n in enumerate(design.slot_joint_name) if n is not None and design.slot_valid[i]}
    return {urdf: slot_of[g] for urdf, g in name_map.items() if g in slot_of}


def slot_values(design: "ge.EnvelopeDesign", hand_id: str, values: Mapping[str, float]) -> Dict[int, float]:
    """``{slot: value}`` for the URDF-joint-named ``values`` (e.g. a grasp pose)."""
    m = joint_slot_map(design, hand_id)
    return {m[name]: float(v) for name, v in values.items() if name in m}


def _mat_to_quat_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return (float(w), float(x), float(y), float(z))


def _quat_wxyz_to_mat(q) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    w, x, y, z = (float(v) for v in q)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def urdf_equivalent_placement(design: "ge.EnvelopeDesign", hand_id: str, pose_entry: Mapping) -> dict:
    """``{"base_pos", "base_rot_wxyz", "spawn_offset", "default_q"}`` putting
    the projected design where ``pose_entry`` (a ``repose_hand_poses.json``
    entry for the URDF hand) puts the URDF hand. ``default_q`` is a (36,)
    array: the design's own default with the entry's joints on their slots
    (a follower carrier takes its leader's value)."""
    _, R = projection(hand_id)
    R = R[:3, :3]
    R_u = _quat_wxyz_to_mat(pose_entry["base_rot"])
    base_rot = _mat_to_quat_wxyz(R_u @ R)
    spawn = R.T @ np.asarray(pose_entry["spawn_offset_local"], dtype=float)
    from .grammar_envelope import palm_up

    default_q = np.array(palm_up(design, n_sweep=0).default_q, dtype=float)
    for slot, v in slot_values(design, hand_id, pose_entry.get("hand_default_joint_pos") or {}).items():
        lo, hi = design.slot_limits[slot]
        margin = min(1e-3, max(0.0, (hi - lo) / 2.0 - 1e-9))
        default_q[slot] = min(max(v, lo + margin), hi - margin)
    default_q = ge.tied_q(design, default_q)
    return {"base_pos": tuple(float(v) for v in pose_entry["base_pos"]), "base_rot_wxyz": base_rot,
            "spawn_offset": tuple(float(v) for v in spawn), "default_q": default_q}


def palm_hull_points(design: "ge.EnvelopeDesign", radius: float) -> np.ndarray:
    """Points (root frame, q = 0) whose convex hull is the palm: the root
    capsule axis and every finger/carrier mount, each offset by +-radius
    along root x and y; the capsule start also by -radius along z. Nothing
    reaches past max(root length, highest mount) along z."""
    T0 = ge.authored_fk(design, np.zeros(ge.N_SLOTS))
    keys = [np.zeros(3), np.array([0.0, 0.0, float(design.root_length_m)])]
    for s in ge.FINGER_BASE_SLOTS + ge.CARRIER_SLOTS:   # finger mounts and palm joints
        if design.slot_valid[s]:
            keys.append(T0[s][:3, 3])
    r = float(radius)
    offsets = [np.array(v) for v in ((r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0))]
    pts = [k + o for k in keys for o in offsets]
    pts.append(np.array([0.0, 0.0, -r]))
    return np.unique(np.round(np.stack(pts), 9), axis=0)
