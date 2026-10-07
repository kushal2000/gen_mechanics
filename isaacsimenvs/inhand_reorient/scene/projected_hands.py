"""Conformed commercial hands in the envelope, related back to their URDFs.
Numpy + `hand_sampler` only (no Kit), like `grammar_envelope.py`.

A commercial design's source is `commercial:<hand id>`; its fit
(`hand_sampler/grammar_bench/conformed_hands.json`) gives the grammar dof ->
URDF joint map, the palm frame in the URDF root frame (`palm_T`) and the
zero-pose differences (`q_off`), so:

- `joint_slot_map`: URDF joint name -> envelope slot;
- `slot_values`: URDF joint values -> slot values (q = q_off + s q_urdf,
  with s the sign between the grammar's derived axis and the URDF's);
- `urdf_equivalent_placement`: the pose, object start and default joints
  that put the design where its single-hand URDF asset stands
  (`repose_hand_poses.json`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from . import grammar_envelope as ge

__all__ = ["hand_id_of", "fit_record", "joint_slot_map", "slot_values", "urdf_equivalent_placement"]

COMMERCIAL_PREFIX = "commercial:"


def hand_id_of(source: str) -> Optional[str]:
    return source[len(COMMERCIAL_PREFIX):] if source.startswith(COMMERCIAL_PREFIX) else None


@lru_cache(maxsize=None)
def _records() -> dict:
    from hand_sampler.grammar import build_conformed

    return build_conformed.load()


def fit_record(hand_id: str) -> dict:
    rec = _records().get(hand_id)
    if rec is None:
        raise KeyError(f"{hand_id!r} not in conformed_hands.json")
    return rec


@lru_cache(maxsize=None)
def _urdf_axes(hand_id: str) -> Dict[str, np.ndarray]:
    """URDF joint name -> its unit axis at the zero pose, in the hand-root frame."""
    from hand_sampler.grammar.commercial import load_real_hand

    real = load_real_hand(hand_id)
    out = {j.name: j.axis for j in real.palm_joints}
    for f in real.fingers:
        out.update({j.name: j.axis for j in f.joints})
    return out


def joint_slot_map(design: "ge.EnvelopeDesign", hand_id: str) -> Dict[str, int]:
    """URDF joint name -> slot, for the design's controlled joints."""
    rec = fit_record(hand_id)
    slot_of = {n: i for i, n in enumerate(design.slot_joint_name) if n is not None and design.slot_valid[i]}
    return {urdf: slot_of[g] for g, urdf in rec["name_map"].items() if g in slot_of}


def _signs_and_offsets(design: "ge.EnvelopeDesign", hand_id: str) -> Tuple[Dict[int, float], Dict[int, float]]:
    from hand_sampler.grammar import derive as gdv

    rec = fit_record(hand_id)
    R = np.asarray(rec["palm_T"])[:3, :3]
    axes = _urdf_axes(hand_id)
    ds = gdv.dofs(design.hand)
    q_off = np.asarray(rec["q_off"], dtype=float)
    slots = ge.slot_of_dof(design)
    sign, off = {}, {}
    for k, d in enumerate(ds):
        urdf = rec["name_map"].get(d.name)
        if urdf is None or urdf not in axes:
            continue
        if d.finger < 0:
            a = gdv.palm_axis(design.hand, d.index)
        else:
            # the link frame of joint (i, j) at the commercial zero pose: the zero-pose
            # differences of the joints before it turn it; its axis is fixed in it
            a = _axis_at(design.hand, d.finger, d.index, q_off)
        sign[slots[k]] = 1.0 if float(a @ (R.T @ axes[urdf])) >= 0 else -1.0
        off[slots[k]] = float(q_off[k])
    return sign, off


def _axis_at(hand, i: int, j: int, q_dofs: np.ndarray) -> np.ndarray:
    """Finger i joint j's signed axis, in the palm frame, at joint vector `q_dofs`."""
    from hand_sampler.grammar import derive as gdv

    pose = gdv.fk(hand, q_dofs)
    T = pose.links[i][j]
    return T[:3, :3] @ gdv.joint_axis(hand, i, j)


def slot_values(design: "ge.EnvelopeDesign", hand_id: str, values: Mapping[str, float]) -> Dict[int, float]:
    """`{slot: value}` for URDF-joint-named `values` (e.g. a grasp pose)."""
    m = joint_slot_map(design, hand_id)
    sign, off = _signs_and_offsets(design, hand_id)
    return {m[n]: off.get(m[n], 0.0) + sign.get(m[n], 1.0) * float(v) for n, v in values.items() if n in m}


def _mat_to_quat_wxyz(R: np.ndarray) -> Tuple[float, float, float, float]:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return (float(w), float(x), float(y), float(z))


def _quat_wxyz_to_mat(q) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    w, x, y, z = (float(v) for v in q)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def urdf_equivalent_placement(design: "ge.EnvelopeDesign", hand_id: str, pose_entry: Mapping) -> dict:
    """`{"base_pos", "base_rot_wxyz", "spawn_offset", "default_q"}` putting the
    design where `pose_entry` puts the URDF hand: the grammar root is the
    palm frame `palm_T` (p_root = R p_palm + o) inside the URDF root."""
    rec = fit_record(hand_id)
    T = np.asarray(rec["palm_T"], dtype=float)
    R, o = T[:3, :3], T[:3, 3]
    R_u = _quat_wxyz_to_mat(pose_entry["base_rot"])
    base_pos = np.asarray(pose_entry["base_pos"], dtype=float) + R_u @ o
    spawn = R.T @ (np.asarray(pose_entry["spawn_offset_local"], dtype=float) - o)
    default_q = np.array(ge.palm_up(design).default_q, dtype=float)
    for slot, v in slot_values(design, hand_id, pose_entry.get("hand_default_joint_pos") or {}).items():
        lo, hi = design.slot_limits[slot]
        margin = min(1e-3, max(0.0, (hi - lo) / 2.0 - 1e-9))
        default_q[slot] = min(max(v, lo + margin), hi - margin)
    return {"base_pos": tuple(float(v) for v in base_pos), "base_rot_wxyz": _mat_to_quat_wxyz(R_u @ R),
            "spawn_offset": tuple(float(v) for v in spawn), "default_q": ge.tied_q(design, default_q)}
