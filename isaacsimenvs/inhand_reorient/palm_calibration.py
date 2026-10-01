"""Pure-python (numpy + stdlib only, no isaaclab) palm-up calibration helpers.

Kept out of ``scene_utils.py``/``calibrate_palm_up.py`` deliberately: those
import ``isaaclab.*`` at module scope, which -- per
``tests/test_config_sanity.py``'s docstring -- bootstraps Kit on ANY import,
so nothing in this file can be reached from a module that also does that if
it is to stay testable under plain pytest (no Kit). Both
``calibrate_palm_up.py`` (produces ``hand_calibration.json``) and
``scene_utils.setup_scene`` (consumes it) import this module for the shared
candidate list, quaternion math, and JSON schema.

Quaternions are ``(w, x, y, z)`` throughout, matching
``HandOnlySpec.base_rot`` / Isaac Lab's own convention.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "CALIB_PATH", "AXIS_NAMES", "MIN_SPAWN_HEIGHT_ABOVE_PALM_M", "candidate_rotations",
    "quat_mul", "quat_inv", "quat_apply", "git_sha", "load_calibration", "save_calibration",
    "REPOSE_HAND_POSES_PATH", "resolve_repose_hand_pose_path", "load_repose_hand_poses",
]

CALIB_PATH = Path(__file__).resolve().parent / "hand_calibration.json"

# Which LOCAL axis of the hand-only URDF's root/palm link is tried as "up".
AXIS_NAMES: tuple[str, ...] = ("+z", "-z", "+x", "-x", "+y", "-y")

MIN_SPAWN_HEIGHT_ABOVE_PALM_M = 0.02
"""The ONE convention a calibrated spawn point and the runtime drop check
share (I34, 2026-09-27): ``quat_apply(base_rot, spawn_offset_local)[2]`` --
the cube's spawn point rotated into WORLD frame, exactly as
``reset_utils.reset_env_state`` does through the palm body's LIVE world
orientation (``quat_apply(palm_quat_w, local_pos)``; for a fixed-base
articulation whose palm body IS the root, ``palm_quat_w`` is ``base_rot``
bit-for-bit at every step, so this is not an approximation) -- must clear
this margin above the palm's own world z.

``calibrate_palm_up.py`` enforces this at SELECTION time (a candidate that
does not clear it scores 0, regardless of stability/reach); tests/
test_palm_calibration.py's ``test_every_calibrated_hand_clears_...`` enforces
it on the committed ``hand_calibration.json`` itself, so a hand cannot be
calibrated at all without satisfying the same thing
``drop_detection.object_below_palm`` checks every reset
(``obj_z - palm_z < -0.5 * drop_distance_m``, i.e. below -0.12 m at the
0.24 m default). 0.02 m is a modest positive margin comfortably inside that
-0.12 m failure threshold, not merely non-negative: before I34,
calibrate_palm_up.py's stability(near the spawn point)+reach(>=2 fingertips
close) scoring had NO requirement at all that the winning spawn point sit
above the palm in world z -- for an identity base_rot (sharpa,
allegro_right, dclaw) the winner happened to end up there anyway, but for a
non-identity axis it did not (xhand_right: -0.147 m, BELOW the palm, past
the runtime's own threshold, terminating episode step 1; wuji_right/
tesollo_dg5f_right: +0.051/-0.002 m, level with the palm, not genuinely held
up against gravity)."""


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, both (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_inv(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / float(np.dot(q, q))


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a unit quaternion (w, x, y, z)."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, qv), quat_inv(q))[1:]


def _axis_to_up_quat(axis: str) -> np.ndarray:
    """A quaternion R such that ``quat_apply(R, local_axis) == (0, 0, 1)``,
    with a 0-rad roll about that axis (an arbitrary but fixed convention --
    the candidate list below adds explicit rolls on top of this)."""
    from scipy.spatial.transform import Rotation

    sign = -1.0 if axis[0] == "-" else 1.0
    dim = axis[1]
    if dim == "z":
        rot = Rotation.identity() if sign > 0 else Rotation.from_euler("x", 180, degrees=True)
    elif dim == "x":
        # R_y(-90) sends local +x -> world +z; R_y(+90) sends local -x -> +z.
        rot = Rotation.from_euler("y", -90.0 * sign, degrees=True)
    elif dim == "y":
        # R_x(+90) sends local +y -> world +z; R_x(-90) sends local -y -> +z.
        rot = Rotation.from_euler("x", 90.0 * sign, degrees=True)
    else:
        raise ValueError(f"bad axis {axis!r}")
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


def _roll_quat(axis: str, degrees: float) -> np.ndarray:
    """Extra rotation about the WORLD +z axis (applied after the up-mapping,
    i.e. a roll of the palm around its own new vertical), as (w, x, y, z)."""
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_euler("z", degrees, degrees=True).as_quat()
    return np.array([w, x, y, z])


def candidate_rotations(full: bool = False) -> list[dict]:
    """``[{"axis": "+x", "roll_deg": 0.0, "quat_wxyz": (w, x, y, z)}, ...]``.

    ``full=False`` (the default; the plan's minimum acceptable scope): the 6
    "which local axis points up" candidates, roll fixed at 0 deg. ``full=True``:
    24 candidates (6 axes x the 4 principal rolls), the plan's "or ... 24
    axis-aligned base rotations".
    """
    rolls = (0.0, 90.0, 180.0, 270.0) if full else (0.0,)
    out = []
    for axis in AXIS_NAMES:
        up = _axis_to_up_quat(axis)
        for roll in rolls:
            q = quat_mul(_roll_quat(axis, roll), up) if roll else up
            q = q / np.linalg.norm(q)
            out.append({"axis": axis, "roll_deg": float(roll),
                        "quat_wxyz": tuple(float(v) for v in q)})
    return out


def git_sha(repo_root: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root or Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, check=True, timeout=5)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def load_calibration(path: Path | None = None) -> dict:
    p = path or CALIB_PATH
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_calibration(data: dict, path: Path | None = None) -> None:
    p = path or CALIB_PATH
    existing = load_calibration(p)
    existing.update(data)
    p.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")


REPOSE_HAND_POSES_PATH = Path(__file__).resolve().parent / "repose_hand_poses.json"
"""Hand poses for the isaaclab_repose task profile, keyed by hand id: where a
hand sits and where its cube spawns when the profile should reproduce a
reference setup rather than use the palm-up calibration above. Schema per
entry: ``base_pos`` (env frame, m), ``base_rot`` (w, x, y, z),
``spawn_offset_local`` (cube spawn point in the root/palm frame, m),
``hand_default_joint_pos`` ({joint: rad}), plus provenance fields."""


def resolve_repose_hand_pose_path(value: str) -> Path | None:
    """``cfg.repose.hand_pose_file``: empty disables the override; a
    relative path is relative to this package."""
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else Path(__file__).resolve().parent / p


def load_repose_hand_poses(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text())

