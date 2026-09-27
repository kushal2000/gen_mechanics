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
    "CALIB_PATH", "AXIS_NAMES", "candidate_rotations", "quat_mul", "quat_inv",
    "quat_apply", "git_sha", "load_calibration", "save_calibration",
]

CALIB_PATH = Path(__file__).resolve().parent / "hand_calibration.json"

# Which LOCAL axis of the hand-only URDF's root/palm link is tried as "up".
AXIS_NAMES: tuple[str, ...] = ("+z", "-z", "+x", "-x", "+y", "-y")


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
