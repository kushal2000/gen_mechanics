"""Point every finger's flexion at the workspace.

Face-based mounting deliberately frees fingers from one side of the palm, but it
left ``roll`` -- the spin about the finger's own axis -- sampled uniformly. Roll
rotates the plane the finger flexes in, so a finger can be mounted on the face
pointing at the table and still curl upward, away from anything it could grasp.
Measured across the cached population, the component of curl direction along the
palm's table-facing axis ranged from -0.78 to +0.79: effectively arbitrary.

That is not what mounting freedom was for. Where a finger sits, how many joints
it has and which way it points should all be free; which way it CLOSES should
not be, because a finger that opens away from the object is not an exotic design
worth exploring, it is a finger doing nothing.

The correction is exact, not a heuristic. Rolling the mount by theta rotates
everything downstream about the finger axis k, so the curl direction is

    c(theta) = c0 cos(theta) + (k x c0) sin(theta) + k (k . c0)(1 - cos(theta))

and its component along the desired direction d is A cos + B sin + C with

    A = d.c0 - (d.k)(k.c0)      B = d.(k x c0)      C = (d.k)(k.c0)

maximised at theta* = atan2(B, A). One FK evaluation per finger gives c0; the
roll follows in closed form. No search, no resampling.

``c0`` is MEASURED by forward kinematics rather than composed by hand: the chain
from mount to fingertip runs through the CMC/MCP virtual links and the fixed
ROLL_FE_TO_AA segment, so the curl direction is not simply the mount frame's +y,
and deriving it analytically would be a second implementation of the URDF
builder's geometry that could silently disagree with it.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from hand_sampler import params as P

# The palm's +x points down at the table when the arm is in its home pose (the
# -x face is the back of the hand, which is why it is excluded from mounting).
# So "flex toward the workspace" is "flex along palm +x".
CURL_TARGET = np.array([1.0, 0.0, 0.0])

# Flexion joints. AA (abduction) joints spread the finger rather than close it,
# so they are held at zero while the curl direction is measured.
_FLEX_SUFFIXES = ("FE", "PIP", "DIP")


def _finger_axis(mount: P.Segment) -> np.ndarray:
    """The finger's own axis (mount-frame +x) in palm coordinates."""

    r, p, y = mount.rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    # First column of Rz(y) Ry(p) Rx(r) -- the image of local +x.
    return np.array([cy * cp, sy * cp, -sp])


def curl_directions(hand: P.HandParams, urdf_path=None) -> dict[int, np.ndarray]:
    """Unit curl direction per active finger index, in the palm frame."""

    import yourdfpy

    from hand_sampler.urdf import urdf_path_for, write_urdf
    from hand_sampler.paths import resolve as resolve_repo_path

    if urdf_path is None:
        urdf_path = urdf_path_for(hand)
        if not urdf_path.exists():
            write_urdf(hand, urdf_path)
    urdf = yourdfpy.URDF.load(str(resolve_repo_path(urdf_path)), load_meshes=False,
                              load_collision_meshes=False, build_scene_graph=True)

    out: dict[int, np.ndarray] = {}
    palm_inv = None
    for i, finger in enumerate(hand.fingers):
        if not finger.active:
            continue
        tip = f"gen_f{i}_DP"
        if tip not in urdf.link_map:
            continue
        flex = [j for j in urdf.joint_map
                if j.startswith(f"gen_f{i}_")
                and j.rsplit("_", 1)[-1] in _FLEX_SUFFIXES]
        if not flex:
            continue

        def tip_in_palm(angle: float) -> np.ndarray:
            urdf.update_cfg({j: angle for j in flex})
            palm = urdf.get_transform("gen_palm")
            return (np.linalg.inv(palm) @ urdf.get_transform(tip))[:3, 3]

        # A finite flexion rather than a derivative: the whole point is where the
        # fingertip ends up when the hand closes, not its instantaneous velocity.
        delta = tip_in_palm(0.6) - tip_in_palm(0.0)
        norm = float(np.linalg.norm(delta))
        if norm < 1e-9:
            continue
        out[i] = delta / norm
    return out


def optimal_roll_offset(curl: np.ndarray, axis: np.ndarray,
                        target: np.ndarray = CURL_TARGET) -> float:
    """Extra roll that best aligns ``curl`` with ``target``, about ``axis``."""

    k = axis / (np.linalg.norm(axis) + 1e-12)
    A = float(np.dot(target, curl) - np.dot(target, k) * np.dot(k, curl))
    B = float(np.dot(target, np.cross(k, curl)))
    return math.atan2(B, A)


def align_flexion_downward(hand: P.HandParams, urdf_path=None) -> P.HandParams:
    """Re-roll every face-mounted finger so it flexes toward the workspace.

    Fingers whose mount did not come from :func:`params.mount_on_face` are left
    alone: their transforms are measured values (SHARPA's, for the reference
    hand) and re-rolling them would silently stop reproducing the robot they
    were taken from.
    """

    curls = curl_directions(hand, urdf_path=urdf_path)
    if not curls:
        return hand

    fingers = list(hand.fingers)
    changed = False
    for i, finger in enumerate(fingers):
        if i not in curls or not finger.mount_params:
            continue
        face, u_frac, v_frac, roll, tilt, tilt_azimuth = finger.mount_params
        d_roll = optimal_roll_offset(curls[i], _finger_axis(finger.mount))
        new_roll = roll + d_roll
        fingers[i] = replace(
            finger,
            mount=P.mount_on_face(face, u_frac, v_frac, new_roll, tilt,
                                  tilt_azimuth, hand.palm_extents),
            mount_params=(face, u_frac, v_frac, new_roll, tilt, tilt_azimuth),
        )
        changed = True

    if not changed:
        return hand
    return replace(hand, fingers=tuple(fingers))


def report(hand: P.HandParams) -> list[tuple[int, str, float]]:
    """(finger index, face, curl component toward the table) per active finger."""

    rows = []
    for i, c in curl_directions(hand).items():
        f = hand.fingers[i]
        face = f.mount_params[0] if f.mount_params else "-"
        rows.append((i, face, float(np.dot(c, CURL_TARGET))))
    return rows


__all__ = ["CURL_TARGET", "align_flexion_downward", "curl_directions",
           "optimal_roll_offset", "report"]
