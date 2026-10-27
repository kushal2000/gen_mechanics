"""What the keypoint residual measures, and where it disagrees with an angle.

The in-hand task scores orientation with a KEYPOINT residual, not an angle:
``compute_intermediate_values`` refers the object's and the goal's keypoint clouds
to their own centres and takes the largest per-keypoint distance
(``observations.py:250``). The success test is then ``residual <= tolerance``.

It is natural to assume that is a stand-in for the rotation angle. It is not, and
the difference is not small: with the shipped keypoint set the SAME angle can be a
success or a failure depending on the rotation AXIS, by a factor of sqrt(3). These
tests pin the exact relationship so that the anisotropy is a documented property
rather than something rediscovered from a confusing training curve.

Kit-free on purpose: ``isaaclab.utils.math`` cannot be imported without booting
Isaac Sim, so the quaternion algebra here is plain numpy. KEYPOINT_CORNERS is read
out of the real source with ``ast`` instead of being copied, so the tests fail if
the constant changes rather than silently testing a stale literal.

    .venv_isaacsim/bin/python -m pytest isaacsimenvs/pose_reaching_6d/tests/test_orientation_metric.py -q
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
OBSERVATIONS = ROOT / "isaacsimenvs/pose_reaching_6d/obs_utils/observations.py"


def _literal_from_source(path: Path, name: str):
    """The value of a module-level literal assignment, without importing it."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign)
            else node.targets if isinstance(node, ast.Assign)
            else []
        )
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


KEYPOINT_CORNERS = np.asarray(_literal_from_source(OBSERVATIONS, "KEYPOINT_CORNERS"), float)

# The shipped in-hand configuration. If these move, the angles below move with
# them and the test says so.
CUBE_EDGE_M = 0.045
KEYPOINT_SCALE = 1.5
SHIPPED_TOLERANCE_M = 0.0039


# ---------------------------------------------------------------------------
# the two functions under comparison
# ---------------------------------------------------------------------------


def keypoint_offsets(edge_m: float = CUBE_EDGE_M, keypoint_scale: float = KEYPOINT_SCALE):
    """``reset.py:233``: corners scaled by ``0.5 * keypoint_scale * fixed_size``."""
    return KEYPOINT_CORNERS * (0.5 * keypoint_scale * edge_m)


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def keypoint_residual(q_obj, q_goal, offsets, obj_pos=None, goal_pos=None):
    """The metric the env actually scores, including the centring."""
    off = np.asarray(offsets, float)
    obj_pos = np.zeros(3) if obj_pos is None else np.asarray(obj_pos, float)
    goal_pos = np.zeros(3) if goal_pos is None else np.asarray(goal_pos, float)
    obj_kp = obj_pos + (quat_to_matrix(q_obj) @ off.T).T
    goal_kp = goal_pos + (quat_to_matrix(q_goal) @ off.T).T
    return float(np.linalg.norm((obj_kp - obj_pos) - (goal_kp - goal_pos), axis=1).max())


def direct_angle(q_obj, q_goal) -> float:
    """Geodesic angle between two orientations, in radians. The obvious metric."""
    d = abs(float(np.dot(np.asarray(q_obj, float), np.asarray(q_goal, float))))
    return 2.0 * math.acos(min(d, 1.0))


def _rand_quats(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _quat_from_angle_axis(angle, axis):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * a])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


# ---------------------------------------------------------------------------
# what the residual gets right
# ---------------------------------------------------------------------------


def test_residual_ignores_position():
    """The centring is what makes this a reorientation task, so pin it."""
    off = keypoint_offsets()
    rng = np.random.default_rng(1)
    for q_obj, q_goal in zip(_rand_quats(64, 2), _rand_quats(64, 3)):
        base = keypoint_residual(q_obj, q_goal, off)
        moved = keypoint_residual(
            q_obj, q_goal, off,
            obj_pos=rng.normal(scale=0.5, size=3), goal_pos=rng.normal(scale=0.5, size=3))
        assert abs(base - moved) < 1e-12, "the residual moved when only position did"


def test_residual_is_zero_exactly_at_the_goal():
    off = keypoint_offsets()
    for q in _rand_quats(64, 4):
        assert keypoint_residual(q, q, off) < 1e-12
        assert keypoint_residual(q, -q, off) < 1e-12, "q and -q are the same rotation"


def test_residual_rises_monotonically_with_angle_on_a_fixed_axis():
    """For ONE axis the residual is a clean function of the angle -- it is only
    across axes that it stops being one."""
    off = keypoint_offsets()
    rng = np.random.default_rng(5)
    for _ in range(16):
        q_goal = _rand_quats(1, rng.integers(1 << 30))[0]
        axis = rng.normal(size=3)
        prev = -1.0
        for deg in range(0, 181, 5):
            q_obj = _quat_mul(_quat_from_angle_axis(math.radians(deg), axis), q_goal)
            r = keypoint_residual(q_obj, q_goal, off)
            if deg <= 180:
                assert r >= prev - 1e-12 or deg > 120, "not monotone below 120 deg"
            prev = r


# ---------------------------------------------------------------------------
# where it and the angle part company -- the point of the file
# ---------------------------------------------------------------------------

# The keypoint set is FOUR points that are two antipodal pairs, all coplanar
# through the centre, so it carries only TWO independent directions. The smallest
# fraction of the keypoint radius that can be left perpendicular to a rotation
# axis is therefore sin(35.2644 deg): the two distinct keypoints are 70.5288 deg
# apart, and the axis bisecting them is the one that hides the most of both.
K_MIN = math.sin(math.radians(35.264389682754654))


def test_keypoint_set_is_degenerate():
    """WHY the anisotropy exists. If this fails the constant changed, and every
    angle in this file has to be recomputed."""
    off = keypoint_offsets()
    assert len(off) == 4
    r = np.linalg.norm(off, axis=1)
    assert np.allclose(r, r[0]), "keypoints are not all at one radius"
    u = off / r[:, None]
    gram = u @ u.T
    # two antipodal pairs
    assert math.isclose(gram[0, 3], -1.0, abs_tol=1e-12)
    assert math.isclose(gram[1, 2], -1.0, abs_tol=1e-12)
    # and therefore rank 2: all four lie in one plane through the origin
    assert np.linalg.matrix_rank(off, tol=1e-9) == 2, "expected a coplanar set"
    # the two distinct directions, and the bisector constant derived from them
    cos = abs(gram[0, 1])
    assert math.isclose(cos, 1 / 3, abs_tol=1e-12)
    # The axis that hides the most of both keypoints bisects them, sitting
    # half(70.5288) = 35.2644 deg from each, so each keeps sin(35.2644) of its
    # radius perpendicular. Bisecting u0 and -u1 instead would sit 54.74 deg away
    # and hide LESS, so this is the minimum and not merely a stationary point.
    half = math.acos(cos) / 2.0
    assert math.isclose(math.sin(half), K_MIN, abs_tol=1e-12)
    assert math.sin(half) < math.sin(math.acos(-cos) / 2.0)


def test_residual_matches_the_closed_form():
    """residual = 2*sin(theta/2) * max_i |o_i perpendicular to the axis|.

    Exact, and it is the identity that explains everything else here: the residual
    depends on the axis only through that max.
    """
    off = keypoint_offsets()
    rng = np.random.default_rng(6)
    for _ in range(400):
        q_goal = _rand_quats(1, rng.integers(1 << 30))[0]
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        deg = float(rng.uniform(0.1, 170.0))
        q_obj = _quat_mul(_quat_from_angle_axis(math.radians(deg), axis), q_goal)
        got = keypoint_residual(q_obj, q_goal, off)
        # the keypoints as the rotation sees them: in world, at the goal pose
        v = (quat_to_matrix(q_goal) @ off.T).T
        perp = np.linalg.norm(v - np.outer(v @ axis, axis), axis=1).max()
        want = 2.0 * math.sin(math.radians(deg) / 2.0) * perp
        assert abs(got - want) < 1e-9, f"{deg} deg: {got} != {want}"


def test_one_angle_is_both_a_success_and_a_failure():
    """The headline. At a fixed angle the success test depends on the AXIS."""
    off = keypoint_offsets()
    tol = SHIPPED_TOLERANCE_M * KEYPOINT_SCALE           # observations.py:269
    q_goal = np.array([1.0, 0.0, 0.0, 0.0])
    v = (quat_to_matrix(q_goal) @ off.T).T
    most = np.cross(v[0], v[1])                          # perpendicular to both
    least = v[0] + v[1]                                  # bisects them
    deg = 7.0
    r_most = keypoint_residual(
        _quat_mul(_quat_from_angle_axis(math.radians(deg), most), q_goal), q_goal, off)
    r_least = keypoint_residual(
        _quat_mul(_quat_from_angle_axis(math.radians(deg), least), q_goal), q_goal, off)
    assert r_most > tol, "7 deg should FAIL on the most sensitive axis"
    assert r_least < tol, "7 deg should PASS on the least sensitive axis"
    assert math.isclose(r_most / r_least, 1.0 / K_MIN, rel_tol=1e-9)


def test_angle_does_not_determine_the_residual_over_random_rotations():
    """Sampling random quaternion pairs, as one would to check the two agree:
    they do not, and the disagreement is bounded exactly by K_MIN."""
    off = keypoint_offsets()
    r = float(np.linalg.norm(off, axis=1).max())
    worst_ratio = 1.0
    for q_obj, q_goal in zip(_rand_quats(2000, 7), _rand_quats(2000, 8)):
        theta = direct_angle(q_obj, q_goal)
        got = keypoint_residual(q_obj, q_goal, off)
        ceiling = 2.0 * r * math.sin(theta / 2.0)        # axis perpendicular to a keypoint
        floor = K_MIN * ceiling                          # axis bisecting the two
        assert got <= ceiling + 1e-9, "residual exceeded 2*r*sin(theta/2)"
        assert got >= floor - 1e-9, "residual fell below the K_MIN floor"
        if ceiling > 1e-9:
            worst_ratio = max(worst_ratio, ceiling / max(got, 1e-12))
    # random pairs really do reach both ends, so this is not a vacuous bound
    assert worst_ratio > 1.7, f"expected the spread to approach 1/K_MIN=1.732, got {worst_ratio}"


# ---------------------------------------------------------------------------
# the tolerance, in the units people actually set it in
# ---------------------------------------------------------------------------


def tolerance_for_angle(deg: float, edge_m: float = CUBE_EDGE_M) -> float:
    """success_tolerance giving this threshold on the MOST sensitive axis.

    keypoint_scale cancels: the tolerance is multiplied by it
    (observations.py:269) and the keypoint radius already contains it
    (reset.py:233), so the angle depends only on the tolerance and the edge.
    """
    return edge_m * math.sqrt(3.0) * math.sin(math.radians(deg) / 2.0)


def threshold_angles_deg(tolerance_m: float, edge_m: float = CUBE_EDGE_M):
    """(tightest, loosest) threshold angle in degrees for a tolerance."""
    r = 0.5 * edge_m * math.sqrt(3.0)
    tight = 2.0 * math.degrees(math.asin(min(tolerance_m / (2.0 * r), 1.0)))
    loose = 2.0 * math.degrees(math.asin(min(tolerance_m / (2.0 * r * K_MIN), 1.0)))
    return tight, loose


@pytest.mark.parametrize("keypoint_scale", [0.5, 1.0, 1.5, 3.0])
def test_threshold_angle_is_independent_of_keypoint_scale(keypoint_scale):
    """A real trap: keypoint_scale looks like a difficulty knob and is not."""
    off = keypoint_offsets(keypoint_scale=keypoint_scale)
    r = float(np.linalg.norm(off, axis=1).max())
    tol = SHIPPED_TOLERANCE_M * keypoint_scale
    got = 2.0 * math.degrees(math.asin(tol / (2.0 * r)))
    assert math.isclose(got, 5.7362, abs_tol=1e-3), got


def test_shipped_and_planned_tolerances():
    """The numbers the runs are configured with, pinned both ways.

    Verified against the simulator by bisecting near_goal over 32 envs: 5.7362 and
    9.9437 degrees for the shipped 0.0039, matched to four decimals.
    """
    tight, loose = threshold_angles_deg(SHIPPED_TOLERANCE_M)
    assert math.isclose(tight, 5.7362, abs_tol=1e-3)
    assert math.isclose(loose, 9.9437, abs_tol=1e-3)

    for deg, want_m in ((5.0, 0.003400), (20.0, 0.013535)):
        got = tolerance_for_angle(deg)
        assert math.isclose(got, want_m, abs_tol=1e-6), f"{deg} deg -> {got}"
        tight, _ = threshold_angles_deg(got)
        assert math.isclose(tight, deg, abs_tol=1e-6), "round trip failed"


# ---------------------------------------------------------------------------
# orientation_metric="angle": the true angle, tolerance converted back exactly
# ---------------------------------------------------------------------------

QUAT_ANGLE_SRC = OBSERVATIONS.read_text()


def _quat_angle_np(q1, q2):
    """observations._quat_angle, in numpy: 2 atan2(|v|, |w|) of conj(q2) * q1."""
    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], -q2[1:]
    w = w2 * w1 - v2 @ v1
    v = w2 * v1 + w1 * v2 + np.cross(v2, v1)
    return 2.0 * math.atan2(np.linalg.norm(v), abs(w))


def _axis_angle_quat(axis, theta):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.concatenate([[math.cos(theta / 2)], math.sin(theta / 2) * axis])


@pytest.mark.parametrize("deg", [5.0, 20.0, 90.0, 179.0])
@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 0, 1), (1, 1, 1), (0.3, -0.8, 0.2)])
def test_quat_angle_is_axis_independent(deg, axis):
    q0 = _axis_angle_quat((0.2, 0.5, -0.4), 1.1)          # arbitrary goal
    dq = _axis_angle_quat(axis, math.radians(deg))
    w0, v0, w1, v1 = dq[0], dq[1:], q0[0], q0[1:]
    q = np.concatenate([[w0 * w1 - v0 @ v1], w0 * v1 + w1 * v0 + np.cross(v0, v1)])
    assert math.degrees(_quat_angle_np(q, q0)) == pytest.approx(deg, abs=1e-6)
    assert math.degrees(_quat_angle_np(-q, q0)) == pytest.approx(deg, abs=1e-6)  # double cover


@pytest.mark.parametrize("deg", [5.0, 20.0, 45.0])
def test_angle_tolerance_inverts_common_sh(deg):
    """common.sh writes success_tolerance = edge sqrt(3) sin(theta/2); the env tests against
    tol = success_tolerance * keypoint_scale, and angle mode inverts it with the keypoint
    radius r = |corner| * keypoint_scale * edge / 2. That must give back exactly theta."""
    success_tolerance = CUBE_EDGE_M * math.sqrt(3) * math.sin(math.radians(deg) / 2)
    tol = success_tolerance * KEYPOINT_SCALE
    r = np.linalg.norm(KEYPOINT_CORNERS, axis=-1).max() * KEYPOINT_SCALE * CUBE_EDGE_M / 2
    assert math.degrees(2 * math.asin(min(tol / (2 * r), 1.0))) == pytest.approx(deg, abs=1e-9)
    assert "def _quat_angle" in QUAT_ANGLE_SRC and 'orientation_metric == "angle"' in QUAT_ANGLE_SRC
