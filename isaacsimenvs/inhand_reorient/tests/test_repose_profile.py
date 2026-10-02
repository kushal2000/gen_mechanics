"""The isaaclab_repose profile against NVIDIA's own code, on CPU.

Importing ``isaaclab_tasks`` (or anything under ``isaaclab``) boots Kit, so
NVIDIA's functions are taken from the installed SOURCE instead: the
module-level helpers and the ``InHandManipulationEnv`` methods are parsed out
of ``inhand_manipulation_env.py`` with ``ast``, their decorators dropped, and
executed against ``isaaclab/utils/math.py`` loaded on its own (it imports only
torch/numpy). The methods run on a stand-in ``self``. Every comparison is
ours vs theirs on the same random tensors.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import types
from pathlib import Path

import pytest
import torch
import yaml

from isaacsimenvs.inhand_reorient import repose_profile as rp

REPO_ROOT = Path(__file__).resolve().parents[3]


def _isaaclab_source_root() -> Path:
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("isaaclab is not installed")
    root = Path(list(spec.submodule_search_locations)[0]) / "source"
    if not root.is_dir():
        pytest.skip(f"no Isaac Lab source tree at {root}")
    return root


def _nvidia_paths() -> dict[str, Path]:
    root = _isaaclab_source_root()
    tasks = root / "isaaclab_tasks" / "isaaclab_tasks" / "direct"
    return {
        "math": root / "isaaclab" / "isaaclab" / "utils" / "math.py",
        "env": tasks / "inhand_manipulation" / "inhand_manipulation_env.py",
        "cfg": tasks / "allegro_hand" / "allegro_hand_env_cfg.py",
        "agent": tasks / "allegro_hand" / "agents" / "rl_games_ppo_cfg.yaml",
        "asset": root / "isaaclab_assets" / "isaaclab_assets" / "robots" / "allegro.py",
    }


@pytest.fixture(scope="module")
def nv():
    """NVIDIA's helpers and env methods, as plain Python functions."""
    paths = _nvidia_paths()
    spec = importlib.util.spec_from_file_location("_isaaclab_math_standalone", paths["math"])
    math_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_mod)

    import numpy as np

    namespace = {
        "torch": torch, "np": np, "Sequence": object,
        "quat_conjugate": math_mod.quat_conjugate, "quat_from_angle_axis": math_mod.quat_from_angle_axis,
        "quat_mul": math_mod.quat_mul, "sample_uniform": math_mod.sample_uniform,
        "saturate": math_mod.saturate,
    }
    tree = ast.parse(paths["env"].read_text())
    functions = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
        if isinstance(node, ast.ClassDef) and node.name == "InHandManipulationEnv":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    functions[f"method_{item.name}"] = item
    out = types.SimpleNamespace(math=math_mod)
    for name, fn in functions.items():
        fn = ast.fix_missing_locations(ast.FunctionDef(
            name=fn.name, args=fn.args, body=fn.body, decorator_list=[], returns=None,
            type_comment=None, type_params=[]))
        module = ast.Module(body=[fn], type_ignores=[])
        exec(compile(module, str(paths["env"]), "exec"), namespace)
        setattr(out, name, namespace[fn.name])
    return out


def _random_quat(n, generator=None):
    q = torch.randn(n, 4, generator=generator)
    q = q / q.norm(dim=-1, keepdim=True)
    return q * torch.where(q[:, :1] < 0, -1.0, 1.0)


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------


def test_quaternion_helpers_match_isaaclab_math(nv):
    torch.manual_seed(0)
    q1, q2 = _random_quat(64), _random_quat(64)
    v = torch.randn(64, 3)
    angle, axis = torch.randn(64), torch.randn(64, 3)
    m = nv.math
    torch.testing.assert_close(rp.quat_mul(q1, q2), m.quat_mul(q1, q2))
    torch.testing.assert_close(rp.quat_conjugate(q1), m.quat_conjugate(q1))
    torch.testing.assert_close(rp.quat_from_angle_axis(angle, axis), m.quat_from_angle_axis(angle, axis))
    torch.testing.assert_close(rp.quat_apply(q1, v), m.quat_apply(q1, v))
    torch.testing.assert_close(rp.quat_apply_inverse(q1, v), m.quat_apply_inverse(q1, v))
    lo, hi = -torch.rand(64, 3), torch.rand(64, 3)
    x = torch.randn(64, 3)
    torch.testing.assert_close(rp.saturate(x, lo, hi), m.saturate(x, lo, hi))


def test_scale_unscale_rotation_distance_match(nv):
    torch.manual_seed(1)
    lo = -torch.rand(32, 16) - 0.1
    hi = torch.rand(32, 16) + 0.1
    a = torch.rand(32, 16) * 2 - 1
    torch.testing.assert_close(rp.scale(a, lo, hi), nv.scale(a, lo, hi))
    x = torch.randn(32, 16)
    torch.testing.assert_close(rp.unscale(x, lo, hi), nv.unscale(x, lo, hi))
    q1, q2 = _random_quat(32), _random_quat(32)
    torch.testing.assert_close(rp.rotation_distance(q1, q2), nv.rotation_distance(q1, q2))
    # theirs and isaaclab's quat_error_magnitude (the legacy profile's) agree
    torch.testing.assert_close(
        rp.rotation_distance(q1, q2), nv.math.quat_error_magnitude(q1, q2), atol=1e-5, rtol=1e-5)


def test_randomize_rotation_matches(nv):
    torch.manual_seed(2)
    r0, r1 = torch.rand(50) * 2 - 1, torch.rand(50) * 2 - 1
    x_unit = torch.tensor([1.0, 0.0, 0.0]).repeat(50, 1)
    y_unit = torch.tensor([0.0, 1.0, 0.0]).repeat(50, 1)
    torch.testing.assert_close(rp.randomize_rotation(r0, r1), nv.randomize_rotation(r0, r1, x_unit, y_unit))


# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------

_REWARD_KW = dict(
    dist_reward_scale=-10.0, rot_reward_scale=1.0, rot_eps=0.1, action_penalty_scale=-0.0002,
    success_tolerance=0.2, reach_goal_bonus=250.0, fall_dist=0.24, fall_penalty=0.0, av_factor=0.1,
)


def _reward_inputs(n=256, seed=3):
    g = torch.Generator().manual_seed(seed)
    target_rot = _random_quat(n, g)
    # A third of the objects within ~0.15 rad of their goal (successes), the
    # rest anywhere.
    small = rp.quat_from_angle_axis(torch.rand(n, generator=g) * 0.3, torch.randn(n, 3, generator=g))
    object_rot = torch.where(
        (torch.arange(n) % 3 == 0).unsqueeze(-1), rp.quat_mul(small, target_rot), _random_quat(n, g))
    target_pos = torch.randn(n, 3, generator=g) * 0.1
    object_pos = target_pos + torch.randn(n, 3, generator=g) * 0.15  # some beyond fall_dist
    return dict(
        reset_buf=torch.rand(n, generator=g) < 0.05,
        reset_goal_buf=torch.zeros(n, dtype=torch.bool),
        successes=torch.randint(0, 5, (n,), generator=g).float(),
        consecutive_successes=torch.tensor([1.3]),
        object_pos=object_pos, object_rot=object_rot, target_pos=target_pos, target_rot=target_rot,
        actions=torch.rand(n, 16, generator=g) * 2 - 1,
    )


@pytest.mark.parametrize("fall_penalty", [0.0, -50.0])
def test_compute_rewards_matches_nvidia(nv, fall_penalty):
    inputs = _reward_inputs()
    kw = dict(_REWARD_KW, fall_penalty=fall_penalty)
    ours = rp.compute_rewards(**inputs, **kw)
    theirs = nv.compute_rewards(
        inputs["reset_buf"], inputs["reset_goal_buf"], inputs["successes"], inputs["consecutive_successes"],
        300.0, inputs["object_pos"], inputs["object_rot"], inputs["target_pos"], inputs["target_rot"],
        kw["dist_reward_scale"], kw["rot_reward_scale"], kw["rot_eps"], inputs["actions"],
        kw["action_penalty_scale"], kw["success_tolerance"], kw["reach_goal_bonus"], kw["fall_dist"],
        kw["fall_penalty"], kw["av_factor"],
    )
    for a, b in zip(ours[:4], theirs):
        torch.testing.assert_close(a.float(), b.float())
    goal_hit = ours[1]
    assert 0 < int(goal_hit.sum()) < goal_hit.numel(), "inputs should include both outcomes"
    terms = ours[4]
    total = (terms["rotation_rew"] + terms["distance_rew"] + terms["goal_bonus_rew"]
             + terms["action_penalty"] + terms["drop_penalty"])
    torch.testing.assert_close(total, ours[0])


def test_rewards_are_palm_frame_invariant():
    """Reward inputs in a rotated/translated frame give the same reward:
    our palm-frame rewards equal NVIDIA's world-frame ones for any fixed
    hand pose."""
    inputs = _reward_inputs(seed=4)
    ours = rp.compute_rewards(**inputs, **_REWARD_KW)
    n = inputs["object_pos"].shape[0]
    palm_q = _random_quat(1).expand(n, 4)
    palm_p = torch.randn(1, 3).expand(n, 3)
    moved = dict(inputs)
    moved["object_pos"] = rp.quat_apply_inverse(palm_q, inputs["object_pos"] - palm_p)
    moved["target_pos"] = rp.quat_apply_inverse(palm_q, inputs["target_pos"] - palm_p)
    moved["object_rot"] = rp.quat_mul(rp.quat_conjugate(palm_q), inputs["object_rot"])
    moved["target_rot"] = rp.quat_mul(rp.quat_conjugate(palm_q), inputs["target_rot"])
    moved_out = rp.compute_rewards(**moved, **_REWARD_KW)
    torch.testing.assert_close(moved_out[0], ours[0], atol=1e-4, rtol=1e-5)
    assert torch.equal(moved_out[1], ours[1])


# --------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------


def _obs_world_state(n=8, j=16, k=4, seed=5):
    g = torch.Generator().manual_seed(seed)
    lower = -torch.rand(n, j, generator=g) - 0.1
    upper = torch.rand(n, j, generator=g) + 0.1
    return dict(
        dof_pos=torch.randn(n, j, generator=g), dof_vel=torch.randn(n, j, generator=g) * 3,
        lower=lower, upper=upper,
        object_pos=torch.randn(n, 3, generator=g), object_rot=_random_quat(n, g),
        object_linvel=torch.randn(n, 3, generator=g), object_angvel=torch.randn(n, 3, generator=g) * 5,
        in_hand_pos=torch.randn(n, 3, generator=g), goal_rot=_random_quat(n, g),
        tip_pos=torch.randn(n, k, 3, generator=g), tip_rot=_random_quat(n * k, g).reshape(n, k, 4),
        tip_vel=torch.randn(n, k, 6, generator=g), actions=torch.rand(n, j, generator=g) * 2 - 1,
    )


def _nvidia_obs(nv, st, vel_obs_scale=0.2):
    n, k = st["tip_pos"].shape[:2]
    self = types.SimpleNamespace(
        hand_dof_pos=st["dof_pos"], hand_dof_vel=st["dof_vel"],
        hand_dof_lower_limits=st["lower"], hand_dof_upper_limits=st["upper"],
        cfg=types.SimpleNamespace(vel_obs_scale=vel_obs_scale),
        object_pos=st["object_pos"], object_rot=st["object_rot"], object_linvel=st["object_linvel"],
        object_angvel=st["object_angvel"], in_hand_pos=st["in_hand_pos"], goal_rot=st["goal_rot"],
        fingertip_pos=st["tip_pos"], fingertip_rot=st["tip_rot"], fingertip_velocities=st["tip_vel"],
        actions=st["actions"], num_envs=n, num_fingertips=k,
    )
    return nv.method_compute_full_observations(self)


def _our_obs_from_world(st, palm_pos, palm_quat, **kw):
    """Our observation from world-frame state, via the env's palm-frame
    conversions."""
    inv = rp.quat_conjugate(palm_quat)
    tip_pos, tip_quat, tip_vel = rp.palm_frame_fingertips(
        palm_pos, palm_quat, st["tip_pos"], st["tip_rot"], st["tip_vel"])
    return rp.build_observation(
        joint_pos=st["dof_pos"], joint_vel=st["dof_vel"], lower=st["lower"], upper=st["upper"],
        object_pos=rp.quat_apply_inverse(palm_quat, st["object_pos"] - palm_pos),
        object_quat=rp.quat_mul(inv, st["object_rot"]),
        object_lin_vel=rp.quat_apply_inverse(palm_quat, st["object_linvel"]),
        object_ang_vel=rp.quat_apply_inverse(palm_quat, st["object_angvel"]),
        in_hand_pos=rp.quat_apply_inverse(palm_quat, st["in_hand_pos"] - palm_pos),
        goal_quat=rp.quat_mul(inv, st["goal_rot"]),
        fingertip_pos=tip_pos, fingertip_quat=tip_quat, fingertip_vel=tip_vel,
        actions=st["actions"], vel_obs_scale=0.2, **kw,
    )


def test_observation_matches_nvidia_full_obs_for_an_identity_palm(nv):
    st = _obs_world_state()
    n = st["dof_pos"].shape[0]
    palm_pos = torch.zeros(n, 3)
    palm_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
    ours = _our_obs_from_world(st, palm_pos, palm_quat)
    theirs = _nvidia_obs(nv, st)
    assert ours.shape == theirs.shape == (n, 124)
    torch.testing.assert_close(ours, theirs)


def test_observation_width_is_nvidias_124_for_allegro():
    assert rp.repose_obs_dim(16, 4) == 124
    assert len(rp.REPOSE_OBS_FIELDS) == 13


def test_observation_in_a_rotated_palm_is_nvidias_obs_of_the_palm_frame_state(nv):
    """For a hand rotated/translated in the world, our obs equals NVIDIA's
    obs computed on the same state expressed in the palm frame."""
    st = _obs_world_state(seed=6)
    n, k = st["tip_pos"].shape[:2]
    palm_pos = torch.randn(1, 3).expand(n, 3)
    palm_quat = _random_quat(1).expand(n, 4)
    ours = _our_obs_from_world(st, palm_pos, palm_quat)
    inv = rp.quat_conjugate(palm_quat)
    q_k = palm_quat.unsqueeze(1).expand(n, k, 4).reshape(-1, 4)
    palm_st = dict(st)
    palm_st["object_pos"] = rp.quat_apply_inverse(palm_quat, st["object_pos"] - palm_pos)
    palm_st["object_rot"] = rp.quat_mul(inv, st["object_rot"])
    palm_st["object_linvel"] = rp.quat_apply_inverse(palm_quat, st["object_linvel"])
    palm_st["object_angvel"] = rp.quat_apply_inverse(palm_quat, st["object_angvel"])
    palm_st["in_hand_pos"] = rp.quat_apply_inverse(palm_quat, st["in_hand_pos"] - palm_pos)
    palm_st["goal_rot"] = rp.quat_mul(inv, st["goal_rot"])
    palm_st["tip_pos"] = rp.quat_apply_inverse(
        q_k, (st["tip_pos"] - palm_pos.unsqueeze(1)).reshape(-1, 3)).reshape(n, k, 3)
    palm_st["tip_rot"] = rp.quat_mul(rp.quat_conjugate(q_k), st["tip_rot"].reshape(-1, 4)).reshape(n, k, 4)
    palm_st["tip_vel"] = torch.cat([
        rp.quat_apply_inverse(q_k, st["tip_vel"][..., :3].reshape(-1, 3)).reshape(n, k, 3),
        rp.quat_apply_inverse(q_k, st["tip_vel"][..., 3:].reshape(-1, 3)).reshape(n, k, 3)], dim=-1)
    torch.testing.assert_close(ours, _nvidia_obs(nv, palm_st), atol=1e-5, rtol=1e-5)


def test_observation_masks_ghost_joints_and_absent_fingertips():
    st = _obs_world_state(seed=7)
    n, j = st["dof_pos"].shape
    k = st["tip_pos"].shape[1]
    # A ghost joint's limits are (0, 1e-8): unscale explodes; the mask must
    # zero it without producing a NaN.
    st["lower"][:, 3] = 0.0
    st["upper"][:, 3] = 0.0
    joint_mask = torch.ones(n, j, dtype=torch.bool)
    joint_mask[:, 3] = False
    tip_mask = torch.ones(n, k, dtype=torch.bool)
    tip_mask[:, 1] = False
    palm_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
    obs = _our_obs_from_world(st, torch.zeros(n, 3), palm_quat, joint_mask=joint_mask, fingertip_mask=tip_mask)
    assert torch.isfinite(obs).all()
    widths = [rp.repose_field_width(f, j, k) for f in rp.REPOSE_OBS_FIELDS]
    fields = dict(zip(rp.REPOSE_OBS_FIELDS, torch.split(obs, widths, dim=-1)))
    for name in ("joint_pos_unscaled", "joint_vel_scaled", "actions"):
        assert (fields[name][:, 3] == 0).all()
        assert (fields[name][:, 2] != 0).all()
    for name, w in (("fingertip_pos_palm", 3), ("fingertip_quat_palm", 4), ("fingertip_vel_palm", 6)):
        per_tip = fields[name].reshape(n, k, w)
        assert (per_tip[:, 1] == 0).all()
        assert (per_tip[:, 0] != 0).any()


# --------------------------------------------------------------------------
# Control, dones, reset
# --------------------------------------------------------------------------


@pytest.mark.parametrize("act_moving_average", [1.0, 0.3])
def test_joint_targets_match_nvidia_apply_action(nv, act_moving_average):
    torch.manual_seed(8)
    n, j = 16, 16
    lower, upper = -torch.rand(n, j) - 0.1, torch.rand(n, j) + 0.1
    actions = torch.rand(n, j) * 2.4 - 1.2  # some beyond [-1, 1]: the clamp
    prev = torch.rand(n, j) * (upper - lower) + lower
    captured = {}
    self = types.SimpleNamespace(
        actions=actions, cfg=types.SimpleNamespace(act_moving_average=act_moving_average),
        cur_targets=torch.zeros(n, j), prev_targets=prev.clone(), actuated_dof_indices=list(range(j)),
        hand_dof_lower_limits=lower, hand_dof_upper_limits=upper,
        hand=types.SimpleNamespace(set_joint_position_target=lambda t, joint_ids: captured.update(t=t.clone())),
    )
    nv.method__apply_action(self)
    ours = rp.joint_targets(actions, prev, lower, upper, act_moving_average)
    torch.testing.assert_close(ours, captured["t"])
    torch.testing.assert_close(ours, self.prev_targets)


def test_fall_and_timeout_match_nvidia_get_dones(nv):
    torch.manual_seed(9)
    n = 64
    in_hand = torch.randn(n, 3) * 0.1
    obj = in_hand + torch.randn(n, 3) * 0.2
    elb = torch.randint(0, 301, (n,))
    self = types.SimpleNamespace(
        _compute_intermediate_values=lambda: None, object_pos=obj, in_hand_pos=in_hand,
        cfg=types.SimpleNamespace(fall_dist=0.24, max_consecutive_success=0),
        episode_length_buf=elb, max_episode_length=300,
    )
    theirs = nv.method__get_dones(self)
    ours = rp.fall_and_timeout(obj, in_hand, elb, 300, 0.24)
    assert torch.equal(ours[0], theirs[0]) and torch.equal(ours[1], theirs[1])
    assert ours[0].any() and (~ours[0]).any() and ours[1].any()


def test_reset_matches_nvidia_reset_idx_under_the_same_seed(nv):
    """NVIDIA's _reset_idx (minus DirectRLEnv's own part) against the
    profile's reset arithmetic, drawing the same random numbers in the same
    order as repose_hooks.reset_env_state does."""
    n, j = 12, 16
    torch.manual_seed(10)
    default_pos = torch.rand(n, j) * 0.4
    lower = default_pos - torch.rand(n, j) - 0.05
    upper = default_pos + torch.rand(n, j) + 0.05
    default_root = torch.zeros(n, 13)
    default_root[:, 0:3] = torch.tensor([0.0, -0.17, 0.56])
    default_root[:, 3] = 1.0
    env_origins = torch.randn(n, 3)
    env_ids = torch.arange(n)
    writes = {}

    def _sample_uniform(lo, hi, size, device):
        return nv.math.sample_uniform(lo, hi, size, device)

    tree = ast.parse(_nvidia_paths()["env"].read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "InHandManipulationEnv")
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_reset_idx")
    # Drop `super()._reset_idx(env_ids)` (DirectRLEnv's own bookkeeping).
    body = [s for s in fn.body if "super()" not in ast.unparse(s)]
    fn = ast.fix_missing_locations(ast.FunctionDef(
        name="_reset_idx", args=fn.args, body=body, decorator_list=[], returns=None,
        type_comment=None, type_params=[]))
    import numpy as np

    ns = {"torch": torch, "np": np, "sample_uniform": _sample_uniform,
          "randomize_rotation": nv.randomize_rotation, "Sequence": object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "nvidia_reset_idx", "exec"), ns)

    goal_rot = torch.zeros(n, 4)
    x_unit = torch.tensor([1.0, 0.0, 0.0]).repeat(n, 1)
    y_unit = torch.tensor([0.0, 1.0, 0.0]).repeat(n, 1)

    def _reset_target_pose(ids):
        rand = _sample_uniform(-1.0, 1.0, (len(ids), 2), device="cpu")
        goal_rot[ids] = nv.randomize_rotation(rand[:, 0], rand[:, 1], x_unit[ids], y_unit[ids])

    self = types.SimpleNamespace(
        hand=types.SimpleNamespace(
            _ALL_INDICES=env_ids,
            data=types.SimpleNamespace(default_joint_pos=default_pos, default_joint_vel=torch.zeros(n, j)),
            set_joint_position_target=lambda t, env_ids: writes.update(target=t.clone()),
            write_joint_state_to_sim=lambda p, v, env_ids: writes.update(pos=p.clone(), vel=v.clone()),
        ),
        object=types.SimpleNamespace(
            data=types.SimpleNamespace(default_root_state=default_root),
            write_root_pose_to_sim=lambda pose, ids: writes.update(obj_pose=pose.clone()),
            write_root_velocity_to_sim=lambda vel, ids: writes.update(obj_vel=vel.clone()),
        ),
        _reset_target_pose=_reset_target_pose,
        cfg=types.SimpleNamespace(reset_position_noise=0.01, reset_dof_pos_noise=0.2, reset_dof_vel_noise=0.0),
        scene=types.SimpleNamespace(env_origins=env_origins),
        x_unit_tensor=x_unit, y_unit_tensor=y_unit,
        hand_dof_upper_limits=upper, hand_dof_lower_limits=lower, num_hand_dofs=j, device="cpu",
        prev_targets=torch.zeros(n, j), cur_targets=torch.zeros(n, j), hand_dof_targets=torch.zeros(n, j),
        successes=torch.ones(n), _compute_intermediate_values=lambda: None,
    )
    torch.manual_seed(11)
    ns["_reset_idx"](self, env_ids)
    theirs_goal = goal_rot.clone()

    # Ours, in repose_hooks.reset_env_state's draw order.
    sample_uniform = rp.sample_uniform
    torch.manual_seed(11)
    rand = sample_uniform(-1.0, 1.0, (n, 2), device="cpu")
    our_goal = rp.randomize_rotation(rand[:, 0], rand[:, 1])
    pos_noise = sample_uniform(-1.0, 1.0, (n, 3), device="cpu")
    spawn_w = default_root[:, 0:3] + env_origins
    our_obj_pos = spawn_w + 0.01 * pos_noise
    rot_noise = sample_uniform(-1.0, 1.0, (n, 2), device="cpu")
    our_obj_rot = rp.randomize_rotation(rot_noise[:, 0], rot_noise[:, 1])
    dof_pos_noise = sample_uniform(-1.0, 1.0, (n, j), device="cpu")
    our_dof = rp.reset_joint_positions(default_pos, lower, upper, dof_pos_noise, 0.2)
    dof_vel_noise = sample_uniform(-1.0, 1.0, (n, j), device="cpu")
    our_vel = torch.zeros(n, j) + 0.0 * dof_vel_noise

    torch.testing.assert_close(our_goal, theirs_goal)
    torch.testing.assert_close(our_obj_pos, writes["obj_pose"][:, 0:3])
    torch.testing.assert_close(our_obj_rot, writes["obj_pose"][:, 3:7])
    torch.testing.assert_close(our_dof, writes["pos"])
    torch.testing.assert_close(our_dof, writes["target"])
    torch.testing.assert_close(our_vel, writes["vel"])
    assert (self.successes == 0).all()


# --------------------------------------------------------------------------
# Config: our repose block vs NVIDIA's installed config
# --------------------------------------------------------------------------


def _nvidia_cfg_constants() -> dict:
    """Literal class attributes of AllegroHandEnvCfg (ints/floats/strings)."""
    tree = ast.parse(_nvidia_paths()["cfg"].read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "AllegroHandEnvCfg")
    out = {}
    for item in cls.body:
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            target = item.targets[0] if isinstance(item, ast.Assign) else item.target
            if isinstance(target, ast.Name) and item.value is not None:
                try:
                    out[target.id] = ast.literal_eval(item.value)
                except ValueError:
                    out[target.id] = ast.unparse(item.value)
    return out


def _task_yaml() -> dict:
    return yaml.safe_load((REPO_ROOT / "coevolution/cfg/task/InHandReorient.yaml").read_text())


def test_repose_block_matches_nvidia_allegro_cfg():
    nvc = _nvidia_cfg_constants()
    ours = _task_yaml()["repose"]
    same_name = (
        "decimation", "episode_length_s", "reset_position_noise", "reset_dof_pos_noise",
        "reset_dof_vel_noise", "dist_reward_scale", "rot_reward_scale", "rot_eps", "action_penalty_scale",
        "reach_goal_bonus", "fall_penalty", "fall_dist", "vel_obs_scale", "success_tolerance",
        "max_consecutive_success", "av_factor", "act_moving_average",
    )
    for name in same_name:
        assert float(ours[name]) == pytest.approx(float(nvc[name])), name
    src = _nvidia_paths()["cfg"].read_text()
    assert "dt=1 / 120" in src and ours["sim_dt"] == pytest.approx(1 / 120)
    assert "static_friction=1.0" in src and ours["static_friction"] == 1.0
    assert "dynamic_friction=1.0" in src and ours["dynamic_friction"] == 1.0
    assert "bounce_threshold_velocity=0.2" in src and ours["bounce_threshold_velocity"] == 0.2
    assert "density=400.0" in src and ours["object_density"] == 400.0
    assert "scale=(1.2, 1.2, 1.2)" in src and ours["object_size_m"] == pytest.approx(0.06 * 1.2)
    assert "solver_position_iteration_count=8" in src and ours["object_solver_position_iterations"] == 8
    assert "max_depenetration_velocity=1000.0" in src
    assert "in_hand_pos[:, 2] -= 0.04" in _nvidia_paths()["env"].read_text()
    assert list(ours["in_hand_pos_offset"]) == [0.0, 0.0, -0.04]
    assert nvc["obs_type"] == "full" and nvc["asymmetric_obs"] is False


def test_repose_hand_props_match_nvidia_allegro_asset():
    src = _nvidia_paths()["asset"].read_text()
    ours = _task_yaml()["repose"]
    assert "disable_gravity=True" in src and ours["hand_disable_gravity"] is True
    assert "angular_damping=0.01" in src and ours["hand_angular_damping"] == 0.01
    assert "solver_position_iteration_count=8" in src and ours["hand_solver_position_iterations"] == 8
    assert "solver_velocity_iteration_count=0" in src and ours["hand_solver_velocity_iterations"] == 0
    assert "stabilization_threshold=0.0005" in src and ours["hand_stabilization_threshold"] == 0.0005
    assert "sleep_threshold=0.005" in src and ours["hand_sleep_threshold"] == 0.005


def test_train_yaml_is_nvidias_agent():
    theirs = yaml.safe_load(_nvidia_paths()["agent"].read_text())["params"]
    ours = yaml.safe_load((REPO_ROOT / "coevolution/cfg/train/InHandReposeIsaacLabPPO.yaml").read_text())["params"]
    assert ours["env"] == theirs["env"]
    assert ours["algo"] == theirs["algo"] and ours["model"] == theirs["model"]
    assert ours["network"]["mlp"] == theirs["network"]["mlp"]
    assert ours["network"]["name"] == theirs["network"]["name"]
    assert theirs["network"]["space"]["continuous"]["fixed_sigma"] is True
    assert ours["network"]["space"]["continuous"]["fixed_sigma"] == "fixed"
    fork_only = {"name", "full_experiment_name", "num_actors", "expl_type", "use_others_experience"}
    for key, value in theirs["config"].items():
        if key in fork_only:
            continue
        assert ours["config"][key] == value, key
    assert set(ours["config"]) - set(theirs["config"]) <= fork_only
    assert ours["config"]["name"].split("_")[0].isdigit()


def test_population_train_yaml_is_the_agent_plus_the_logstd_bound():
    base = yaml.safe_load((REPO_ROOT / "coevolution/cfg/train/InHandReposeIsaacLabPPO.yaml").read_text())["params"]
    pop = yaml.safe_load((REPO_ROOT / "coevolution/cfg/train/InHandReposeIsaacLabPopPPO.yaml").read_text())["params"]
    assert pop["network"]["name"] == "inhand_actor_critic"
    assert pop["network"]["space"]["continuous"]["logstd_max"] == 0.0
    assert pop["config"]["entropy_coef"] == 0.0
    for key, value in base["config"].items():
        if key in ("name", "full_experiment_name"):
            continue
        assert pop["config"][key] == value, key
    assert pop["network"]["mlp"] == base["network"]["mlp"]


def test_apply_profile_to_cfg_sets_env_level_fields_and_leaves_legacy_alone():
    def _cfg(profile):
        repose = types.SimpleNamespace(**{
            "decimation": 4, "episode_length_s": 10.0, "sim_dt": 1 / 120, "static_friction": 1.0,
            "dynamic_friction": 1.0, "restitution": 0.0, "bounce_threshold_velocity": 0.2,
        })
        return types.SimpleNamespace(
            task_profile=profile, repose=repose, decimation=2, episode_length_s=5.0,
            sim=types.SimpleNamespace(
                dt=0.01, render_interval=2,
                physics_material=types.SimpleNamespace(static_friction=0.5, dynamic_friction=0.5, restitution=0.0),
                physx=types.SimpleNamespace(bounce_threshold_velocity=0.5)),
            obs=types.SimpleNamespace(obs_list=("joint_pos",), state_list=("joint_pos",)),
        )

    cfg = _cfg("isaaclab_repose")
    rp.apply_profile_to_cfg(cfg)
    assert cfg.decimation == 4 and cfg.sim.render_interval == 4
    assert cfg.sim.dt == pytest.approx(1 / 120) and cfg.episode_length_s == 10.0
    assert cfg.sim.physics_material.static_friction == 1.0 == cfg.sim.physics_material.dynamic_friction
    assert cfg.sim.physx.bounce_threshold_velocity == 0.2
    assert cfg.obs.obs_list == rp.REPOSE_OBS_FIELDS == cfg.obs.state_list

    legacy = _cfg("legacy")
    rp.apply_profile_to_cfg(legacy)
    assert legacy.decimation == 2 and legacy.sim.dt == 0.01 and legacy.obs.obs_list == ("joint_pos",)
    assert legacy.sim.physics_material.static_friction == 0.5

    with pytest.raises(ValueError):
        rp.is_repose(_cfg("isaaclab"))


def test_task_yaml_defaults_to_the_legacy_profile_and_keeps_legacy_values():
    task = _task_yaml()
    assert task["task_profile"] == "legacy"
    # The legacy profile's own values are still the top-level ones.
    assert task["decimation"] == 2 and task["reward"]["drop_penalty"] == 50.0
    assert task["reset"]["goal_curriculum_stages"] == ["axis"]
    assert task["termination"]["success_tolerance"] == 0.4


def test_episode_is_300_policy_steps_at_30hz():
    r = _task_yaml()["repose"]
    steps = math.ceil(r["episode_length_s"] / (r["sim_dt"] * r["decimation"]))
    assert steps == 300


# --------------------------------------------------------------------------
# Hand poses for the profile (repose_hand_poses.json)
# --------------------------------------------------------------------------


def test_repose_hand_pose_file_reproduces_nvidias_allegro_cube_spawn():
    """allegro_right's entry places our hand so the cube spawn point is
    NVIDIA's (0, -0.17, 0.56) in the env frame, with the hand's palm
    facing up, and NVIDIA's default joint pose (0, thumb base 0.28)."""
    from isaacsimenvs.inhand_reorient import palm_calibration as pc

    poses = pc.load_repose_hand_poses(pc.resolve_repose_hand_pose_path(_task_yaml()["repose"]["hand_pose_file"]))
    entry = poses["allegro_right"]
    rot = torch.tensor([entry["base_rot"]], dtype=torch.float64)
    assert abs(float(rot.norm()) - 1.0) < 1e-5
    spawn_env = torch.tensor(entry["base_pos"], dtype=torch.float64) + rp.quat_apply(
        rot, torch.tensor([entry["spawn_offset_local"]], dtype=torch.float64))[0]
    cfg_src = _nvidia_paths()["cfg"].read_text()
    assert "pos=(0.0, -0.17, 0.56)" in cfg_src
    torch.testing.assert_close(spawn_env, torch.tensor([0.0, -0.17, 0.56], dtype=torch.float64), atol=1e-4, rtol=0)
    # The Allegro palm faces local +x: it must point up.
    palm_normal_world = rp.quat_apply(rot, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))[0]
    assert palm_normal_world[2] > 0.99
    assert "thumb_joint_0\": 0.28" in _nvidia_paths()["asset"].read_text().replace("'", "\"")
    defaults = entry["hand_default_joint_pos"]
    assert defaults.pop("joint_12") == 0.28 and set(defaults.values()) == {0.0} and len(defaults) == 15


def test_repose_cube_mass_is_nvidias_effective_mass():
    """The DexCube USD authors physics:mass 0.216, which wins over the
    configured density 400 (0.149 kg); 0.216 kg is what NVIDIA's env
    simulates (read back from its PhysX view in Kit)."""
    r = _task_yaml()["repose"]
    assert r["object_mass_kg"] == 0.216
    assert 400.0 * r["object_size_m"] ** 3 == pytest.approx(0.1493, abs=1e-4)


def test_every_repose_hand_pose_spawns_the_cube_above_the_palm():
    from isaacsimenvs.inhand_reorient import palm_calibration as pc

    poses = pc.load_repose_hand_poses(pc.resolve_repose_hand_pose_path(_task_yaml()["repose"]["hand_pose_file"]))
    assert {"allegro_right", "sharpa"} <= set(poses)
    for hand, entry in poses.items():
        rot = torch.tensor([entry["base_rot"]], dtype=torch.float64)
        lift = rp.quat_apply(rot, torch.tensor([entry["spawn_offset_local"]], dtype=torch.float64))[0, 2]
        assert lift >= pc.MIN_SPAWN_HEIGHT_ABOVE_PALM_M, hand
        assert entry["source"], hand


def test_repose_hand_actuator_defaults_are_nvidias_allegro():
    src = _nvidia_paths()["asset"].read_text()
    r = _task_yaml()["repose"]
    for text in ("stiffness=3.0", "damping=0.1", "friction=0.01", "effort_limit_sim=0.5"):
        assert text in src, text
    assert (r["hand_stiffness"], r["hand_damping"], r["hand_joint_friction"], r["hand_effort_limit"]) == (
        3.0, 0.1, 0.01, 0.5)
    # Armature and velocity limit come from NVIDIA's USD (read in Kit: 0.0 and
    # 6.283 rad/s, i.e. physxJoint:maxJointVelocity 359.99 deg/s).
    assert r["hand_armature"] == 0.0 and r["hand_velocity_limit"] == pytest.approx(2 * math.pi, abs=1e-3)


def test_allegro_right_collides_through_its_visual_meshes_under_repose():
    from isaacsimenvs.inhand_reorient import palm_calibration as pc

    poses = pc.load_repose_hand_poses(pc.resolve_repose_hand_pose_path(_task_yaml()["repose"]["hand_pose_file"]))
    assert poses["allegro_right"]["collision_from_visuals"] is True
    assert not poses["sharpa"].get("collision_from_visuals", False)  # SHARPA's URDF already collides through meshes
    assert _task_yaml()["repose"]["collision_from_visuals"] is False
