"""The wandb pose viewer (coevolution/pose_viewer.py) for the in-hand olympics env: the same Three.js HTML of one
env's rollout, captured every ``capture_interval`` steps for ``capture_len`` steps, with this env's bodies:

    robot   the hand on its dummy wrist, embedded (its URDF lives in a temp dir; mesh paths become raw GitHub URLs)
    object  the pen: a cylinder with spherical caps (or a box) along +x, its tip red
    goal    the same pen, translucent, at the goal pose
    table   the env's table box

train.py picks this class up from ``env.POSE_VIEWER`` in place of the PoseReach one.
"""
from __future__ import annotations

from pathlib import Path

from coevolution.pose_viewer import PoseViewerWrapper, _pose_xyzw, _to_numpy


def _pen_box_urdf(length: float, diameter: float) -> str:
    """A box pen along +x, its tip end (a thin slab) red."""
    tip = 0.15 * diameter
    return (f'<robot name="pen"><link name="pen">'
            f'<visual><origin xyz="{-tip / 2:g} 0 0"/><geometry><box size="{length - tip:g} {diameter:g} {diameter:g}"/>'
            f'</geometry><material name="body"><color rgba="0.15 0.35 0.8 1"/></material></visual>'
            f'<visual><origin xyz="{length / 2 - tip / 2:g} 0 0"/><geometry><box size="{tip:g} {diameter:g} {diameter:g}"/>'
            f'</geometry><material name="tip"><color rgba="0.86 0.16 0.16 1"/></material></visual></link></robot>')


def _pen_urdf(length: float, diameter: float) -> str:
    r, h = diameter / 2, length - diameter
    cap = lambda x, rgba: (f'<visual><origin xyz="{x:g} 0 0"/><geometry><sphere radius="{r:g}"/></geometry>'  # noqa
                           f'<material name="m{x:g}"><color rgba="{rgba}"/></material></visual>')
    return (f'<robot name="pen"><link name="pen">'
            f'<visual><origin rpy="0 1.5707963 0"/><geometry><cylinder radius="{r:g}" length="{h:g}"/></geometry>'
            f'<material name="body"><color rgba="0.15 0.35 0.8 1"/></material></visual>'
            f'{cap(h / 2, "0.86 0.16 0.16 1")}{cap(-h / 2, "0.15 0.35 0.8 1")}</link></robot>')


def _table_urdf(size) -> str:
    sx, sy, sz = size
    return (f'<robot name="table"><link name="table"><visual><geometry><box size="{sx:g} {sy:g} {sz:g}"/></geometry>'
            f'<material name="wood"><color rgba="0.55 0.42 0.3 1"/></material></visual></link></robot>')


class OlympicsPoseViewer(PoseViewerWrapper):
    def __init__(self, env, *, output_dir, capture_len: int, capture_interval: int, env_id: int = 0,
                 wandb_key: str = "interactive_viewer", github_raw_base: str | None = None,
                 url_check: str = "skip") -> None:
        import gymnasium as gym
        gym.Wrapper.__init__(self, env)
        inner = self.env.unwrapped
        if not 0 <= env_id < inner.num_envs:
            raise ValueError(f"capture_viewer_env_id={env_id} out of range for num_envs={inner.num_envs}")
        self.output_dir = Path(output_dir)
        self.capture_len, self.capture_interval = int(capture_len), int(capture_interval)
        self.env_id, self.wandb_key = int(env_id), wandb_key
        self.github_raw_base, self.url_check = github_raw_base, url_check
        cfg = inner.cfg
        pen_urdf = _pen_box_urdf if cfg.pen.shape == "box" else _pen_urdf
        self._object_urdf_text, self._object_urdf_path = pen_urdf(cfg.pen.length, cfg.pen.diameter), None
        self._table_urdf_text, self._table_urdf_path = _table_urdf(cfg.table.size), None
        self._hole_urdf_text = self._hole_urdf_path = None
        self._robot_urdf_text = Path(inner.wrist_urdf_path).read_text(encoding="utf-8")
        self._robot_urdf_relpath = None
        self._step, self._capture_index, self._frames = 0, 0, []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[pose_viewer] olympics: env_id={self.env_id} len={self.capture_len} "
              f"interval={self.capture_interval} output_dir={self.output_dir}", flush=True)

    def step(self, action):
        result = self.env.step(action)
        self._step += 1
        if self._frames is None and self.capture_interval > 0 and self._step % self.capture_interval == 0:
            self._frames = []
        if self._frames is not None:
            self._frames.append(capture_frame(self.env.unwrapped, self.env_id))
            if len(self._frames) >= self.capture_len:
                self._finalize_capture()
        return result


def capture_frame(env, i: int) -> dict:
    origin = env.scene.env_origins[i]
    pen_pos, pen_quat = env.pen_pose()
    return {
        "env_id": i,
        "robot_joint_names": list(env.robot.joint_names),
        "robot_joint_pos": _to_numpy(env.robot.data.joint_pos[i]),
        "robot_base_pose": _pose_xyzw(env.robot.data.root_pos_w[i] - origin, env.robot.data.root_quat_w[i]),
        "object_pose": _pose_xyzw(pen_pos[i], pen_quat[i]),
        "goal_pose": _pose_xyzw(env.goal_pos[i], env.goal_quat[i]),
        "table_pose": _pose_xyzw(env.table.data.root_pos_w[i] - origin, env.table.data.root_quat_w[i]),
    }
