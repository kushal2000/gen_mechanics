"""The wandb pose viewer (coevolution/pose_viewer.py) for the in-hand olympics env: the same Three.js HTML of one
env's rollout, captured every ``capture_interval`` steps for ``capture_len`` steps, with this env's bodies:

    robot   the hand on its dummy wrist, embedded (its URDF lives in a temp dir; mesh paths become raw GitHub URLs)
    object  the pen: a cylinder with spherical caps (or a box) along +x, its tip red
    goal    the same pen, translucent, at the goal pose
    table   the env's table box
The twist pen (pen.twist): object and goal are its barrel, and two more bodies -- cap (red, a white stripe so its
turning shows) and cap_goal (the same, translucent).

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


def _twist_part_urdf(name: str, x0: float, x1: float, diameter: float, rgba: str, stripe: bool,
                     round_: bool = False) -> str:
    """One part of the twist pen from x0 to x1 along +x (its frame at the pen's centre, as the sim's link frames
    are): a box, or a cylinder (round_); optionally a white stripe along its top."""
    mid, ln, d = (x0 + x1) / 2, x1 - x0, diameter
    geom = (f'<origin xyz="{mid:g} 0 0" rpy="0 1.5707963 0"/><geometry><cylinder radius="{d / 2:g}" length="{ln:g}"/>'
            if round_ else f'<origin xyz="{mid:g} 0 0"/><geometry><box size="{ln:g} {d:g} {d:g}"/>')
    body = (f'<visual>{geom}</geometry>'
            f'<material name="{name}"><color rgba="{rgba}"/></material></visual>')
    if stripe:
        body += (f'<visual><origin xyz="{mid:g} 0 {d / 2 + 0.04 * d:g}"/><geometry><box size="{ln:g} {0.25 * d:g} '
                 f'{0.08 * d:g}"/></geometry><material name="{name}_stripe"><color rgba="1 1 1 1"/></material></visual>')
    return f'<robot name="{name}"><link name="{name}">{body}</link></robot>'


def _table_urdf(size) -> str:
    sx, sy, sz = size
    return (f'<robot name="table"><link name="table"><visual><geometry><box size="{sx:g} {sy:g} {sz:g}"/></geometry>'
            f'<material name="wood"><color rgba="0.55 0.42 0.3 1"/></material></visual></link></robot>')


def _inline_primitive_palm(urdf_text: str) -> str:
    """A generated primitive hand's palm (experiments/14oct_design_search/primitive_hands.py) is a prism mesh,
    ``palm_disc_r<mm>_t<mm>.stl``, which lives in a gitignored folder: the HTML's raw-GitHub URL for it would 404.
    Draw it as the URDF cylinder it approximates instead (axis +x of the palm frame, as the mesh)."""
    import math
    import re
    import xml.etree.ElementTree as ET
    if "palm_disc_r" not in urdf_text:
        return urdf_text
    root = ET.fromstring(urdf_text)
    for el in root.iter():
        for geom in list(el.findall("geometry")):
            mesh = geom.find("mesh")
            m = re.search(r"palm_disc_r(\d+)_t(\d+)\.stl$", mesh.get("filename", "")) if mesh is not None else None
            if not m:
                continue
            geom.remove(mesh)
            ET.SubElement(geom, "cylinder", radius=f"{int(m.group(1)) / 1000:g}", length=f"{int(m.group(2)) / 1000:g}")
            origin = el.find("origin")
            if origin is None:
                origin = ET.SubElement(el, "origin", xyz="0 0 0")
            origin.set("rpy", f"0 {math.pi / 2:.6f} 0")      # the cylinder's z axis onto the palm's +x
    return ET.tostring(root, encoding="unicode")


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
        self._extra_objects = None
        if cfg.pen.twist:
            from .twist_pen import link_extents
            ext = link_extents(cfg.pen.length, cfg.pen.cap_length, 0.001)
            rnd = cfg.pen.shape == "cylinder"
            self._object_urdf_text = _twist_part_urdf("barrel", *ext["barrel"], cfg.pen.diameter, "0.15 0.35 0.8 1",
                                                      stripe=True, round_=rnd)
            # The cap's frame is the pen's centre too: drawn at the cap link's pose, so offset back from its centre.
            cap = _twist_part_urdf("cap", *ext["cap"], cfg.pen.diameter, "0.86 0.24 0.16 1", stripe=True, round_=rnd)
            self._extra_objects = [("cap", cap, 1.0), ("cap_goal", cap, 0.38)]
        self._table_urdf_text, self._table_urdf_path = _table_urdf(cfg.table.size), None
        self._hole_urdf_text = self._hole_urdf_path = None
        self._robot_urdf_text = _inline_primitive_palm(Path(inner.wrist_urdf_path).read_text(encoding="utf-8"))
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
    names, q = list(env.robot.joint_names), env.robot.data.joint_pos[i]
    if env.hand_set is not None:               # a multi-hand scene: env i's own hand (its URDF is the viewer's)
        names = env.robot.real_joint_names(i)
        q = q[: len(names)]
    return {
        "env_id": i,
        "robot_joint_names": names,
        "robot_joint_pos": _to_numpy(q),
        "robot_base_pose": _pose_xyzw(env.robot.data.root_pos_w[i] - origin, env.robot.data.root_quat_w[i]),
        "object_pose": _pose_xyzw(pen_pos[i], pen_quat[i]),
        "goal_pose": _pose_xyzw(env.goal_pos[i], env.goal_quat[i]),
        "table_pose": _pose_xyzw(env.table.data.root_pos_w[i] - origin, env.table.data.root_quat_w[i]),
        **(_twist_frame(env, i) if env.cfg.pen.twist else {}),
    }


def _twist_frame(env, i: int) -> dict:
    """The cap's and its goal's poses, at the cap's LINK frame (the pen's centre) as the cap URDF is drawn: the cap
    goal is stored at the cap's centre, so step back along its axis."""
    from isaaclab.utils.math import quat_apply
    origin = env.scene.env_origins[i]
    q = env.pen.data.body_quat_w[i, env.cap_body]
    gq = env.cap_goal_quat[i]
    goal_link = env.cap_goal_pos[i] - quat_apply(gq.unsqueeze(0), env.cap_centre.unsqueeze(0))[0]
    return {"cap_pose": _pose_xyzw(env.pen.data.body_pos_w[i, env.cap_body] - origin, q),
            "cap_goal_pose": _pose_xyzw(goal_link, gq)}
