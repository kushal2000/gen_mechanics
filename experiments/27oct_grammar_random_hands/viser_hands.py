"""Look at a list of hand URDFs in viser: pick one, move its joints, or lay them all out side by side. CPU only.

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/viser_hands.py \
        assets/urdf/27oct_grammar_random_hands [more URDFs, .spec.json files, folders or globs ...] [--port 8099]

Each input is a URDF, a ``.spec.json`` (its ``urdf``), a folder (every URDF under it), or a glob. What is drawn is
the URDF's visual geometry; a cylinder is drawn as the capsule the simulator makes of it when the hand's spec says
``replace_cylinders_with_capsules`` (the grammar hands' fingertips, the primitive hands' modules).

"repose env" (on by default) lays the scene out as the in-hand olympics pen-repose env builds it: the table, the box
pen at its spawn pose (lying flat on the table at the origin; the yaw slider turns it, as resets do), and the hand on
its dummy wrist, mounted palm down exactly as ``isaacsimenvs/inhand_olympics/wrist.py`` mounts it, at the wrist's
default pose. The wrist sliders move it as the repose policy can: translation only, within the wrist's ranges (its
rotation is locked, ``wrist_max_speed_ang=0``). The fingers start at the spec's default joint positions (training
resets them uniformly over their ranges: "random pose"). Off: the hand alone, palm down, the pen under it for scale.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
import yourdfpy

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from isaacsimenvs.inhand_olympics.wrist import palm_down_rotation  # noqa: E402  (Kit-free)

# The olympics env's defaults (isaacsimenvs/inhand_olympics/env_cfg.py: TableCfg, PenCfg with shape=box, WristCfg),
# copied because env_cfg needs Kit.
TABLE_SIZE, TABLE_TOP = (0.8, 0.8, 0.04), 0.40
PEN = (0.10, 0.03, 0.03)                      # box pen: length, diameter, diameter (m)
PEN_INIT_XY = (0.0, 0.0)
WRIST_RANGES = {"x": (-0.10, 0.10), "y": (-0.10, 0.10), "z": (0.02, 0.20)}
WRIST_DEFAULT = {"x": 0.0, "y": 0.0, "z": 0.15}   # WristCfg.default_pose; the wrist base sits on the table top
GRID_PITCH = 0.40                             # m between hands in the "all" layout


def find_urdfs(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        matches = [Path(p) for p in sorted(glob.glob(item))] or [Path(item)]
        for p in matches:
            if p.is_dir():
                out += sorted(p.rglob("*.urdf"))
            elif p.name.endswith(".spec.json"):
                u = Path(json.loads(p.read_text())["urdf"])
                out.append(u if u.is_absolute() else REPO / u)
            elif p.suffix == ".urdf":
                out.append(p)
            else:
                raise SystemExit(f"not a URDF, .spec.json or folder: {item}")
    seen, uniq = set(), []
    for p in out:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p.resolve())
    if not uniq:
        raise SystemExit("no URDFs found")
    return uniq


def spec_for(urdf: Path) -> dict:
    s = urdf.with_name(urdf.stem + ".spec.json")
    return json.loads(s.read_text()) if s.exists() else {}


def _capsule(radius: float, length: float) -> trimesh.Trimesh:
    return trimesh.creation.capsule(height=length, radius=radius, count=[16, 16])     # along z, centred


class Hand:
    """One URDF: its kinematics (yourdfpy) and one mesh per visual, in its link's frame."""

    def __init__(self, urdf: Path, capsules: bool | None = None):
        self.path, self.name = urdf, urdf.stem
        self.spec = spec_for(urdf)
        if capsules is None:
            capsules = bool(self.spec.get("replace_cylinders_with_capsules", False))
        self.u = yourdfpy.URDF.load(str(urdf), load_meshes=False, build_scene_graph=True)
        self.root = self.u.base_link
        self.visuals: list[tuple[str, trimesh.Trimesh, np.ndarray, tuple]] = []   # link, mesh, origin, rgb
        for link in self.u.robot.links:
            for v in link.visuals:
                g, O = v.geometry, (v.origin if v.origin is not None else np.eye(4))
                if g.box is not None:
                    m = trimesh.creation.box(extents=g.box.size)
                elif g.cylinder is not None:
                    m = (_capsule(g.cylinder.radius, g.cylinder.length) if capsules else
                         trimesh.creation.cylinder(radius=g.cylinder.radius, height=g.cylinder.length, sections=24))
                elif g.sphere is not None:
                    m = trimesh.creation.icosphere(subdivisions=2, radius=g.sphere.radius)
                elif g.mesh is not None:
                    fn = g.mesh.filename.replace("package://", "")
                    p = Path(fn) if Path(fn).is_absolute() else urdf.parent / fn
                    m = trimesh.load(str(p), force="mesh")
                    if g.mesh.scale is not None:
                        m.apply_scale(g.mesh.scale)
                else:
                    continue
                rgba = (v.material.color.rgba if v.material is not None and v.material.color is not None
                        else (0.7, 0.7, 0.7, 1.0))
                self.visuals.append((link.name, m, O, tuple(int(255 * c) for c in rgba[:3])))
        self.joints = [(n, self.u.joint_map[n]) for n in self.u.actuated_joint_names]
        mass = sum(l.inertial.mass for l in self.u.robot.links if l.inertial is not None and l.inertial.mass)
        self.info = (f"**{self.name}**  \n{len(self.joints)} joints, "
                     f"{len(self.spec.get('fingertip_body_names', [])) or '?'} fingertips, {mass * 1e3:.0f} g  \n"
                     f"`{self.path.relative_to(REPO) if self.path.is_relative_to(REPO) else self.path}`")

    def limits(self, name: str) -> tuple[float, float]:
        lim = self.u.joint_map[name].limit
        lo = lim.lower if lim is not None and lim.lower is not None else -math.pi
        hi = lim.upper if lim is not None and lim.upper is not None else math.pi
        return float(lo), float(hi)

    def link_poses(self, q: dict[str, float]) -> dict[str, np.ndarray]:
        self.u.update_cfg(q)
        return {l.name: self.u.get_transform(l.name, self.root) for l in self.u.robot.links}


def _wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in trimesh.transformations.quaternion_from_matrix(
        np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="URDFs, .spec.json files, folders or globs")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    paths = find_urdfs(args.inputs)
    print(f"{len(paths)} hands", flush=True)
    cache: dict[Path, Hand] = {}

    def hand(i: int) -> Hand:
        if paths[i] not in cache:
            cache[paths[i]] = Hand(paths[i])
        return cache[paths[i]]

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    names = [p.stem for p in paths]
    g_hand = server.gui.add_dropdown("hand", names, initial_value=names[0])
    g_prev, g_next = server.gui.add_button("previous"), server.gui.add_button("next")
    g_env = server.gui.add_checkbox("repose env", True)
    with server.gui.add_folder("wrist (translation only)"):
        g_wrist = {k: server.gui.add_slider(f"wrist {k}", lo, hi, 0.005, WRIST_DEFAULT[k])
                   for k, (lo, hi) in WRIST_RANGES.items()}
        g_wrist_reset = server.gui.add_button("wrist to default")
    g_yaw = server.gui.add_slider("pen yaw (deg)", -180.0, 180.0, 1.0, 0.0)
    g_all = server.gui.add_checkbox(f"show all {len(paths)} (zero pose)", False)
    g_info = server.gui.add_markdown("")
    folder = {"f": None}
    state: dict = {"handles": {}, "sliders": {}}

    def mount(h: Hand) -> np.ndarray:
        """The hand root's orientation: palm down, as wrist.py mounts it (the palm frame from the hand's spec)."""
        return palm_down_rotation(np.asarray(h.spec.get("palm_frame", np.eye(3)), float))

    def wrist_pos() -> np.ndarray:
        return np.array([g_wrist["x"].value, g_wrist["y"].value, TABLE_TOP + g_wrist["z"].value])

    def draw_env() -> None:
        server.scene.add_box("/table", dimensions=TABLE_SIZE, color=(150, 120, 90),
                             position=(0.0, 0.0, TABLE_TOP - TABLE_SIZE[2] / 2))
        yaw = math.radians(g_yaw.value)
        server.scene.add_box("/pen", dimensions=PEN, color=(40, 90, 200),
                             wxyz=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
                             position=(PEN_INIT_XY[0], PEN_INIT_XY[1], TABLE_TOP + PEN[2] / 2))

    def draw(prefix: str, h: Hand, offset: np.ndarray, q: dict[str, float], env: bool = False) -> dict:
        server.scene.add_frame(prefix, wxyz=_wxyz(mount(h)), position=tuple(wrist_pos() if env else offset),
                               show_axes=False)
        frames = {}
        for link, T in h.link_poses(q).items():
            frames[link] = server.scene.add_frame(f"{prefix}/{link}", wxyz=_wxyz(T[:3, :3]),
                                                  position=tuple(T[:3, 3]), show_axes=False)
        for k, (link, m, O, rgb) in enumerate(h.visuals):
            server.scene.add_mesh_simple(f"{prefix}/{link}/v{k}", vertices=np.asarray(m.vertices, np.float32),
                                         faces=np.asarray(m.faces, np.uint32), color=rgb, wxyz=_wxyz(O[:3, :3]),
                                         position=tuple(O[:3, 3]))
        if not env:                          # the pen under the palm, for scale
            server.scene.add_box(f"{prefix}_pen", dimensions=PEN, color=(40, 90, 200),
                                 position=tuple(offset + np.array([0.0, 0.0, -0.08])))
        return frames

    def update_pose() -> None:
        h = hand(names.index(g_hand.value))
        poses = h.link_poses({n: float(s.value) for n, s in state["sliders"].items()})
        for link, fr in state["handles"].items():
            T = poses[link]
            fr.wxyz, fr.position = _wxyz(T[:3, :3]), tuple(T[:3, 3])

    def rebuild(_=None) -> None:
        server.scene.reset()
        if folder["f"] is not None:
            folder["f"].remove()
        state["sliders"], state["handles"] = {}, {}
        if g_all.value:
            cols = math.ceil(math.sqrt(len(paths)))
            for i in range(len(paths)):
                r, c = divmod(i, cols)
                h = hand(i)
                draw(f"/all/{i}", h, np.array([c * GRID_PITCH, -r * GRID_PITCH, 0.0]), {})
                server.scene.add_label(f"/all/{i}_label", h.name,
                                       position=(c * GRID_PITCH, -r * GRID_PITCH, 0.12))
            g_info.content = f"{len(paths)} hands, row by row in the list's order"
            folder["f"] = None
            return
        h = hand(names.index(g_hand.value))
        g_info.content = h.info
        folder["f"] = server.gui.add_folder("joints")
        with folder["f"]:
            b_zero, b_rand = server.gui.add_button("zero"), server.gui.add_button("random pose")
            for n, _j in h.joints:
                lo, hi = h.limits(n)
                s = server.gui.add_slider(n, lo, hi, (hi - lo) / 200.0, min(max(0.0, lo), hi))
                s.on_update(lambda _e: update_pose())
                state["sliders"][n] = s

        def set_all(fn) -> None:
            for n, s in state["sliders"].items():
                s.value = fn(*h.limits(n))
            update_pose()
        b_zero.on_click(lambda _e: set_all(lambda lo, hi: min(max(0.0, lo), hi)))
        b_rand.on_click(lambda _e: set_all(lambda lo, hi: float(np.random.uniform(lo, hi))))
        env = bool(g_env.value)
        if env:
            draw_env()
        state["handles"] = draw("/hand", h, np.zeros(3), {n: float(s.value) for n, s in state["sliders"].items()},
                                env=env)

    def step(d: int) -> None:
        g_hand.value = names[(names.index(g_hand.value) + d) % len(names)]
    g_prev.on_click(lambda _e: step(-1))
    g_next.on_click(lambda _e: step(+1))
    def move_wrist(_=None) -> None:
        if g_env.value and not g_all.value:
            server.scene.add_frame("/hand", wxyz=_wxyz(mount(hand(names.index(g_hand.value)))),
                                   position=tuple(wrist_pos()), show_axes=False)

    def turn_pen(_=None) -> None:
        if g_env.value and not g_all.value:
            draw_env()

    def wrist_default(_=None) -> None:
        for k, sl in g_wrist.items():
            sl.value = WRIST_DEFAULT[k]
    for sl in g_wrist.values():
        sl.on_update(move_wrist)
    g_wrist_reset.on_click(wrist_default)
    g_yaw.on_update(turn_pen)
    for g in (g_hand, g_env, g_all):
        g.on_update(rebuild)
    rebuild()
    print(f"viser at http://{'localhost' if args.host == '0.0.0.0' else args.host}:{args.port}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
