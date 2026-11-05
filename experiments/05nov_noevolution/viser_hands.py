"""Drawing grammar hands in viser, shared by viser_children.py and viser_beam.py: each hand exactly as built (its URDF,
its root -- the mount / wrist frame -- at the given point), links coloured by their joint's kind (flexion blue,
abduction red, roll green, roll tips striped, palm grey), a black bar across the finger at every joint (the links
are the stretches between bars), each link its own mesh (hover highlights it, click reports it), and optional finger
labels f0, f1, ... at the bases.

One global view convention: the palm's grasp side (+x) toward world +z (the camera), plate +y -> world +y,
plate +z -> world -x (a rotation: no hand is mirrored).
"""
from __future__ import annotations

import json
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
KIND = {(0, 0): "flex", (90, 0): "abd", (0, 90): "roll"}
PALM_UP = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def design_of(spec_rel: str) -> dict:
    s = json.loads((REPO / spec_rel).read_text())
    return s["design"] if isinstance(s["design"], dict) else json.loads(s["design"])


def describe_design(d: dict) -> list[str]:
    lines = [f"{len(d['fingers'])} fingers, {sum(len(f['joints']) for f in d['fingers'])} motors"]
    for i, f in enumerate(d["fingers"]):
        lines.append(f"- f{i}: base ({f['y']}, {f['z']}) facing {f['facing']} tilt {f['tilt']}: "
                     + ", ".join(f"{KIND.get(tuple(j['axis']), j['axis'])} {j['length']}" for j in f["joints"]))
    return lines


class HandDrawer:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="viser_hands_"))
        self._load = lru_cache(maxsize=1024)(self._load_uncached)

    def _load_uncached(self, spec_rel: str):
        import yourdfpy
        from hand_sampler.grammar.visual_colours import colour_urdf_text
        spec = json.loads((REPO / spec_rel).read_text())
        design = spec["design"] if isinstance(spec["design"], dict) else json.loads(spec["design"])
        urdf = REPO / spec["urdf"]
        out = self.tmp / urdf.parent.name / urdf.name
        out.parent.mkdir(parents=True, exist_ok=True)
        for f in urdf.parent.iterdir():
            if f.suffix in (".obj", ".stl") and not (out.parent / f.name).exists():
                (out.parent / f.name).symlink_to(f)
        out.write_text(colour_urdf_text(urdf.read_text(), design))
        return yourdfpy.URDF.load(str(out), load_meshes=True, build_collision_scene_graph=False), design

    def geometry(self, spec_rel: str, curl: float):
        """(parts [(V, F, rgb)], edge segments (N, 2, 3), finger bases [(3,)]), in the view frame, root at 0."""
        import trimesh
        u, design = self._load(spec_rel)
        q = []
        for name in u.actuated_joint_names:
            j = u.joint_map[name]
            m = re.match(r"f(\d+)_j(\d+)$", name)
            flex = m is not None and tuple(design["fingers"][int(m[1])]["joints"][int(m[2])]["axis"]) == (0, 0)
            q.append(curl * j.limit.upper if flex and j.limit is not None else 0.0)
        u.update_cfg(np.array(q))
        # One mesh per (link, colour): hovering highlights that link alone, clicking reports it. A link is a finger
        # link f<i>_link<k> (its visual, and a roll tip's stripes) or the palm (plate, mount, base motors).
        groups: dict[tuple, list] = {}
        g = u.scene.graph
        for node in g.nodes_geometry:
            T, geom = g[node]
            mesh = u.scene.geometry.get(geom)
            if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
                continue
            mm = re.match(r"(f\d+_link\d+)", str(geom))
            link = mm[1] if mm else "palm"
            v = trimesh.transform_points(np.asarray(mesh.vertices), T) @ PALM_UP.T
            rgb = tuple(int(c) for c in np.asarray(mesh.visual.main_color[:3]))
            groups.setdefault((link, rgb), []).append((v, np.asarray(mesh.faces)))
        parts = []
        for (link, rgb), vfs in groups.items():
            V, F, n = [], [], 0
            for v, f in vfs:
                V.append(v)
                F.append(f + n)
                n += len(v)
            parts.append((np.concatenate(V), np.concatenate(F), rgb, link))
        # A black bar across the finger at every joint (perpendicular to its link in the view, just above the
        # hand): the links are the stretches between bars. (Link outlines overlapped: a link's box also covers half
        # of each neighbouring motor housing.)
        top = max(float(V[:, 2].max()) for V, _, _, _ in parts) + 0.003
        segs, bases = [], []
        for i, f in enumerate(design["fingers"]):
            for k in range(len(f["joints"])):
                try:
                    T = u.get_transform(frame_to=f"f{i}_link{k}")
                except Exception:
                    continue
                p, d = PALM_UP @ T[:3, 3], PALM_UP @ T[:3, 0]
                if k == 0:
                    bases.append(p)
                n2 = np.array([-d[1], d[0]])
                n2 = n2 / np.linalg.norm(n2) if np.linalg.norm(n2) > 1e-6 else np.array([1.0, 0.0])
                a3 = np.array([*(p[:2] + 0.022 * n2), top])
                b3 = np.array([*(p[:2] - 0.022 * n2), top])
                segs.append([a3, b3])
        edges = np.asarray(segs) if segs else np.zeros((0, 2, 3))
        return parts, edges, bases

    def draw(self, server, name: str, spec_rel: str, off, curl: float = 0.0, outline: bool = True,
             finger_labels: bool = True, on_click=None):
        """Add the hand under scene node `name` with its root at off; returns (handles, (lo, hi) bounding box).
        on_click(link) is called with the clicked link's name ("palm" or "f<i>_link<k>")."""
        off = np.asarray(off, dtype=float)
        parts, edges, bases = self.geometry(spec_rel, curl)
        hs = []
        for n, (V, F, rgb, link) in enumerate(parts):
            m = server.scene.add_mesh_simple(f"{name}/{link}_{n}", vertices=(V + off).astype(np.float32),
                                             faces=F.astype(np.uint32), color=rgb, flat_shading=False)
            if on_click is not None:
                m.on_click(lambda _e, link=link: on_click(link))
            hs.append(m)
        if outline and len(edges):
            hs.append(server.scene.add_line_segments(f"{name}/joints", points=(edges + off).astype(np.float32),
                                                     colors=(0, 0, 0), line_width=3.0))
        if finger_labels:
            for i, b in enumerate(bases):
                hs.append(server.scene.add_label(f"{name}/f{i}", f"f{i}", position=tuple(off + b + [0, 0, 0.03])))
        allv = np.concatenate([V for V, _, _, _ in parts])
        return hs, (allv.min(0) + off, allv.max(0) + off)


def describe_link(design: dict, link: str) -> str:
    """'f1_link2' -> 'finger 1, joint 2: flexion, link 40 mm (joint 3 of 5)'; 'palm' -> the palm."""
    m = re.match(r"f(\d+)_link(\d+)$", link)
    if not m:
        return "**palm** (plate, mount and the fingers' base motors)"
    i, k = int(m[1]), int(m[2])
    f = design["fingers"][i]
    j = f["joints"][k]
    kind = {"flex": "flexion", "abd": "abduction", "roll": "roll"}.get(KIND.get(tuple(j["axis"]), ""), str(j["axis"]))
    return (f"**finger {i}, joint {k}**: {kind}, link {j['length']} mm (joint {k + 1} of {len(f['joints'])}; "
            f"finger base ({f['y']}, {f['z']}), facing {f['facing']})")


def aim(client, lo, hi) -> None:
    """Look straight down (a slight tilt keeps the up vector defined) at the middle of the box lo..hi, far enough
    that its width and height both fit the client's view."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):      # nothing drawn yet: leave the camera alone
        return
    c = (lo + hi) / 2
    w, h = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    fov = float(getattr(client.camera, "fov", 1.3))
    aspect = float(getattr(client.camera, "aspect", 1.6) or 1.6)
    t = np.tan(fov / 2)
    d = 1.08 * max(h / 2 / t, w / 2 / (t * aspect)) + float(hi[2])
    client.camera.up_direction = (0.0, 0.0, 1.0)
    client.camera.look_at = (float(c[0]), float(c[1]), 0.0)
    client.camera.position = (float(c[0]), float(c[1]) - 0.02 * d, d)
    print(f"[aim] client {client.client_id}: box {np.round(lo, 2).tolist()} .. {np.round(hi, 2).tolist()}, fov {fov:.3f}, "
          f"aspect {aspect:.2f} -> look_at {np.round(c, 2).tolist()}, height {d:.2f} m", flush=True)
