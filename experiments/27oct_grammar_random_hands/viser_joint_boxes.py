"""The joint-token link boxes, as built today and with each of three proposed fixes. CPU only.

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/viser_joint_boxes.py [hands ...] [--port 8086]

Today's box (hand_sampler/design_space.py ``_ordered_box``, from the link's collision vertices in its own frame,
``_collision_vertices``) has three problems; each fix can be switched on separately:

  fix 1  along "distal" (joint -> the link's centroid) the box runs from the joint (0) to the span: geometry behind the
         joint is cut off and as much empty space added past the end. Fixed: from the geometry's min to its max.
  fix 2  along "bend" and along the joint axis the box is centred on the joint: wrong wherever the link is not, worst
         for a twist joint (its axis runs along the link). Fixed: min to max along both.
  fix 3  a URDF cylinder counts as its own bounding box, but the importer makes it a capsule
         (replace_cylinders_with_capsules): the caps are missing. Fixed: the capsule's two end points added.

Red: today's box. Green: the box with the fixes ticked. White edge: the box's "distal" edge; blue edge: its "axis" edge
(the joint axis, signed: the token's only source for it). Dots: where each box puts the joint (corner + half the bend and
axis edges -- exact for today's box, which is centred on the joint), against the true joint (white). The hand is drawn as the
simulator collides with it (cylinders as capsules). The table gives, per joint, how much of the link's real geometry
(capsules included) lies outside each box and the box's IoU with the geometry's own extent.
"""
from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from hand_sampler import design_space as ds  # noqa: E402
from viser_hands import Hand, _wxyz, find_urdfs  # noqa: E402

G = "assets/urdf/27oct_grammar_random_hands"
DEFAULT_HANDS = [f"{G}/coarse_s0_000", f"{G}/coarse_s0_072", f"{G}/coarse_s0_025", f"{G}/coarse_s0_039",
                 "assets/urdf/27oct_grammar_claude_hands/claude_design_0",
                 "assets/urdf/generated/14oct_design_search/prim4x4_r10l25_g090_090_090_090.spec.json",
                 "assets/urdf/unified_dynamics_commercial_hands/allegro/allegro_left.spec.json",
                 "assets/urdf/unified_dynamics_commercial_hands/wuji2/wuji2_left.spec.json"]
VARIANTS = {"today": (False, False, False), "fix 1": (True, False, False), "fix 2": (False, True, False),
            "fix 3": (False, False, True), "all 3": (True, True, True)}
EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def link_vertices(link: ET.Element, base: Path, caps: bool) -> np.ndarray:
    """The link's collision vertices in its frame (_collision_vertices); ``caps``: each cylinder's capsule ends too."""
    pieces = []
    for c in link.findall("collision"):
        pieces.append(ds._collision_vertices(c, base))
        cyl = c.find("geometry/cylinder")
        if caps and cyl is not None:
            r, L, O = float(cyl.get("radius")), float(cyl.get("length")), ds._origin(c.find("origin"))
            pieces.append(np.array([[0, 0, L / 2 + r], [0, 0, -L / 2 - r]]) @ O[:3, :3].T + O[:3, 3])
    pieces = [p for p in pieces if p.size]
    return np.concatenate(pieces) if pieces else np.zeros((0, 3))


def box(V: np.ndarray, axis: np.ndarray, fix1: bool, fix2: bool) -> np.ndarray | None:
    """_ordered_box's four points (corner, then the ends of its distal / bend / axis edges), with fixes 1 and 2."""
    if not len(V):
        return None
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    distal = V.mean(axis=0)
    distal = distal - axis * float(distal @ axis)
    if np.linalg.norm(distal) < 1e-8:
        transverse = V - np.outer(V @ axis, axis)
        distal = transverse[np.argmax(np.linalg.norm(transverse, axis=1))]
    if np.linalg.norm(distal) < 1e-8:
        distal = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    distal /= np.linalg.norm(distal)
    bend = np.cross(axis, distal)
    bend /= max(float(np.linalg.norm(bend)), 1e-12)
    P = np.stack([V @ distal, V @ bend, V @ axis], axis=-1)
    lo, hi = P.min(0), P.max(0)
    span = np.maximum(hi - lo, 1e-6)
    if fix1:
        s0, length = lo[0], span[0]
    else:                                   # today: from the joint
        s0, length = 0.0, max(float(span[0]), float(hi[0]), 1e-6)
    s1, s2 = (lo[1], lo[2]) if fix2 else (-0.5 * span[1], -0.5 * span[2])     # today: centred on the joint
    p0 = s0 * distal + s1 * bend + s2 * axis
    return np.array([p0, p0 + length * distal, p0 + span[1] * bend, p0 + span[2] * axis])


def implied_origin(B: np.ndarray) -> np.ndarray:
    """Where the box says the joint is: its corner plus half its bend and axis edges (today's box is built that way)."""
    return B[0] + 0.5 * ((B[2] - B[0]) + (B[3] - B[0]))


def corners(B: np.ndarray) -> np.ndarray:
    p0, e = B[0], B[1:] - B[0]
    return np.array([p0 + a * e[0] + b * e[1] + c * e[2] for a in (0, 1) for b in (0, 1) for c in (0, 1)])


def score(B: np.ndarray | None, V: np.ndarray) -> tuple[float, float]:
    """(mm of V outside B, IoU of B with V's extent in B's axes)."""
    if B is None or not len(V):
        return float("nan"), float("nan")
    p0, e = B[0], B[1:] - B[0]
    n = np.linalg.norm(e, axis=1)
    loc = (V - p0) @ (e / n[:, None]).T
    miss = float((np.maximum(0, -loc) + np.maximum(0, loc - n)).max())
    lo, hi = loc.min(0), loc.max(0)
    inter = float(np.prod(np.clip(np.minimum(hi, n) - np.maximum(lo, 0), 0, None)))
    return 1e3 * miss, inter / (float(np.prod(n)) + float(np.prod(hi - lo)) - inter)


class Boxes:
    """Every hand joint's link box under each variant, for one URDF (joints from its spec, else every movable one)."""

    def __init__(self, hand: Hand):
        root = ET.parse(hand.path).getroot()
        links = {l.get("name"): l for l in root.findall("link")}
        joints = {j.get("name"): j for j in root.findall("joint")}
        names = hand.spec.get("hand_joint_names") or [n for n, _ in hand.joints]
        caps_sim = bool(hand.spec.get("replace_cylinders_with_capsules", False))
        self.rows = []
        for jn in names:
            j = joints[jn]
            child = j.find("child").get("link")
            ax = j.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
            truth = link_vertices(links[child], hand.path.parent, caps_sim)     # what PhysX collides with
            V = {c: link_vertices(links[child], hand.path.parent, c and caps_sim) for c in (False, True)}
            combo = {(f1, f2, f3): box(V[f3], axis, f1, f2)
                     for f1 in (False, True) for f2 in (False, True) for f3 in (False, True)}
            boxes = {name: combo[v] for name, v in VARIANTS.items()}
            self.rows.append({"joint": jn, "link": child, "truth": truth, "boxes": boxes, "combo": combo,
                              "scores": {k: score(b, truth) for k, b in boxes.items()}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=DEFAULT_HANDS)
    ap.add_argument("--port", type=int, default=8086)
    args = ap.parse_args()
    inputs = [p for p in args.inputs if Path(REPO / p).exists() or Path(p).exists() or any(ch in p for ch in "*?")]
    paths = find_urdfs(inputs)
    import viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.set_up_direction("+z")
    names = [p.stem for p in paths]
    cache: dict[int, tuple[Hand, Boxes]] = {}

    def get(i: int):
        if i not in cache:
            h = Hand(paths[i])
            cache[i] = (h, Boxes(h))
        return cache[i]

    g_hand = server.gui.add_dropdown("hand", names, initial_value=names[0])
    g_f = [server.gui.add_checkbox(f"fix {k}", True) for k in (1, 2, 3)]
    g_today = server.gui.add_checkbox("show today's box (red)", True)
    g_all = server.gui.add_checkbox("every joint (else the one picked)", True)
    g_joint = server.gui.add_dropdown("joint", ("-",), initial_value="-")
    g_pose = server.gui.add_button("random pose")
    g_zero = server.gui.add_button("zero pose")
    md_sum = server.gui.add_markdown("")
    md_joint = server.gui.add_markdown("")
    state = {"q": {}, "frames": {}}

    def variant_name() -> str:
        f = tuple(bool(g.value) for g in g_f)
        for k, v in VARIANTS.items():
            if v == f:
                return k
        return "custom"

    def chosen_box(row) -> np.ndarray | None:
        return row["combo"][tuple(bool(g.value) for g in g_f)]

    def draw_boxes() -> None:
        h, bx = get(names.index(g_hand.value))
        sel = g_joint.value
        for row in bx.rows:
            show = g_all.value or row["joint"] == sel
            wide = row["joint"] == sel
            for tag, B, col, on in (("today", row["boxes"]["today"], (230, 50, 50), g_today.value),
                                    ("fixed", chosen_box(row), (40, 210, 80), True)):
                name = f"/hand/{row['link']}/box_{tag}"
                if B is None:
                    continue
                C = corners(B)
                segs = np.array([[C[a], C[b]] for a, b in EDGES], np.float32)
                server.scene.add_line_segments(name, points=segs, colors=np.array(col, np.uint8),
                                               line_width=4.0 if wide else 2.0, visible=bool(show and on))
                server.scene.add_line_segments(name + "_distal", points=np.array([[B[0], B[1]]], np.float32),
                                               colors=np.array((255, 255, 255), np.uint8),
                                               line_width=5.0 if wide else 2.5, visible=bool(show and on))
                server.scene.add_line_segments(name + "_axis", points=np.array([[B[0], B[3]]], np.float32),
                                               colors=np.array((70, 140, 255), np.uint8),
                                               line_width=5.0 if wide else 2.5, visible=bool(show and on))
                server.scene.add_point_cloud(name + "_origin", points=implied_origin(B)[None].astype(np.float32),
                                             colors=col, point_size=0.006, visible=bool(show and on))
            server.scene.add_point_cloud(f"/hand/{row['link']}/joint", points=np.zeros((1, 3), np.float32),
                                         colors=(255, 255, 255), point_size=0.007, visible=bool(show))
        r = next((x for x in bx.rows if x["joint"] == sel), None)
        if r is None:
            md_joint.content = ""
            return
        sc = {k: (*v, r["boxes"][k]) for k, v in r["scores"].items()}
        sc["ticked: " + variant_name()] = (*score(chosen_box(r), r["truth"]), chosen_box(r))
        table = "\n".join(f"| {k} | {v[0]:.1f} | {v[1]:.2f} | {1e3 * np.linalg.norm(implied_origin(v[2])):.1f} |"
                          for k, v in sc.items() if v[2] is not None)
        md_joint.content = (f"**{sel}** (link `{r['link']}`)\n\n| box | outside (mm) | IoU | joint from box: error (mm) |"
                            f"\n|---|---|---|---|\n{table}")

    def summarise() -> None:
        h, bx = get(names.index(g_hand.value))
        lines = []
        for k in VARIANTS:
            s = np.array([r["scores"][k] for r in bx.rows], float)
            lines.append(f"| {k} | {np.nanmedian(s[:, 0]):.1f} / {np.nanmax(s[:, 0]):.1f} | "
                         f"{np.nanmedian(s[:, 1]):.2f} / {np.nanmin(s[:, 1]):.2f} |")
        md_sum.content = (f"**{h.name}**: {len(bx.rows)} joints\n\n| box | outside mm (median / max) | "
                          f"IoU (median / min) |\n|---|---|---|\n" + "\n".join(lines))

    def place() -> None:
        h, _ = get(names.index(g_hand.value))
        for link, T in h.link_poses(state["q"]).items():
            fr = state["frames"].get(link)
            if fr is not None:
                fr.wxyz, fr.position = _wxyz(T[:3, :3]), tuple(T[:3, 3])

    def rebuild(_=None) -> None:
        server.scene.reset()
        h, bx = get(names.index(g_hand.value))
        state["q"] = {}
        server.scene.add_frame("/hand", wxyz=_wxyz(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)),
                               show_axes=False)                       # palm frame +x (grasp normal) down
        state["frames"] = {l: server.scene.add_frame(f"/hand/{l}", show_axes=False) for l in h.link_poses({})}
        for k, (link, m, O, rgb) in enumerate(h.visuals):
            server.scene.add_mesh_simple(f"/hand/{link}/v{k}", vertices=np.asarray(m.vertices, np.float32),
                                         faces=np.asarray(m.faces, np.uint32), color=rgb, opacity=0.35,
                                         wxyz=_wxyz(O[:3, :3]), position=tuple(O[:3, 3]))
        place()
        worst = sorted(bx.rows, key=lambda r: -np.nan_to_num(r["scores"]["today"][0]))
        g_joint.options = tuple(r["joint"] for r in worst)
        g_joint.value = worst[0]["joint"]
        summarise()
        draw_boxes()

    def set_pose(random: bool) -> None:
        h, _ = get(names.index(g_hand.value))
        state["q"] = {n: (float(np.random.uniform(*h.limits(n))) if random else 0.0) for n, _ in h.joints}
        place()

    g_hand.on_update(rebuild)
    for g in (*g_f, g_today, g_all, g_joint):
        g.on_update(lambda _e: draw_boxes())
    g_pose.on_click(lambda _e: set_pose(True))
    g_zero.on_click(lambda _e: set_pose(False))
    rebuild()
    print(f"[joint boxes] http://localhost:{args.port}  ({len(paths)} hands)", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
