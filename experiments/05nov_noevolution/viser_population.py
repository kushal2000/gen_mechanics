"""Browse a hand population in viser, every link coloured by its joint's kind (flexion blue, abduction red, roll
green; roll tips striped; palm grey).

    .venv_isaacsim/bin/python experiments/05nov_noevolution/viser_population.py \
        [--manifest debug_outputs/noevolution/seed_4096/manifest.json] [--port 8085]

    then, from your laptop: ssh -L 8085:<this node>:8085 <login host>  ->  http://localhost:8085

Kit-free (yourdfpy + trimesh + viser). Two modes: "one hand" (a niche, then a slider over its hands, with the hand's
design spelled out) or "whole niche" (all of a niche's hands on a grid, labelled). A "curl" slider bends every
flexion joint to that fraction of its upper limit, so the joint kinds show in motion; the palm normal points up.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
KIND = {(0, 0): "flex", (90, 0): "abd", (0, 90): "roll"}
PALM_UP = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])   # palm normal (+x, grasp side) -> world +z (toward the camera); plate +y -> world +y, +z -> world -x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="debug_outputs/noevolution/seed_4096/manifest.json")
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--scores-from", default=None,
                    help="a study_beam.py state.json: label each hand with its mean score there (matched by design)")
    ap.add_argument("--cols", type=int, default=12, help="hands per row in the 'all groups' view")
    args = ap.parse_args()
    import trimesh
    import viser
    import yourdfpy
    from hand_sampler.grammar.visual_colours import colour_urdf_text

    man = json.loads((REPO / args.manifest).read_text())
    by_niche: dict[str, list[dict]] = {}
    for h in man["hands"]:
        by_niche.setdefault(h.get("niche", "all"), []).append(h)
    niches = list(by_niche)
    score_of: dict[str, float] = {}
    if args.scores_from:                                    # matched by canonical design key
        from statistics import mean
        from coevolution.evolution.variation import design_key
        from hand_sampler.grammar import hand as gh
        rules = getattr(gh, man.get("rules", "FEETECH_COARSE_NOCUT_RULES"))
        st_ = json.loads((REPO / args.scores_from).read_text())
        by_key = {r["key"]: mean(r["scores"]) for r in st_["designs"].values() if r.get("scores")}
        for h in man["hands"]:
            sp = json.loads((REPO / h["spec"]).read_text())
            d = sp["design"] if isinstance(sp["design"], dict) else json.loads(sp["design"])
            k = design_key(gh.hand_from_dict(d), rules)
            if k in by_key:
                score_of[h["id"]] = by_key[k]
        for hs in by_niche.values():
            hs.sort(key=lambda h: -score_of.get(h["id"], -1.0))
        print(f"[viser] scores for {len(score_of)}/{len(man['hands'])} hands from {args.scores_from}", flush=True)
    parent_score = next((score_of.get(h["id"]) for h in man["hands"] if h.get("niche") == "parent"), None)
    tmp = Path(tempfile.mkdtemp(prefix="viser_population_"))

    @lru_cache(maxsize=512)
    def load(spec_rel: str):
        spec = json.loads((REPO / spec_rel).read_text())
        design = spec["design"] if isinstance(spec["design"], dict) else json.loads(spec["design"])
        urdf = REPO / spec["urdf"]
        coloured = tmp / urdf.parent.name / urdf.name
        coloured.parent.mkdir(parents=True, exist_ok=True)
        for f in urdf.parent.iterdir():                     # the palm meshes next to it
            if f.suffix in (".obj", ".stl") and not (coloured.parent / f.name).exists():
                (coloured.parent / f.name).symlink_to(f)
        coloured.write_text(colour_urdf_text(urdf.read_text(), design))
        u = yourdfpy.URDF.load(str(coloured), load_meshes=True, build_collision_scene_graph=False)
        return u, design

    def meshes(spec_rel: str, curl: float):
        """(vertices, faces, rgb) per colour, the hand at the curl pose, palm up, metres."""
        u, design = load(spec_rel)
        q = []
        for name in u.actuated_joint_names:
            j = u.joint_map[name]
            i, k = (int(v) for v in name[1:].split("_j")) if name.startswith("f") and "_j" in name else (-1, -1)
            flex = i >= 0 and tuple(design["fingers"][i]["joints"][k]["axis"]) == (0, 0)
            q.append(curl * j.limit.upper if flex and j.limit is not None else 0.0)
        u.update_cfg(np.array(q))
        groups: dict[tuple, list] = {}
        for m in u.scene.dump(concatenate=False):
            if not isinstance(m, trimesh.Trimesh) or not len(m.faces):
                continue
            rgb = tuple(int(c) for c in np.asarray(m.visual.main_color[:3]))
            groups.setdefault(rgb, []).append(m)
        out = []
        for rgb, ms in groups.items():
            m = trimesh.util.concatenate(ms)
            out.append((np.asarray(m.vertices) @ PALM_UP.T, np.asarray(m.faces), rgb))
        return out

    def describe(h: dict) -> str:
        _, d = load(h["spec"])
        lines = [f"**{h['id']}**: {len(d['fingers'])} fingers, {sum(len(f['joints']) for f in d['fingers'])} motors"]
        if h.get("edit"):                                   # a children manifest (children_manifest.py)
            lines.append(f"edit: **{h['edit']}** · niche {h.get('cell', '?')}")
        if h["id"] in score_of:
            lines.append(f"score: **{score_of[h['id']]:.2f}** goals / episode (100 envs)"
                         + (f", **{score_of[h['id']] - parent_score:+.2f}** vs the parent"
                            if parent_score is not None and h.get("niche") != "parent" else ""))
        for i, f in enumerate(d["fingers"]):
            lines.append(f"- finger {i}: base ({f['y']}, {f['z']}) facing {f['facing']} tilt {f['tilt']}: "
                         + ", ".join(f"{KIND.get(tuple(j['axis']), j['axis'])} {j['length']}" for j in f["joints"]))
        return "\n".join(lines)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    with server.gui.add_folder("Population"):
        g_mode = server.gui.add_dropdown("view", ("one hand", "whole niche", "all groups"),
                                         initial_value="all groups" if len(man["hands"]) <= 400 else "one hand")
        g_niche = server.gui.add_dropdown("niche", niches, initial_value=niches[0])
        g_idx = server.gui.add_slider("hand", min=0, max=len(by_niche[niches[0]]) - 1, step=1, initial_value=0)
        g_curl = server.gui.add_slider("curl (flexion)", min=0.0, max=1.0, step=0.05, initial_value=0.0)
        g_info = server.gui.add_markdown("")
    server.gui.add_markdown("flexion **blue**, abduction **red**, roll **green** (roll tips striped), palm grey")
    handles: list = []

    def redraw(*_):
        for hd in handles:
            hd.remove()
        handles.clear()
        hands = by_niche[g_niche.value]
        g_idx.max = len(hands) - 1
        g_idx.value = min(g_idx.value, g_idx.max)
        if g_mode.value == "one hand":
            todo = [(hands[int(g_idx.value)], np.zeros(3))]
            g_info.content = describe(hands[int(g_idx.value)])
        elif g_mode.value == "whole niche":
            cols = int(np.ceil(np.sqrt(len(hands))))
            todo = [(h, np.array([(k % cols) * 0.42, -(k // cols) * 0.42, 0.0])) for k, h in enumerate(hands)]
            g_info.content = f"**{g_niche.value}**: {len(hands)} hands, {cols} per row (0.42 m apart)"
        else:                                               # every group, one row (wrapped) per group
            todo, row = [], 0
            for n in niches:
                hs = by_niche[n]
                handles.append(server.scene.add_label(f"/groups/{n}", f"{n} ({len(hs)})",
                                                      position=(-0.45, -row * 0.42, 0.0)))
                for k, h in enumerate(hs):
                    todo.append((h, np.array([(k % args.cols) * 0.42, -(row + k // args.cols) * 0.42, 0.0])))
                row += (len(hs) - 1) // args.cols + 1
            g_info.content = (f"**all {len(todo)} hands**, one row per group (rows wrap at {args.cols}); "
                              + ("best first; " if score_of else "") + "click a hand for its design")
        for k, (h, off) in enumerate(todo):
            for n, (V, F, rgb) in enumerate(meshes(h["spec"], float(g_curl.value))):
                m = server.scene.add_mesh_simple(f"/hands/{k}/{n}", vertices=(V + off).astype(np.float32),
                                                 faces=F.astype(np.uint32), color=rgb, flat_shading=False)
                if g_mode.value != "one hand":
                    m.on_click(lambda _e, h=h: setattr(g_info, "content", describe(h)))
                handles.append(m)
            if g_mode.value != "one hand":
                tag = h["id"].rsplit("_", 1)[1] if g_mode.value == "whole niche" else h["id"]
                if h["id"] in score_of:
                    tag += f" · {score_of[h['id']]:.2f}"
                    if parent_score is not None and h.get("niche") != "parent":
                        tag += f" ({score_of[h['id']] - parent_score:+.2f})"
                handles.append(server.scene.add_label(f"/hands/{k}/label", tag,
                                                      position=tuple(off + np.array([0.0, -0.17, 0.0]))))

    for g in (g_mode, g_niche, g_idx, g_curl):
        g.on_update(redraw)
    redraw()
    print(f"[viser] {len(man['hands'])} hands in {len(niches)} niches: http://localhost:{args.port} "
          f"(forward the port from this node)", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
