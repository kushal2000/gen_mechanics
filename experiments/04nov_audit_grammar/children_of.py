"""Every child one mutation away from a design, grouped by level, described in words, and drawn.

    .venv_isaacsim/bin/python experiments/04nov_audit_grammar/children_of.py \
        [--design debug_outputs/evolution/map_elites/hands/g0058_c00/g0058_c00.design.json] [--per-op 3] [--seed 0]

Kit-free. Rules: the coarse Feetech rules without palm cuts (the next experiment's grammar). Children: every valid
application of every operator (`operators.iter_valid`, exhaustive), one per distinct design (canonical key). Levels:
chain (length, axis_az, axis_el, split, merge), placement (move_finger, turn_finger, tilt_finger), topology
(add_finger -- a copy of one of the hand's fingers, remove_finger). Each child gets a plain-words description of its edit
(from the move, before re-centring and canonicalisation) and the proposed 32-niche cell (fingers {2,3,4,5-6} x mean
joints per finger <=2.5/<=3/<=3.5/>3.5 x non-flexion share <=0.29 / >0.29).
Writes, under debug_outputs/grammar_audit/children_<id>/: children.json (every child), parent.png, and
children.png (the parent, then --per-op randomly chosen children per operator, drawn as plot_niches.py draws hands).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/30oct_evolution_evaluation"))
LEVEL = {"length": "chain", "axis_az": "chain", "axis_el": "chain", "split": "chain", "merge": "chain",
         "move_finger": "placement", "turn_finger": "placement", "tilt_finger": "placement",
         "add_finger": "topology", "remove_finger": "topology"}
KIND = {(0, 0): "flex", (90, 0): "abd", (0, 90): "roll"}


def kind(ax) -> str:
    return KIND.get(tuple(ax), str(tuple(ax)))


def finger_str(f) -> str:
    return " ".join(f"{kind(j.axis)}{j.length}" for j in f.joints)


def describe(op: str, mv: tuple, parent) -> str:
    F = parent.fingers
    if op == "length":
        i, j, s = mv
        return f"finger {i} link {j} {'longer' if s > 0 else 'shorter'} (was {F[i].joints[j].length} mm)"
    if op in ("axis_az", "axis_el"):
        i, j, s = mv
        return f"finger {i} joint {j} kind: {kind(F[i].joints[j].axis)} -> " + ("abd/flex swap" if op == "axis_az" else "roll/flex swap")
    if op == "split":
        i, j, a, k = mv
        return f"finger {i}: new {k} joint {a} mm into link {j}"
    if op == "merge":
        i, j = mv
        return f"finger {i}: joint {j} ({kind(F[i].joints[j].axis)}) removed"
    if op == "move_finger":
        i, c, s = mv
        return f"finger {i} base {c} {'+' if s > 0 else '-'}40 mm"
    if op == "turn_finger":
        i, s = mv
        return f"finger {i} facing {'+' if s > 0 else '-'}30 deg"
    if op == "tilt_finger":
        i, s = mv
        return f"finger {i} tilt {'+' if s > 0 else '-'}30 deg"
    if op == "add_finger":
        y, z, i = mv
        return f"copy of finger {i} at ({y}, {z})"
    if op == "remove_finger":
        return f"finger {mv[0]} removed"
    return f"{op} {mv}"


def niche32(design: dict) -> str:
    fs = design["fingers"]
    n = len(fs)
    js = [j for f in fs for j in f["joints"]]
    per = len(js) / n
    jb = 0 if per <= 2.5 else 1 if per <= 3.0 else 2 if per <= 3.5 else 3
    nonflex = sum(tuple(j["axis"]) != (0, 0) for j in js) / len(js)
    return f"f{min(n, 5)}{'+' if n >= 5 else ''}_j{jb}_{'mixed' if nonflex > 0.29 else 'flex'}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="debug_outputs/evolution/map_elites/hands/g0058_c00/g0058_c00.design.json")
    ap.add_argument("--per-op", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from coevolution.evolution.variation import design_key
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import operators as gops
    from hand_sampler.grammar.urdf_export import build

    rules = replace(gh.FEETECH_COARSE_RULES, max_palm_joints=0)
    path = REPO / args.design
    pid = path.name.replace(".design.json", "")
    parent = gh.hand_from_dict(json.loads(path.read_text()))
    out = REPO / "debug_outputs/grammar_audit" / f"children_{pid}"
    out.mkdir(parents=True, exist_ok=True)

    seen, kids = {design_key(parent, rules)}, []
    for op in gops.OPERATORS:
        if op.name not in LEVEL:
            continue
        for mv, child in gops.iter_valid(parent, op, rules):
            k = design_key(child, rules)
            if k in seen:
                continue
            seen.add(k)
            d = gh.hand_to_dict(child)
            kids.append({"operator": op.name, "level": LEVEL[op.name], "move": list(mv),
                         "edit": describe(op.name, mv, parent), "niche": niche32(d), "design": d})
    pd = gh.hand_to_dict(parent)
    summary = {"parent": pid, "parent_niche": niche32(pd),
               "parent_fingers": [finger_str(f) for f in parent.fingers], "children": len(kids),
               "by_level": {}, "by_operator": {}, "leave_niche": sum(c["niche"] != niche32(pd) for c in kids)}
    for c in kids:
        summary["by_level"][c["level"]] = summary["by_level"].get(c["level"], 0) + 1
        summary["by_operator"][c["operator"]] = summary["by_operator"].get(c["operator"], 0) + 1
    (out / "children.json").write_text(json.dumps({"summary": summary, "children": kids}, indent=1))
    print(json.dumps(summary, indent=1))

    # drawing
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import matplotlib.patheffects as pe
    from matplotlib.patches import Patch
    from plot_niches import LIGHT, _wrist, project
    import yourdfpy
    wrist = _wrist()
    light = LIGHT / np.linalg.norm(LIGHT)
    KIND_RGB = {"flex": (0.16, 0.42, 0.85), "abd": (0.86, 0.20, 0.18), "roll": (0.15, 0.65, 0.30)}
    PALM_RGB = (0.62, 0.62, 0.64)

    def kind_faces(spec: Path, Rm: np.ndarray, hand):
        """Every visual triangle at q = 0 in the palm-down mount, each finger link coloured by the kind of the joint
        that moves it (joint j moves link j: flexion blue, abduction red, roll green); palm and base motors grey."""
        u = yourdfpy.URDF.load(str(spec.with_name(spec.name.replace(".spec.json", ".urdf"))), load_meshes=True,
                               build_collision_scene_graph=False)
        u.update_cfg(np.zeros(len(u.actuated_joint_names)))
        tris, cols = [], []
        for node in u.scene.graph.nodes_geometry:
            T, gname = u.scene.graph[node]
            m = u.scene.geometry[gname]
            if not len(getattr(m, "faces", [])):
                continue
            v = (np.c_[m.vertices, np.ones(len(m.vertices))] @ T.T)[:, :3]
            rgb = PALM_RGB
            if gname.startswith("f") and "_link" in gname:
                i, j = int(gname.split("_")[0][1:]), int(gname.split("_")[1][4:])
                rgb = KIND_RGB[kind(hand.fingers[i].joints[j].axis)]
            tris.append(v[m.faces] @ Rm.T)
            cols.append(np.repeat(np.array(rgb)[None], len(m.faces), 0))
        return np.concatenate(tris), np.concatenate(cols)

    def draw(ax, hand, hid, title, color="#0b0b0b", labels=False):
        with tempfile.TemporaryDirectory() as tmp:
            row = build(hand, hid, Path(tmp) / hid)
            spec = Path(row["spec"]) if Path(row["spec"]).is_absolute() else REPO / row["spec"]
            frame = np.asarray(json.loads(spec.read_text())["palm_frame"], float)
            Rm = wrist.palm_down_rotation(frame)
            tris, cols = kind_faces(spec, Rm, hand)
        xy, depth = project(tris)
        nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        order = np.argsort(-depth.mean(1))
        shade = (0.35 + 0.65 * np.abs(nrm @ light))[order, None]
        ax.add_collection(PolyCollection(xy[order], facecolors=cols[order] * shade, edgecolors="none"))
        if labels:                       # finger numbers at each base, in the drawing's frame
            for i, f in enumerate(hand.fingers):
                p = (Rm @ np.array([0.0, f.y, f.z]) * 1e-3)[None, None, :]
                q, _ = project(p)
                ax.text(q[0, 0, 0], q[0, 0, 1], str(i), fontsize=13, fontweight="bold", color="black",
                        ha="center", va="center", path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax.set_aspect("equal"); ax.autoscale_view(); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=8, color=color, loc="left")

    fig, ax = plt.subplots(figsize=(6, 6))
    draw(ax, parent, pid, f"{pid}  ({summary['parent_niche']})\n" + "\n".join(
        f"finger {i}: {s}" for i, s in enumerate(summary["parent_fingers"])), labels=True)
    legend = [Patch(color=KIND_RGB["flex"], label="flexion"), Patch(color=KIND_RGB["abd"], label="abduction"),
              Patch(color=KIND_RGB["roll"], label="roll")]
    ax.legend(handles=legend, loc="lower right", fontsize=9, frameon=False)
    fig.savefig(out / "parent.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rnd = random.Random(args.seed)
    pick = []
    for op in [o for o in LEVEL if summary["by_operator"].get(o)]:
        cs = [c for c in kids if c["operator"] == op]
        pick += rnd.sample(cs, min(args.per_op, len(cs)))
    cols_n = args.per_op + 1
    ops_shown = list(dict.fromkeys(c["operator"] for c in pick))
    fig, axes = plt.subplots(len(ops_shown), cols_n, figsize=(3.1 * cols_n, 3.1 * len(ops_shown)), squeeze=False)
    colour = {"chain": "#1b6faf", "placement": "#1baf7a", "topology": "#eb6834"}
    for r, op in enumerate(ops_shown):
        draw(axes[r, 0], parent, pid, f"parent ({summary['parent_niche']})", labels=True)
        cs = [c for c in pick if c["operator"] == op]
        for k in range(1, cols_n):
            ax = axes[r, k]
            if k - 1 >= len(cs):
                ax.axis("off")
                continue
            c = cs[k - 1]
            stay = "" if c["niche"] == summary["parent_niche"] else f"  -> {c['niche']}"
            # drawn as the edit leaves it (re-centred, not canonicalised), so it lines up with the parent and its
            # finger numbers match the description; the stored design is the canonical copy (same hand, turned)
            shown = gops.recentre(gops.OPERATOR_BY_NAME[op].apply(parent, tuple(c["move"]), rules), rules)
            draw(ax, shown, f"{pid}_child", textwrap.fill(f"[{c['level']}] {c['edit']}{stay}", 38),
                 color=colour[c["level"]], labels=True)
    fig.suptitle(f"{pid}: {summary['children']} distinct children one mutation away (no palm cuts); "
                 f"{args.per_op} drawn per operator", fontsize=11, x=0.01, ha="left")
    fig.legend(handles=legend, loc="upper right", ncol=3, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out / "children.png", dpi=110, facecolor="white")
    print(out / "children.png")


if __name__ == "__main__":
    main()
