"""One design and its children by one mutation, side by side, in viser.

    .venv_isaacsim/bin/python experiments/05nov_noevolution/viser_children.py \
        --manifest debug_outputs/noevolution/children/<design>/manifest.json [--scores-from <study_beam state.json>] \
        [--port 8089]

The manifest is children_manifest.py's (the parent, then every one-step child as the grammar deploys it, each with
its operator and edit). Pick a mutation: the parent stands on the left (PARENT), that mutation's children on a grid
to its right, best first when scores are given (a study_beam.py state.json, matched by design), each labelled with its
rank, score, change against the parent and its edit. Every hand is drawn exactly as built (viser_hands.py: joint-kind
colours, black link outlines, finger labels); click one for its full design.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viser_hands import KIND, HandDrawer, aim, describe_design, describe_link, design_of  # noqa: E402

GAP = 0.55           # m between hands' roots
NAMES = {"split": "add a joint (split)", "merge": "remove a joint (merge)", "length": "link length +-20 mm",
         "kind": "change a joint's kind", "axis_az": "swap flexion <-> abduction", "axis_el": "swap flexion <-> roll", "move_finger": "move a finger",
         "turn_finger": "turn a finger", "tilt_finger": "tilt a finger", "add_finger": "copy a finger",
         "remove_finger": "remove a finger"}


def short(edit: str, child: dict | None = None) -> str:
    """A label-sized edit; a kind swap names the new kind (read from the child design)."""
    m = re.match(r"finger (\d+) joint (\d+) kind: (\w+) ->", edit or "")
    if m and child is not None:
        try:
            new = KIND.get(tuple(child["fingers"][int(m[1])]["joints"][int(m[2])]["axis"]), "?")
            return f"f{m[1]} j{m[2]}: {m[3]} -> {new}"
        except (IndexError, KeyError):
            pass
    e = re.sub(r"finger (\d+)", r"f\1", edit or "")
    return re.sub(r"new (\w+) joint (\d+) mm into link (\d+)", r"+\1 at \2 mm of link \3", e).strip()


def scores_by_key(state_file: str, specs: list[str]) -> dict[str, float]:
    """spec -> mean score in a study_beam.py state.json, matched by (symmetry-invariant) design key."""
    from statistics import mean
    from coevolution.evolution.variation import design_key
    from hand_sampler.grammar import hand as gh
    rules = gh.FEETECH_COARSE_NOCUT_RULES
    by_key = {r["key"]: mean(r["scores"]) for r in json.loads((REPO / state_file).read_text())["designs"].values()
              if r.get("scores")}
    out = {}
    for sp in specs:
        k = design_key(gh.hand_from_dict(design_of(sp)), rules)
        if k in by_key:
            out[sp] = by_key[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scores-from", default=None)
    ap.add_argument("--port", type=int, default=8089)
    args = ap.parse_args()
    import viser

    man = json.loads((REPO / args.manifest).read_text())
    parent = next(h for h in man["hands"] if h.get("niche") == "parent")
    groups: dict[str, list[dict]] = {}
    for h in man["hands"]:
        if h is not parent:
            groups.setdefault(h["niche"], []).append(h)
    score = scores_by_key(args.scores_from, [h["spec"] for h in man["hands"]]) if args.scores_from else {}
    for hs in groups.values():
        hs.sort(key=lambda h: -score.get(h["spec"], -1.0))
    ps = score.get(parent["spec"])
    drawer = HandDrawer()

    def describe(h: dict) -> str:
        s = f" · score **{score[h['spec']]:.2f}**" if h["spec"] in score else ""
        if h is not parent and h["spec"] in score and ps is not None:
            s += f" (**{score[h['spec']] - ps:+.2f}** vs the parent)"
        head = [f"**{'PARENT ' if h is parent else ''}{h['id']}**{s}", f"niche {h.get('cell', '?')}"
                + ("" if h is parent else f" · edit: **{h['edit']}**")]
        return "  \n".join(head + describe_design(design_of(h["spec"])))

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    opts = [f"{NAMES.get(op, op)} ({len(hs)})" for op, hs in groups.items()]
    op_of = dict(zip(opts, groups))
    with server.gui.add_folder("mutation"):
        g_op = server.gui.add_dropdown("show children by", tuple(opts), initial_value=opts[0])
        g_curl = server.gui.add_slider("curl (flexion)", min=0.0, max=1.0, step=0.05, initial_value=0.0)
        g_flab = server.gui.add_checkbox("label fingers (f0, f1, ...)", True)
        g_line = server.gui.add_checkbox("black bar at every joint", True)
        g_info = server.gui.add_markdown("")
    server.gui.add_markdown("flexion **blue**, abduction **red**, roll **green** (roll tips striped), palm grey; "
                            "a black bar at every joint  \nhover a link to highlight it, click it for the link and the design")
    g_sel = server.gui.add_markdown("")
    handles: list = []
    box = [np.zeros(3), np.ones(3)]          # the layout being drawn
    shown = [np.zeros(3), np.ones(3)]        # the last finished layout: what the camera frames

    def place(h: dict, off, title: str, sub: str) -> None:
        hs, (lo, hi) = drawer.draw(server, f"/hands/{h['id']}", h["spec"], off, float(g_curl.value),
                                   bool(g_line.value), bool(g_flab.value),
                                   on_click=lambda link, h=h: setattr(g_sel, "content", describe_link(design_of(h["spec"]), link)
                                                                     + "  \n\n" + describe(h)))
        handles.extend(hs)
        y = lo[1] - 0.03
        handles.append(server.scene.add_label(f"/hands/{h['id']}/title", title, position=(off[0], y, 0.0)))
        if sub:
            handles.append(server.scene.add_label(f"/hands/{h['id']}/sub", sub, position=(off[0], y - 0.045, 0.0)))
        box[0] = np.minimum(box[0], lo - [0.0, 0.08, 0.0])
        box[1] = np.maximum(box[1], hi)

    def redraw(*_) -> None:
        for hd in handles:
            hd.remove()
        handles.clear()
        op = op_of[str(g_op.value)]
        hs = groups[op]
        cols = max(1, int(np.ceil(np.sqrt(len(hs) * 1.6))))
        rows = (len(hs) - 1) // cols + 1
        x0 = 1.4 * GAP
        box[0], box[1] = np.full(3, np.inf), np.full(3, -np.inf)
        place(parent, np.array([0.0, -(rows - 1) * GAP / 2, 0.0]), "PARENT" + (f" {ps:.2f}" if ps else ""), "")
        for k, h in enumerate(hs):
            s = score.get(h["spec"])
            t = f"#{k + 1}" + (f"  {s:.2f}" if s is not None else "") + (f" ({s - ps:+.2f})" if s is not None and ps else "")
            place(h, np.array([x0 + (k % cols) * GAP, -(k // cols) * GAP, 0.0]), t, short(h["edit"], design_of(h["spec"])))
        better = sum(score.get(h["spec"], -9) > ps for h in hs) if ps is not None else None
        g_info.content = (f"**{NAMES.get(op, op)}**: {len(hs)} children" + (", best first" if score else "")
                          + (f"; {better} beat the parent ({ps:.2f})" if better is not None else ""))
        shown[0], shown[1] = box[0].copy(), box[1].copy()
        for c in server.get_clients().values():
            aim(c, *shown)
        g_sel.content = describe(parent)

    server.on_client_connect(lambda c: (time.sleep(0.3), aim(c, *shown)))
    for g in (g_op, g_curl, g_flab, g_line):
        g.on_update(redraw)
    redraw()
    print(f"[viser-children] {parent['id']}: {sum(len(v) for v in groups.values())} children by "
          f"{len(groups)} mutations; http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
