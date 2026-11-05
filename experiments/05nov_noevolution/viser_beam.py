"""The beam search of study_beam.py, live, in viser, in two uncluttered views.

    .venv_isaacsim/bin/python experiments/05nov_noevolution/viser_beam.py \
        --state debug_outputs/noevolution/studies/beam_<start>_it<N>/state.json [--port 8091]

  best design's lineage   the current best traced back to the start, left to right, one hand per step: each with its
                          round and score, and between neighbours the edit that made the next one (the improvement
                          path, a handful of hands)
  one round's beam        a slider picks the round; its beam (the 10 best of the previous beam and its children) on a
                          spaced 5 x 2 grid, best first, each with its rank, score, edit (or "kept") and parent

Every hand is drawn exactly as deployed (viser_hands.py: joint-kind colours, a black bar at every joint, each link its
own mesh: hover highlights it, click reports the link, the design and its lineage). The state file is re-read every
30 s, so new rounds appear as they finish.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viser_children import NAMES, short  # noqa: E402
from viser_hands import HandDrawer, aim, describe_design, describe_link, design_of  # noqa: E402

STEP = 0.75          # m between hands in the lineage
COL, ROW = 0.62, 0.75     # m between hands in a round's grid
COLS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()
    import viser

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    drawer = HandDrawer()
    with server.gui.add_folder("beam search"):
        g_view = server.gui.add_dropdown("view", ("best design's lineage", "one round's beam"),
                                         initial_value="best design's lineage")
        g_round = server.gui.add_slider("round", min=0, max=1, step=1, initial_value=0)
        g_curl = server.gui.add_slider("curl (flexion)", min=0.0, max=1.0, step=0.05, initial_value=0.0)
        g_flab = server.gui.add_checkbox("label fingers (f0, f1, ...)", True)
        g_bars = server.gui.add_checkbox("black bar at every joint", True)
        g_rounds = server.gui.add_markdown("")
    server.gui.add_markdown("flexion **blue**, abduction **red**, roll **green** (roll tips striped), palm grey; "
                            "a black bar at every joint  \nhover a link to highlight it, click it for the link, "
                            "the design and its lineage")
    g_sel = server.gui.add_markdown("")
    handles: list = []
    box = [np.zeros(3), np.ones(3)]          # the layout being drawn
    shown = [np.zeros(3), np.ones(3)]        # the last finished layout: what the camera frames
    cur = {"mtime": None, "S": None}

    def score(D, i):
        return st.mean(D[i]["scores"]) if D[i]["scores"] else float("nan")

    def chain(D, i) -> list[str]:
        out = []
        while i is not None:
            out.append(i)
            i = D[i]["parent"]
        return out[::-1]

    def lineage_md(D, i) -> str:
        rows = []
        for j in chain(D, i):
            r = D[j]
            rows.append(f"- round {r['round']}: **{j}** {score(D, j):.2f}"
                        + (f" ← {NAMES.get(r['operator'], r['operator'])}: {r['edit']}" if r["operator"] else " (start)"))
        return "  \n".join(rows)

    def describe(D, i, link=None) -> str:
        r = D[i]
        head = ([describe_link(design_of(r["spec"]), link), ""] if link else []) + [
            f"**{i}** · score **{score(D, i):.2f}** ({len(r['scores'])} evaluations of 100 envs) · niche {r['niche']}"]
        return "  \n".join(head + describe_design(design_of(r["spec"])) + ["", "**lineage**", lineage_md(D, i)])

    def put(D, i, off, lines: list[str]) -> None:
        hs, (lo, hi) = drawer.draw(server, f"/h/{i}", D[i]["spec"], off, float(g_curl.value), bool(g_bars.value),
                                   bool(g_flab.value),
                                   on_click=lambda link, i=i: setattr(g_sel, "content", describe(D, i, link)))
        handles.extend(hs)
        y = lo[1] - 0.03
        for n, t in enumerate(lines):
            handles.append(server.scene.add_label(f"/h/{i}/l{n}", t, position=(off[0], y - 0.045 * n, 0.0)))
        box[0] = np.minimum(box[0], lo - [0.0, 0.04 + 0.045 * len(lines), 0.0])
        box[1] = np.maximum(box[1], hi)

    def redraw(*_) -> None:
        S = cur["S"]
        if S is None:
            return
        for hd in handles:
            hd.remove()
        handles.clear()
        D, R = S["designs"], S["rounds"]
        box[0], box[1] = np.full(3, np.inf), np.full(3, -np.inf)
        if g_view.value == "best design's lineage":
            best = R[-1]["best"]
            ids = chain(D, best)
            for k, i in enumerate(ids):
                r = D[i]
                lines = [f"round {r['round']}  ·  {score(D, i):.2f}"]
                if r["operator"]:
                    lines.append(f"{NAMES.get(r['operator'], r['operator'])}: {short(r['edit'], design_of(r['spec']))}")
                else:
                    lines.append("start")
                put(D, i, np.array([k * STEP, 0.0, 0.0]), lines)
                if k:
                    handles.append(server.scene.add_label(f"/arrow/{k}", "→", position=((k - 0.5) * STEP, 0.0, 0.05)))
            g_sel.content = describe(D, best)
            title = f"the best after round {len(R) - 1} (**{best}**, {score(D, best):.2f}) and its {len(ids) - 1} edits"
        else:
            r = int(g_round.value)
            beam = R[r]["beam"]
            for k, i in enumerate(beam):
                rec = D[i]
                made = ("start" if rec["parent"] is None else f"kept from round {rec['round']}" if rec["round"] < r
                        else f"{NAMES.get(rec['operator'], rec['operator'])}: {short(rec['edit'], design_of(rec['spec']))}")
                lines = [f"#{k + 1}  ·  {score(D, i):.2f}", made]
                if rec["parent"] is not None and rec["round"] == r:
                    lines.append(f"from {rec['parent']}")
                put(D, i, np.array([(k % COLS) * COL, -(k // COLS) * ROW, 0.0]), lines)
            g_sel.content = describe(D, beam[0])
            rd = R[r]
            title = (f"round {r}'s beam: best **{rd['best_score']:.2f}**"
                     + (f", mean {rd['beam_mean']:.2f}, {rd['new_in_beam']} new" if "beam_mean" in rd else ""))
        rows = "  \n".join(f"round {k}: best **{rd['best_score']:.2f}**"
                           + (f" · mean {rd['beam_mean']:.2f} · {rd['children']} children" if "beam_mean" in rd else "")
                           for k, rd in enumerate(R))
        g_rounds.content = (f"{title}  \n\nstart **{S['args']['start']}**, beam {S['args']['beam']}, "
                            f"{S['args']['envs']} envs per design  \n" + rows)
        shown[0], shown[1] = box[0].copy(), box[1].copy()
        for c in server.get_clients().values():
            aim(c, *shown)

    def reload() -> bool:
        p = REPO / args.state
        m = p.stat().st_mtime if p.exists() else None
        if m is None or m == cur["mtime"]:
            return False
        cur["mtime"], cur["S"] = m, json.loads(p.read_text())
        n = len(cur["S"]["rounds"])
        g_round.max = max(n - 1, 1)
        if g_round.value > n - 1:
            g_round.value = n - 1
        return True

    server.on_client_connect(lambda c: (time.sleep(0.3), aim(c, *shown)))
    for g in (g_view, g_round, g_curl, g_flab, g_bars):
        g.on_update(redraw)
    reload()
    g_round.value = len(cur["S"]["rounds"]) - 1 if cur["S"] else 0
    redraw()
    print(f"[viser-beam] {args.state}: {len(cur['S']['rounds']) if cur['S'] else 0} rounds; "
          f"http://localhost:{args.port}", flush=True)
    while True:
        time.sleep(30)
        if reload():
            redraw()
            print(f"[viser-beam] now {len(cur['S']['rounds'])} rounds", flush=True)


if __name__ == "__main__":
    main()
