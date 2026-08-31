"""Step through the hand-design grammar one random draw at a time, in viser.

    .venv_isaacsim/bin/python -m viewers.grammar_viewer --port 8089

Pure kinematics: no Isaac Sim, no GPU, starts in seconds.

WHAT IT SHOWS. A design is a sequence of independent draws followed by two
gates, and the sampler rejects roughly 93% of what it draws. That rejection is
where the design space actually lives -- the accepted population is the thin
residue -- so this walks the draws in order, then runs the SAME two gates the
population build runs and shows exactly which check failed and by how much.

    draws        n_fingers, palm extents, then per finger:
                 n_joints -> radius_scale -> metacarpal -> mount -> AA -> links
    gate 1       params.validate: reach bounds and mount separation
    gate 2       capsule_collision.is_collision_free: analytic self-collision

**The draws are recorded from the real sampler, not re-enacted.** A tracing RNG
records every call ``params.sample`` makes, and the labels are assigned by
walking the grammar over that recording. If the sampler's draw order ever
changes, the labeller runs out of trace and says so instead of quietly
mislabelling a column of numbers -- which is the only failure mode that would
make this tool actively misleading.

Geometry is drawn with ``capsule_collision.hand_capsules`` and ``palm_box``,
i.e. the very primitives the collision gate tests, so what is on screen is what
is being checked rather than a lookalike.
"""

from __future__ import annotations

import argparse
import functools
import math
import random
import time
import traceback

import numpy as np

from hand_sampler import params as P
from genmech.tools import capsule_collision as CC

PALM_COLOR = (201, 198, 189)
FINGER_COLORS = [(58, 122, 214), (232, 122, 63), (36, 168, 128),
                 (214, 96, 150), (156, 122, 214)]
HIT_COLOR = (214, 62, 62)
PENDING_COLOR = (170, 168, 160)


class TracingRandom(random.Random):
    """A Random that records every draw the sampler makes, in order."""

    def __init__(self, seed):
        super().__init__(seed)
        self.trace: list[tuple[str, tuple, object]] = []

    def uniform(self, a, b):
        v = super().uniform(a, b)
        self.trace.append(("uniform", (a, b), v))
        return v

    def randint(self, a, b):
        v = super().randint(a, b)
        self.trace.append(("randint", (a, b), v))
        return v

    def choice(self, seq):
        v = super().choice(seq)
        self.trace.append(("choice", tuple(seq), v))
        return v


def label_trace(trace, n_fingers_fixed=None) -> list[dict]:
    """Walk the grammar over a recorded trace and name every draw.

    Consumes the trace in the order ``params.sample`` produces it. Raises if the
    two disagree: a silent mislabel would be worse than no tool.
    """
    i = 0
    out: list[dict] = []

    def take(kind, label, group, note=""):
        nonlocal i
        if i >= len(trace):
            raise RuntimeError(
                f"grammar/trace mismatch: expected {kind} for {label!r} but the "
                f"trace ended at {i}. params.sample's draw order changed; update "
                f"label_trace.")
        k, rng_range, value = trace[i]
        if k != kind:
            raise RuntimeError(
                f"grammar/trace mismatch at draw {i}: expected {kind} for "
                f"{label!r}, recorded {k}. Update label_trace.")
        i += 1
        out.append({"label": label, "group": group, "kind": k,
                    "range": rng_range, "value": value, "note": note})
        return value

    if n_fingers_fixed is None:
        n_fingers = take("choice", "n_fingers", "hand",
                         "how many finger slots are active")
    else:
        n_fingers = n_fingers_fixed
    for axis, note in zip("xyz", ("thickness", "width", "height")):
        take("uniform", f"palm_extents.{axis}", "hand", note)

    for f in range(n_fingers):
        g = f"finger {f}"
        n_joints = take("randint", "n_joints", g,
                        "first n rungs of ACTIVATION_ORDER")
        take("uniform", "radius_scale", g,
             "drawn BEFORE lengths: it sets each segment's minimum buildable length")
        if n_joints >= P.MIN_JOINTS_FOR_METACARPAL:
            take("uniform", "mc region", g, "virtual-link band or buildable band")
            take("uniform", "mc_length", g)
        else:
            out.append({"label": "mc_length", "group": g, "kind": "fixed",
                        "range": None, "value": 0.0,
                        "note": f"no metacarpal below {P.MIN_JOINTS_FOR_METACARPAL} joints"})
        take("choice", "mount.face", g, "which palm face the finger sits on")
        take("uniform", "mount.u", g, "position across the face")
        take("uniform", "mount.v", g, "position along the face")
        take("uniform", "mount.roll", g)
        take("uniform", "mount.tilt", g)
        take("uniform", "mount.tilt_azimuth", g)
        take("uniform", "AA half-range", g, "the one joint limit the sampler varies")
        for tier in ("pp", "mp", "dp"):
            take("uniform", f"{tier} region", g)
            take("uniform", f"{tier}_length", g)

    if i != len(trace):
        raise RuntimeError(
            f"grammar/trace mismatch: labelled {i} draws but the sampler made "
            f"{len(trace)}. Update label_trace.")
    return out


def draw_attempt(seed_rng: TracingRandom, name: str, **sample_kwargs) -> dict:
    """One attempt: draw, label, and run BOTH gates. Never raises on rejection."""
    start = len(seed_rng.trace)
    hand, verdict, detail = None, None, ""
    # validate() is suppressed during the draw so an INVALID hand still comes
    # back as geometry. Otherwise sample() raises and the one thing you want to
    # look at -- the hand that was rejected -- is the one thing you cannot see.
    # The checks are then run in full by validity_report, which reports every
    # failure rather than only the first.
    original_validate = P.validate
    P.validate = lambda _h: None
    try:
        hand = P.sample(seed_rng, name=name, **sample_kwargs)
    except Exception as exc:  # noqa: BLE001 - _u_segment can raise ValueError
        verdict, detail = "error", f"{type(exc).__name__}: {exc}"
    finally:
        P.validate = original_validate

    trace = seed_rng.trace[start:]
    try:
        steps = label_trace(trace, n_fingers_fixed=sample_kwargs.get("n_fingers"))
    except RuntimeError as exc:
        steps = [{"label": "LABELLER OUT OF SYNC", "group": "!", "kind": "-",
                  "range": None, "value": str(exc), "note": ""}]

    hits, report = [], []
    if hand is not None:
        report = validity_report(hand)
        failed = [w for w, ok, _ in report if not ok]
        hits = CC.analytic_hand_hits(hand)
        if failed:
            verdict = "invalid"
            detail = f"{len(failed)} check(s) failed: {failed[0]}"
        elif hits:
            verdict = "collision"
            detail = (f"{len(hits)} overlapping pair(s); deepest "
                      f"{hits[0][0]} vs {hits[0][1]} by {hits[0][2]*1000:.2f} mm")
        else:
            verdict, detail = "accepted", "passes both gates"

    return {"hand": hand, "steps": steps, "verdict": verdict, "report": report,
            "detail": detail, "hits": hits, "name": name}


def validity_report(hand: P.HandParams) -> list[tuple[str, bool, str]]:
    """Re-run params.validate's checks, but report every one instead of the first.

    validate() raises on the first failure, which is right for a sampler and
    wrong for a viewer: seeing that a hand fails reach tells you nothing about
    whether its mounts were also too close.
    """
    out = []
    active = hand.active_fingers
    for f in active:
        r = f.reach()
        out.append((f"{f.name}: reach {r*1000:.1f} mm",
                    P.MIN_REACH <= r <= P.MAX_REACH,
                    f"must be {P.MIN_REACH*1000:.0f}-{P.MAX_REACH*1000:.0f} mm"))
    for i, a in enumerate(active):
        for b in active[i+1:]:
            d = math.dist(a.mount.xyz, b.mount.xyz)
            out.append((f"{a.name}/{b.name}: mounts {d*1000:.1f} mm apart",
                        d >= P.MIN_MOUNT_SEPARATION,
                        f"must be >= {P.MIN_MOUNT_SEPARATION*1000:.0f} mm"))
    return out


@functools.lru_cache(maxsize=512)
def _unit_capsule(height_mm: int, radius_mm: int):
    """A capsule along +z, cached by its rounded dimensions.

    Regenerating trimesh capsules on every redraw cost 127 ms for one hand,
    which made scrubbing feel broken. Scrubbing changes ONE segment per draw and
    revisits the same dimensions constantly, so caching on rounded mm collapses
    that to a lookup plus a vertex transform.
    """
    import trimesh
    h = max(height_mm, 1) / 1000.0
    r = max(radius_mm, 1) / 1000.0
    return trimesh.creation.capsule(height=h, radius=r, count=[6, 10])


def capsule_mesh(p0, p1, radius):
    """A capsule spanning p0->p1, as (vertices, faces)."""
    import trimesh

    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    axis = p1 - p0
    height = float(np.linalg.norm(axis))
    if height < 1e-9:
        m = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        return m.vertices + p0, m.faces

    base = _unit_capsule(int(round(height * 1000)), int(round(radius * 1000)))
    z = np.array([0.0, 0.0, 1.0])
    a = axis / height
    v = np.cross(z, a)
    c = float(np.dot(z, a))
    R = np.eye(3)
    if np.linalg.norm(v) > 1e-9:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    elif c < 0:
        R = np.diag([1.0, -1.0, -1.0])
    # Transform the cached vertices instead of rebuilding the mesh.
    return base.vertices @ R.T + p0, base.faces


def partial_capsules(hand: P.HandParams, revealed: dict[int, set[str]]):
    """Collision capsules for only the parts the revealed draws have defined.

    Per finger rather than per hand: ``HandParams`` refuses to exist with fewer
    than two active fingers ("a hand needs at least 2 to oppose anything"), so a
    partially drawn hand cannot be expressed as one. This calls the same
    ``finger_link_transforms`` / tier-radius path ``hand_capsules`` uses, so a
    partial render is a true prefix of the final one rather than a lookalike.
    """
    from dataclasses import replace

    from hand_sampler.urdf import has_collision_geometry, link_name

    out = []
    for i, fp in enumerate(hand.fingers):
        seen = revealed.get(i)
        if not fp.active or seen is None or "mount" not in seen:
            continue
        # Undrawn segments contribute nothing yet.
        fp = replace(
            fp,
            mc=fp.mc if "mc" in seen else P.Segment(xyz=(0.0, 0.0, 0.0)),
            pp_length=fp.pp_length if "pp" in seen else 0.0,
            mp_length=fp.mp_length if "mp" in seen else 0.0,
            dp_length=fp.dp_length if "dp" in seen else 0.0,
        )
        tf = CC.finger_link_transforms(fp)
        for part, tier in CC._PART_TIER:
            if not has_collision_geometry(fp, tier):
                continue
            length = fp.segment_length(tier)
            radius = CC.A.TIER_RADIUS_M[tier] * fp.radius_scale
            m = tf[part]
            p0 = m @ np.array([radius, 0.0, 0.0, 1.0])
            p1 = m @ np.array([length - radius, 0.0, 0.0, 1.0])
            out.append((link_name(i, part), p0[:3], p1[:3], radius, i))
    return out


def revealed_state(steps: list[dict], cursor: int) -> tuple[dict[int, set[str]], bool]:
    """Which parts of which finger the first `cursor` draws have defined."""
    out: dict[int, set[str]] = {}
    palm = 0
    for s in steps[:cursor]:
        if s["label"].startswith("palm_extents"):
            palm += 1
            continue
        if not s["group"].startswith("finger"):
            continue
        i = int(s["group"].split()[-1])
        seen = out.setdefault(i, set())
        lab = s["label"]
        if lab == "mount.tilt_azimuth":
            seen.add("mount")           # the mount is only placed once all 6 land
        elif lab == "mc_length":
            seen.add("mc")
        elif lab.endswith("_length"):
            seen.add(lab[:2])
        seen.add(lab)
    return out, palm >= 3


# Plain text, no raw HTML: viser's markdown renderer refuses a <span style=...>
# and shows "Markdown Failed to Render" instead of the panel.
VERDICT_STYLE = {
    "accepted":  "ACCEPTED - passes both gates",
    "invalid":   "REJECTED - failed validity",
    "collision": "REJECTED - self-collision",
    "error":     "SAMPLER ERROR",
}


class GrammarViewer:
    """One attempt at a time, scrubbed draw by draw."""

    def __init__(self, args):
        import viser

        self.args = args
        self.rng = TracingRandom(args.seed)
        self.attempt_no = 0
        self.counts = {"accepted": 0, "invalid": 0, "collision": 0, "error": 0}
        self.att = None
        self.handles: list = []
        self._pending = None
        self._last_tick = 0.0

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)

        @self.server.on_client_connect
        def _(client) -> None:
            client.camera.position = (0.34, -0.30, 0.24)
            client.camera.look_at = (0.0, 0.0, 0.07)

        self.server.scene.add_grid("/grid", width=0.6, height=0.6, cell_size=0.02)
        self._build_gui()
        self._new_attempt()

    # --- gui ---------------------------------------------------------------
    def _build_gui(self):
        g = self.server.gui
        self.md_verdict = g.add_markdown("")

        with g.add_folder("Scrub the draws", expand_by_default=True):
            # The primary control. A slider you can drag both ways beats a row
            # of Next buttons: the question is usually "what did THAT draw do",
            # which needs stepping back as often as forward.
            self.sl_step = g.add_slider("draw", min=0, max=1, step=1,
                                        initial_value=0)
            self.sl_step.on_update(lambda _: self._flag("scrub"))
            self.md_now = g.add_markdown("")
            self.cb_play = g.add_checkbox("play", False)
            self.sl_speed = g.add_slider("speed", min=1, max=20, step=1,
                                         initial_value=5)

        with g.add_folder("Next attempt", expand_by_default=True):
            self.btn_any = g.add_button("Any")
            self.btn_acc = g.add_button("Accepted")
            self.btn_rej = g.add_button("Rejected")
            self.btn_any.on_click(lambda _: self._flag("new"))
            self.btn_acc.on_click(lambda _: self._flag("accept"))
            self.btn_rej.on_click(lambda _: self._flag("reject"))
            self.md_stats = g.add_markdown("")

        with g.add_folder("Why", expand_by_default=True):
            self.md_why = g.add_markdown("")

        with g.add_folder("All draws", expand_by_default=False):
            self.md_draws = g.add_markdown("")

    def _flag(self, what):
        # viser callbacks run on their own thread; the main loop owns all state.
        self._pending = what

    # --- attempts ----------------------------------------------------------
    def _new_attempt(self, cursor=None, redraw=True):
        self.attempt_no += 1
        kw = {"n_fingers": self.args.n_fingers} if self.args.n_fingers else {}
        self.att = draw_attempt(self.rng, f"draw_{self.attempt_no:05d}", **kw)
        self.counts[self.att["verdict"]] = self.counts.get(self.att["verdict"], 0) + 1
        n = len(self.att["steps"])
        self.sl_step.max = max(1, n)
        self.sl_step.value = n if cursor is None else cursor
        if redraw:
            self._redraw()

    def _seek(self, want_accept: bool):
        """Draw until the verdict matches, WITHOUT rendering the ones skipped.

        Acceptance is ~3%, so "next accepted" burns about 30 attempts. Rendering
        each one meant ~30 full scene rebuilds -- every capsule removed and
        regenerated through trimesh -- back to back on the main loop, which
        froze the UI on "drawing..." for seconds at a time. Only the attempt you
        actually land on is worth drawing.
        """
        for _ in range(6000):
            self._new_attempt(cursor=0, redraw=False)
            if (self.att["verdict"] == "accepted") == want_accept:
                self._redraw()
                return
        self._redraw()
        self.md_why.content = "**no such attempt in 6000 draws**"

    # --- scene -------------------------------------------------------------
    def _clear(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def _redraw(self):
        self._clear()
        att = self.att
        hand, steps = att["hand"], att["steps"]
        cursor = int(self.sl_step.value)
        done = cursor >= len(steps)

        if hand is not None:
            revealed, palm_ready = revealed_state(steps, cursor)
            if palm_ready:
                lo, hi = CC.palm_box(hand)
                self.handles.append(self.server.scene.add_box(
                    "/palm", dimensions=tuple(hi - lo),
                    position=tuple((lo + hi) / 2), color=PALM_COLOR, opacity=0.5))

            hit_bodies = {n for a, b, _ in att["hits"] for n in (a, b)} if done else set()
            for nm, p0, p1, r, idx in partial_capsules(hand, revealed):
                color = FINGER_COLORS[idx % len(FINGER_COLORS)]
                if nm in hit_bodies:
                    color = HIT_COLOR
                verts, faces = capsule_mesh(p0, p1, r)
                self.handles.append(self.server.scene.add_mesh_simple(
                    f"/cap/{nm}", vertices=verts, faces=faces, color=color))

            # A mount that is placed but has no segments yet still shows, so the
            # six mount draws visibly move something.
            for i, f in enumerate(hand.fingers):
                seen = revealed.get(i)
                if not f.active or seen is None or "mount" not in seen:
                    continue
                self.handles.append(self.server.scene.add_frame(
                    f"/mount/{i}", axes_length=0.022, axes_radius=0.0012,
                    position=tuple(float(v) for v in f.mount.xyz)))
        self._refresh_text(cursor, done)

    def _refresh_text(self, cursor, done):
        att, steps = self.att, self.att["steps"]
        n = len(steps)

        if done:
            title = VERDICT_STYLE.get(att["verdict"], "?")
            self.md_verdict.content = (
                f"## {title}\n\nattempt {self.attempt_no} - {att['detail']}")
        else:
            self.md_verdict.content = (
                f"# drawing...\n\nattempt {self.attempt_no} - "
                f"draw {cursor} of {n}, gates not run")

        if cursor == 0:
            self.md_now.content = "_drag the slider, or press play_"
        else:
            s = steps[cursor - 1]
            v = s["value"]
            vs = f"{v:.4f}" if isinstance(v, float) else str(v)
            if s["kind"] == "uniform":
                src = f"U({s['range'][0]:.4g}, {s['range'][1]:.4g})"
            elif s["kind"] == "randint":
                src = f"randint{s['range']}"
            elif s["kind"] == "choice":
                src = f"choice{tuple(s['range'])}"
            else:
                src = "fixed"
            self.md_now.content = (
                f"**{s['group']}**\n\n"
                f"## `{s['label']}` = {vs}\n\n"
                f"drawn from `{src}`"
                + (f"\n\n_{s['note']}_" if s["note"] else ""))

        if not done:
            self.md_why.content = "_gates run once every draw is revealed_"
        elif att["hand"] is None:
            self.md_why.content = f"**{att['detail']}**"
        else:
            lines = ["**validity**"]
            for what, ok, req in att["report"]:
                lines.append(f"- {'PASS' if ok else '**FAIL**'} {what} _{req}_")
            lines.append("\n**self-collision**")
            if att["hits"]:
                lines += [f"- `{a}` / `{b}` overlap **{d*1000:.2f} mm**"
                          for a, b, d in att["hits"][:6]]
                if len(att["hits"]) > 6:
                    lines.append(f"- _...and {len(att['hits'])-6} more_")
            else:
                lines.append("- PASS, nothing overlaps")
            self.md_why.content = "\n".join(lines)

        rows, group = [], None
        for k, s in enumerate(steps):
            if s["group"] != group:
                group = s["group"]
                rows.append(f"\n**{group}**")
            v = s["value"]
            vs = f"{v:.4f}" if isinstance(v, float) else str(v)
            if k < cursor:
                mark = "**>>** " if k == cursor - 1 else "- "
                rows.append(f"{mark}`{s['label']}` = {vs}")
            else:
                rows.append(f"- `{s['label']}` = ?")
        self.md_draws.content = "\n\n".join(rows)

        c, tot = self.counts, max(1, self.attempt_no)
        self.md_stats.content = (
            f"**{c['accepted']}** accepted of **{self.attempt_no}** "
            f"({100*c['accepted']/tot:.1f}%)\n\n"
            f"rejected: {c['invalid']} validity, {c['collision']} collision")

    # --- loop --------------------------------------------------------------
    def run(self):
        print(f"\n  design-grammar viewer  http://localhost:{self.args.port}\n")
        try:
            while True:
                p, self._pending = self._pending, None
                if p == "scrub":
                    self._redraw()
                elif p == "new":
                    self._new_attempt(cursor=0)
                elif p == "accept":
                    self._seek(True)
                elif p == "reject":
                    self._seek(False)

                if self.cb_play.value:
                    if time.time() - self._last_tick >= 1.0 / max(1, self.sl_speed.value):
                        self._last_tick = time.time()
                        if int(self.sl_step.value) >= len(self.att["steps"]):
                            self._new_attempt(cursor=0)
                        else:
                            self.sl_step.value = int(self.sl_step.value) + 1
                            self._redraw()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\n[grammar] stopped")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--port", type=int, default=8089)
    p.add_argument("--n_fingers", type=int, default=None,
                   help="Pin the finger count instead of drawing it")
    args = p.parse_args()
    GrammarViewer(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
