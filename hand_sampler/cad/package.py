"""A design's kit: print-ready STLs, the OpenSCAD source, the parts list,
the assembly steps, the check report, all in one folder and a zip.

    outputs/designs/<id>/
        hand.json                 the genotype
        scad/hand.scad            the parametric source (and the library)
        stl/<part>.stl            each part in its own frame (the viewer's overlay)
        step/<part>.step          the same as a solid for CAD (`step.py`: its mesh, flat faces merged)
        print/<part>.stl          each part on the bed the way up it is strongest (`print_up`)
        BOM.md  ASSEMBLY.md  checks.md  report.json
    outputs/designs/<id>.zip
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from hand_sampler.grammar import derive as dv
from hand_sampler.grammar.hand import Hand, finger_section, hand_to_json

from . import checks, render, scad, step
from .design import MOUNT_DEPTH_MM, MOUNT_R_MM, Design, Part, ServoSpot
from .servo import HL3915

DESIGNS_DIR = Path("outputs/designs")
PRINT_BED_MM = 256.0
PETG_G_CM3 = 1.27
INFILL_SHARE = 0.5            # filament as a share of the solid volume (walls and 30-40% infill)


def design_id(hand: Hand) -> str:
    return hashlib.sha1(hand_to_json(hand).encode()).hexdigest()[:10]


@dataclass
class Kit:
    """A hand's kit. `build_parts` fills in what drawing it needs (the
    design, every part's STL); `finish` adds the checks, the print layout,
    the pictures, the parts list and the steps, and the zip."""
    folder: Path
    zip_path: Optional[Path]
    report: Optional[checks.Report]
    parts: List[str]
    seconds: float
    design: Optional[Design] = None
    rendered: Dict = field(default_factory=dict, repr=False)   # part name -> render.Rendered
    scad_path: Optional[Path] = None
    started: float = 0.0

    @property
    def ready(self) -> bool:
        return self.zip_path is not None

    def part_frames(self) -> Dict[str, tuple]:
        """Each printed part's STL: (its rigid body, its frame in that body,
        mm). Palm parts are in the palm's frame; the bay and the mount in
        the bay's frame, fixed to the main palm."""
        out = {}
        for p in self.design.parts.values():
            out[p.name] = (p.body, np.eye(4))
        for p in self.design.extras:
            out[p.name] = (("palm", -1), p.frame)
        return out


def generate(hand: Hand, out_root: Path = DESIGNS_DIR, progress: Optional[Callable[[str], None]] = None) -> Kit:
    """Design, render, check and package `hand`."""
    return finish(build_parts(hand, out_root, progress), progress)


def build_parts(hand: Hand, out_root: Path = DESIGNS_DIR, progress: Optional[Callable[[str], None]] = None) -> Kit:
    """What drawing the parts needs: the design and every part rendered
    (a link that came out in pieces is joined again). `finish` does the
    rest."""
    say = progress or (lambda s: None)
    t0 = time.time()
    did = design_id(hand)
    folder = Path(out_root) / did
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "scad").mkdir(parents=True)
    (folder / "hand.json").write_text(hand_to_json(hand))
    say("designing the parts")
    d = Design(hand)
    path = scad.write(d, folder / "scad")
    names = scad.printed_names(d)
    say(f"rendering {len(names)} parts with OpenSCAD")
    res = render.render_all(path, names, folder / "stl")
    # a link that came out in pieces (something turns through its tube) gets tubes to its cradle's
    # other faces; the design is rebuilt so its neighbours give way to them, and rendered again
    split = {k for k, p in d.parts.items() if k[0] == "link" and res.get(p.name) is not None
             and res[p.name].ok and _pieces(res[p.name].stl) > 1}
    if split:
        say(f"joining {len(split)} links again")
        d = Design(hand, more_tubes=split)
        path = scad.write(d, folder / "scad")
        res = render.render_all(path, names, folder / "stl")
    return Kit(folder, None, None, names, time.time() - t0, d, res, path, t0)


def finish(kit: Kit, progress: Optional[Callable[[str], None]] = None) -> Kit:
    """The rest of the kit, after `build_parts`: the checks, the parts laid
    out for printing, the STEP files, the assembly pictures, the parts list
    and the steps, the report and the zip."""
    say = progress or (lambda s: None)
    folder, d, res, names, path = kit.folder, kit.design, kit.rendered, kit.parts, kit.scad_path
    did = folder.name
    failed = [n for n, r in res.items() if not r.ok]
    say("checking the parts")
    rep = checks.check(d, folder / "stl")
    for n in failed:
        rep.parts.setdefault(n, {"ok": False, "problems": []})
        rep.parts[n]["ok"] = False
        rep.parts[n]["problems"].append("OpenSCAD failed: " + res[n].log.strip().splitlines()[-1][:200]
                                        if res[n].log.strip() else "OpenSCAD failed")
    say("laying the parts out for printing")
    (folder / "print").mkdir()
    by_name = {p.name: p for p in d.part_list()}
    for n in names:
        if res[n].ok:
            ext = lay_for_strength(folder / "stl" / f"{n}.stl", folder / "print" / f"{n}.stl", d, by_name[n])
            rep.parts[n]["print_extent_mm"] = ext
            if max(ext[:2]) > PRINT_BED_MM:
                rep.warnings.append(f"{n} is {max(ext[:2]):.0f} mm across: larger than a {PRINT_BED_MM:.0f} mm bed")
    say("writing the STEP files")
    (folder / "step").mkdir()
    for n in names:
        if res[n].ok:
            try:
                step.write_part(folder / "stl" / f"{n}.off", folder / "step" / f"{n}.step", n)
            except (ValueError, KeyError, ZeroDivisionError, FloatingPointError) as e:
                rep.warnings.append(f"{n}: no STEP file ({type(e).__name__}: {e})")
    say("drawing the assembly steps")
    steps = step_pictures(d, path, folder / "steps")
    (folder / "BOM.md").write_text(bom(d, rep))
    (folder / "ASSEMBLY.md").write_text(assembly(d, rep.orders, steps))
    (folder / "checks.md").write_text("# Checks\n\n```\n" + rep.summary() + "\n```\n")
    (folder / "report.json").write_text(json.dumps({"id": did, "ok": rep.ok, "parts": rep.parts,
                                                    "clashes": rep.clashes, "warnings": rep.warnings}, indent=1))
    say("zipping")
    zpath = folder.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.suffix != ".off":        # the OFF meshes are only for the checks
                z.write(f, f"hand_{did}/{f.relative_to(folder)}")
    kit.report, kit.zip_path, kit.seconds = rep, zpath, time.time() - kit.started
    return kit


def _pieces(stl: Path) -> int:
    _, man = render.load(stl)
    return sum(1 for q in man.decompose() if q.volume() > 1.0)


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------


FLEX_WEIGHT = 0.5
"""What a link's own bend axis counts toward which way up it prints, beside
the joints it carries (`print_up`): the finger closing on something bends
it about that axis, and a fingertip on a twist has nothing else."""


def print_up(d: Design, part: Part, verts: np.ndarray) -> np.ndarray:
    """The direction in `part`'s frame that points up on the printer, for
    strength rather than supports (Martin, 11/06). A printed part is weakest
    across its layers, so the moments it carries should turn within them.
    A link carries the torque of its own joint and of every joint beyond
    it, so it stands on the axis most of those turn about: for most fingers
    the bend axis, the U's arms flat on the bed, every layer a whole copy of
    the link as it bends. A link never stands on its long side, which would
    put its length across the layers. A palm part lies on its face."""
    if part.body[0] != "link":
        return np.array([1.0, 0.0, 0.0])
    _, i, j = part.body
    # at q = 0 every link of a finger has the same frame, so the joints beyond are in this one's too
    turns = [np.asarray(a, float) / np.linalg.norm(a) for a in dv.finger_geometry(d.hand.fingers[i]).axes[j:]]
    bend = np.array([0.0, 0.0, 1.0])

    def extent(u: np.ndarray) -> float:
        x = verts @ u
        return float(x.max() - x.min())

    best, key = bend, None
    for c in turns + [np.array([0.0, 1.0, 0.0]), bend]:
        e1 = np.cross(c, [1.0, 0.0, 0.0] if abs(c[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(c, e1)
        across = max(extent(math.cos(t) * e1 + math.sin(t) * e2) for t in np.linspace(0.0, math.pi, 12, endpoint=False))
        if extent(c) > across:          # on its long side
            continue
        k = (round(sum(abs(float(c @ a)) for a in turns) + FLEX_WEIGHT * abs(float(c @ bend)), 6), -extent(c))
        if key is None or k > key:
            best, key = c, k
    return best


def lay_for_strength(src: Path, dst: Path, d: Design, part: Part) -> List[float]:
    """The part with `print_up` pointing up, on whichever of its two sides
    across that has more of it flat on the bed, turned on the bed to its
    smallest rectangle, at z = 0, centred, its mesh the STEP file's (a
    slicer, joining its vertices, reads one closed solid: `step.opened`).
    Returns its extent (mm). Its overhangs need supports."""
    import trimesh

    m = trimesh.Trimesh(*step.opened(*step.part_mesh(src)), process=False)
    up = print_up(d, part, np.asarray(m.vertices, dtype=float))
    best, best_area = np.eye(4), -1.0
    for s in (1.0, -1.0):
        Tp = trimesh.geometry.align_vectors(s * up, [0.0, 0.0, 1.0])
        mm = m.copy()
        mm.apply_transform(Tp)
        z0 = float(mm.bounds[0, 2])
        foot = (mm.face_normals[:, 2] < -0.99) & (mm.triangles_center[:, 2] < z0 + 0.2)
        area = float(mm.area_faces[foot].sum())
        if area > best_area + 1e-6:
            best, best_area = Tp, area
    m.apply_transform(best)
    T2, _ = trimesh.bounds.oriented_bounds_2D(m.vertices[:, :2])      # turned on the bed to its smallest rectangle
    R = np.eye(4)
    R[:2, :2], R[:2, 3] = T2[:2, :2], T2[:2, 2]
    m.apply_transform(R)
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
    m.export(str(dst))
    return [round(float(x), 1) for x in (hi - lo)]


# --------------------------------------------------------------------------
# the parts list
# --------------------------------------------------------------------------


def _count_screws(d: Design) -> Dict[str, int]:
    out = {"box": 0, "M2x5": 0}
    for sp in d.servos.values():
        out["box"] += 4                           # the case: its far holes, or its idler face's four (a flange cup)
        out["M2x5"] += 4                          # the driven arm (or the crossing plate)
        if sp.yoke == "u":
            out["M2x5"] += 4                      # the idler arm (3 mm, as the driven one: a box screw is too short)
    return out


def bom(d: Design, rep: checks.Report) -> str:
    n_servo = len(d.servos)
    sc = _count_screws(d)
    grams = {n: round(info.get("volume_cm3", 0.0) * PETG_G_CM3 * INFILL_SHARE, 1) for n, info in rep.parts.items()}
    lines = ["# Parts list", "",
             f"Hand `{design_id(d.hand)}`: {len(d.hand.fingers)} fingers, {n_servo} servos"
             + (", a hinged palm" if d.hand.palm_joints else "") + ".", "",
             "## Bought", "",
             "| Item | Count | Notes |", "|---|---|---|",
             f"| {HL3915.name} (or HL-2915-C001) | {n_servo} | each comes with 8 M2 x 3.5 screws; "
             f"{sc['box']} of them are used here"
             + (f"; buy {sc['box'] - 8 * n_servo} more M2 x 3.5 pan heads" if sc['box'] > 8 * n_servo else "") + " |",
             f"| M2 x 5 pan head | {sc['M2x5']} | 4 on each arm of a yoke (or a crossing's plate) |",
             f"| 3-pin 1.25 mm (PicoBlade) servo cable | {n_servo} | one per servo, chained along each finger |",
             "| Palm Hub rev B.2 | 1 | the board: "
             + ("screwed to the palm's back |" if d.board else "no room on the palm's back: it floats on its cables |")]
    if d.board:
        lines.append("| M2 x 8 pan head | 4 | the board, into the 1.8 mm holes in its standoffs |")
    if d.mount:
        lines.append("| M5 heat-set insert (6.4 mm hole, up to 11 mm long) | 4 | the back mount: "
                     "ISO 9409-1-31.5-4-M5 |")
    lines += ["",
             "## Printed (PETG)", "",
              "| Part | Grams (about) | Size as printed (mm) |", "|---|---|---|"]
    for n, info in rep.parts.items():
        ext = info.get("print_extent_mm", info.get("extent_mm", []))
        lines.append(f"| `{n}` | {grams.get(n, 0)} | {' x '.join(str(x) for x in ext)} |")
    lines += ["", f"About {sum(grams.values()):.0f} g of filament in all.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the assembly
# --------------------------------------------------------------------------


def servo_id(sp: ServoSpot) -> int:
    if sp.key == ("hinge",):
        return 1
    _, i, j = sp.key
    return 10 * (i + 1) + j


def wiring(d: Design) -> Dict[int, str]:
    """The board's port (CH1-CH6) for each finger's chain; the hinge servo
    joins the chain of the first finger on the main palm."""
    out = {}
    for i in range(len(d.hand.fingers)):
        out[i] = f"CH{i + 1}"
    return out


def step_pictures(d: Design, scad_path: Path, out: Path) -> Dict[tuple, str]:
    """One picture per joint (`step_<servo>` in hand.scad), looking at the
    servo's driven horn from in front: {servo key: file name}."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    out.mkdir(parents=True, exist_ok=True)

    def one(sp):
        tag = "_".join(str(k) for k in sp.key)
        S = sp.S_parent
        c = S[:3, 3]
        dirn = S[:3, 0] - 0.8 * S[:3, 1] + 0.5 * S[:3, 2]
        eye = c + 250.0 * dirn / np.linalg.norm(dirn)
        png = out / f"step_{tag}.png"
        cam = ",".join(f"{v:.2f}" for v in (*eye, *c))
        cmd = [render.openscad(), "-o", str(png.resolve()), "--render", "--imgsize=900,700", "--viewall",
               "--autocenter", "--projection=p", f"--camera={cam}", "--colorscheme=Tomorrow", *render.backend_flags(),
               "-D", f'part="step_{tag}"', str(Path(scad_path).resolve())]
        try:
            subprocess.run(cmd, capture_output=True, timeout=300, cwd=str(Path(scad_path).parent))
        except Exception:
            return sp.key, None
        return sp.key, (png.name if png.exists() else None)

    with ThreadPoolExecutor(max_workers=8) as ex:
        return {k: v for k, v in ex.map(one, list(d.servos.values())) if v}


def assembly(d: Design, orders: Optional[Dict] = None, steps: Optional[Dict] = None) -> str:
    hand = d.hand
    ports = wiring(d)
    L = ["# Assembly", "",
         f"Hand `{design_id(hand)}`. Parts are named `f<finger>_l<link>` (the link after joint <link>), "
         "`f<finger>_tip`, `palm` (and `palm_beyond` past the cut).", "",
         "## Printing", "",
         "Each part in `print/` lies the way up it is strongest, not the way that needs the fewest supports. A "
         "printed part is weakest across its layers, so each link stands on the axis its joints turn about (for "
         "most links the bend axis, the U's arms flat on the bed): its length, its bending and the servos' torques "
         "all lie within its layers. The palm lies on its face. Print with supports: the upper arm of each U and "
         "a fingertip's round side overhang.", "",
         "## 1. Servos", "",
         "Set each servo's ID and move it to its centre (position 2048) before fitting anything to its horns "
         "(Feetech's FD software, or the board).", "",
         "| Servo | ID | Joint | Held by | Horns in | Arms | Order | Board port |", "|---|---|---|---|---|---|---|---|"]
    names = {k: p.name for k, p in d.parts.items()}
    main_finger = next((i for i in range(len(hand.fingers)) if finger_section(hand, i) < 0), 0)
    for sp in sorted(d.servos.values(), key=servo_id):
        if sp.key == ("hinge",):
            joint, port = "palm hinge", ports[main_finger]
        else:
            _, i, j = sp.key
            kind = "bend" if abs(sp.S_child[2, 0]) > 0.99 else ("twist" if abs(sp.S_child[0, 0]) > 0.99 else "spread/oblique")
            how = ", turned" if sp.turned else (", flipped" if sp.flipped else "")
            joint, port = f"finger {i} joint {j} ({kind}{how})", ports[i]
        arms = {"u": "two (driven + idler)", "single": "one on the driven horn (idler horn off; flange cup)",
                "crossing": "plate on the driven horn"}[sp.yoke]
        held = names[sp.parent]
        order = (orders or {}).get(sp.key) or "servo first"
        L.append(f"| {'/'.join(map(str, sp.key))} | {servo_id(sp)} | {joint} | {held} | {names[sp.child]} | "
                 f"{arms} | {order} | {port} |")
    L += ["",
          "Servos marked 'one on the driven horn' (a twist): unscrew the idler horn's centre screw and pull the "
          "horn off. The bare stub goes through the plate of a flange cup on the part before the joint: slide the "
          "case into the cup along its axis and screw its idler face to the plate (4 box screws, near and far "
          "holes). Do this before that part goes onto the previous servo's horns: the screwdriver reaches the "
          "screws through the holes in that part's bridge.", "",
          "## 2. Fitting a link to a servo's horns (every joint)", "",
          "1. Take out both horns' centre screws.",
          "2. Slide the link's yoke over the horns from the servo's front, its arms flat on the horns, with the "
          "joint at its zero pose (the finger straight, the link on the line of the one before).",
          "3. Put the centre screws back through the arms' holes.",
          "4. Each arm: 4 x M2 x 5 into its horn.", "",
          "## 3. Palm", "",
          "The palm is a tray: a plate whose face stands 3.5 mm over the base servos' tops, one wall round its "
          "edge and a grid of ribs under the face, open at the back. Each base servo screws straight into its "
          "housing under the face; each finger's base turns in a walled pocket, open where the finger leaves "
          "the palm. Every finger is built off the hand on its base servo (step 4) and slotted into the palm "
          "whole, so no screwdriver needs a way through the palm to an arm.", "",
          "1. A finger whose base servo lies at the palm's edge slides in from the edge, the servo's back end "
          "into its housing; one rising out of the face drops in through its window; a crossing's goes in from "
          "the palm's open back, the finger up through its pocket.",
          "2. Screw the base servo's case to its housing: the screwdriver reaches the case screws along the "
          "servo's axis under the face, or from outside through the holes in the palm's wall. Two screws on "
          "one side are enough where the other side is closed in.",
          "3. The cables leave through the side walls' windows and out at the palm's open back.", ""]
    if hand.palm_joints:
        L += ["3. The hinge servo sits in a housing of the main palm; `palm_beyond` is fitted to its horns as in "
              "step 2.", ""]
    if steps:
        L += ["## Pictures", "",
              "Each joint: the part holding the servo, the servo (horns, centre screws, case screws in gold) and "
              "the next link pulled off the way it slides on, with its arm screws.", ""]
        for sp in sorted(d.servos.values(), key=servo_id):
            if sp.key in steps:
                L += [f"**Servo {'/'.join(map(str, sp.key))}** (ID {servo_id(sp)}, "
                      f"{(orders or {}).get(sp.key) or 'servo first'}): `{names[sp.parent]}` holds it, "
                      f"`{names[sp.child]}` goes on its horns.", "", f"![servo {sp.key}](steps/{steps[sp.key]})", ""]
    L += ["## 4. Fingers, built off the hand", "",
          "Each finger from its base servo out, on the bench, then into the palm (step 3). For each joint, in "
          "the order the table gives:", "",
          "- **servo first:** slide the servo into its housing from the front (its horns pass through the "
          "slots), push it to the back wall and screw the case to the side plates (4 of the servo's box screws, "
          "through the far holes). Then fit the part the table's 'Horns in' names to its horns (step 2).",
          "- **link first:** fit that part to the servo's horns on the bench (step 2), then slide the "
          "servo, part and all, into its housing from the front and screw the case to the side plates.", "",
          "A flipped servo lies along the link after its joint: that link holds its case, and the link before "
          "it carries its horns (so that link has a yoke at each end). A turned one stands on the finger's back.", "",
          "## 5. Cables", "",
          "Each finger's servos are chained on the finger's back: board port -> base servo -> next servo. Leave "
          "about 25 mm of slack across each joint and tie the cable to the links. The palm's cables run in its "
          "open back.", "",
          "| Finger | Board port |", "|---|---|"]
    for i in range(len(hand.fingers)):
        L.append(f"| {i} | {ports[i]} |")
    L += ["", "## 6. Board and arm", ""]
    if d.mount and d.mount["column"]:
        L += [f"The back mount is the column behind the palm ({2 * MOUNT_R_MM:.0f} mm across, "
              f"{MOUNT_DEPTH_MM:.0f} mm long, where the simulator mounts the hand). Press the four M5 heat-set inserts "
              "into its end with a soldering iron; they take an arm's ISO 9409-1-31.5-4-M5 flange, or an adapter "
              "to another."]
    elif d.mount:
        L += ["Something stands behind the palm where the back mount's column would, so the four M5 heat-set "
              "inserts go straight into the palm's back face round its centre (ISO 9409-1-31.5-4-M5)."]
    if d.mount and d.mount["turn_deg"]:
        L[-1] += f" The pattern is turned {d.mount['turn_deg']} degrees from the palm's axes."
    if d.mount and d.mount["ways"]:
        L += ["", f"The {d.mount['ways']} holes through the column (4 mm) are a screwdriver's way to the case "
              "screws of the servos beside it: drive those screws with a long-shafted driver before fitting the "
              "hand to the arm."]
    if d.board:
        y, z = d.board["centre"]
        uy, uz = d.board["usb_faces"]
        L += ["", f"The board lies on the palm's back, its parts facing out, screwed to the four standoffs "
              f"with M2 x 8 pan heads (its centre at y {y:.0f}, z {z:.0f} mm in the palm's frame, its USB-C "
              f"and 12 V edge facing ({uy:.2f}, {uz:.2f}) in y, z). Fit it last: it covers part of the palm's "
              "open back, where the cables run under it."]
    else:
        L += ["", "The palm's back has no room for the board: it floats on its cables (tie it where it suits)."]
    L += ["",
          "## 7. Calibration", "",
          "The horns' 8 holes every 45 degrees and the 25-tooth spline put each link within about a degree of "
          "its zero pose; set each servo's zero offset so that position 2048 is the finger straight. A "
          "joint's positive direction in the grammar may be either way round on the servo: check each one "
          "with a small move before running the hand.", ""]
    return "\n".join(L)
