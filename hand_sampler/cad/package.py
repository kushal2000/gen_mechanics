"""A design's kit: print-ready STLs, the OpenSCAD source, the parts list,
the assembly steps, the check report, all in one folder and a zip.

    outputs/designs/<id>/
        hand.json                 the genotype
        scad/hand.scad            the parametric source (and the library)
        stl/<part>.stl            each part in its own frame (the viewer's overlay)
        print/<part>.stl          each part laid flat for printing
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

from hand_sampler.grammar.hand import Hand, finger_section, hand_to_json

from . import checks, render, scad
from .design import Design, ServoSpot
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
    out for printing, the assembly pictures, the parts list and the steps,
    the report and the zip."""
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
    for n in names:
        if res[n].ok:
            ext = lay_flat(folder / "stl" / f"{n}.stl", folder / "print" / f"{n}.stl")
            rep.parts[n]["print_extent_mm"] = ext
            if max(ext[:2]) > PRINT_BED_MM:
                rep.warnings.append(f"{n} is {max(ext[:2]):.0f} mm across: larger than a {PRINT_BED_MM:.0f} mm bed")
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


def lay_flat(src: Path, dst: Path) -> List[float]:
    """The part turned onto the stable face that leaves the least overhanging
    area (faces leaning more than 45 degrees out over nothing), on the bed at
    z = 0, centred. Returns its extent (mm)."""
    import trimesh

    m = trimesh.load(str(src), force="mesh")
    try:
        poses, probs = trimesh.poses.compute_stable_poses(m, n_samples=1, threshold=0.0)
    except Exception:
        poses = [np.eye(4)]
    best, best_cost = np.eye(4), math.inf
    for Tp in list(poses)[:12] + [np.eye(4)]:
        mm = m.copy()
        mm.apply_transform(Tp)
        n = mm.face_normals
        z0 = float(mm.bounds[0, 2])
        centres = mm.triangles_center
        down = (n[:, 2] < -math.cos(math.radians(45.0))) & (centres[:, 2] > z0 + 0.3)
        cost = float(mm.area_faces[down].sum())
        if cost < best_cost:
            best, best_cost = Tp, cost
    m.apply_transform(best)
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
    m.export(str(dst))
    return [round(float(x), 1) for x in (hi - lo)]


# --------------------------------------------------------------------------
# the parts list
# --------------------------------------------------------------------------


def _count_screws(d: Design) -> Dict[str, int]:
    out = {"box": 0, "M2x5": 0, "M3x12_column": 0, "M3x8_skirt": 4, "M2x6_board": 4, "M3x8_cradle": 0}
    for sp in d.servos.values():
        out["box"] += 4                           # the case: its far holes, or its idler face's four (a flange cup)
        out["M2x5"] += 4                          # the driven arm (or the crossing plate)
        if sp.yoke == "u":
            out["box"] += 4                       # the idler arm
    for p in d.part_list():
        out["M3x12_column"] += len(getattr(p, "column_screws", []))
        out["M3x8_cradle"] += len(p.pads)
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
             "| Palm Hub rev B.2 board | 1 | `feetech-hand-pcb` |",
             f"| M2 x 5 pan head | {sc['M2x5']} | the driven horn's arm (4 per servo) |",
             f"| M2 x 6 pan head | {sc['M2x6_board']} | the board on its standoffs |",
             f"| M3 x 12 socket head | {sc['M3x12_column']} | through the palm's plate into the mount's column |",
             f"| M3 x 8 pan head | {sc['M3x8_skirt'] + sc['M3x8_cradle']} | the lid's skirt to the bay ("
             f"{sc['M3x8_skirt']})" + (f"; each base servo's cradle to the palm's plate, from behind "
                                       f"({sc['M3x8_cradle']})" if sc['M3x8_cradle'] else "") + " |",
             "| M5 x 12 or M6 x 12 button head, washer | 4 | the bay floor to the arm's flange "
             "(ISO 9409-1-31.5-4-M5 or -50-4-M6) |",
             "| 3-pin 1.25 mm (PicoBlade) servo cable | "
             f"{n_servo + len(d.hand.fingers)} | one per servo, plus one from the board to each finger |",
             "", "## Printed (PETG for the links and the palm; PLA is fine for the bay and the mount)", "",
             "| Part | Grams (about) | Size laid flat (mm) |", "|---|---|---|"]
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
         "`f<finger>_tip`, `palm` (and `palm_beyond` past the cut), `f<finger>_cradle` (a base servo's cradle; "
         "`hinge_cradle` for the hinge), `bay`, `mount`.", "",
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
            joint, port = f"finger {i} joint {j} ({kind}{', turned' if sp.turned else ''})", ports[i]
        arms = {"u": "two (driven + idler)", "single": "one on the driven horn (idler horn off; flange cup)",
                "crossing": "plate on the driven horn"}[sp.yoke]
        held = names[sp.holder or sp.parent]
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
          "4. Driven arm: 4 x M2 x 5. Idler arm: 4 of the servo's box screws.", "",
          "## 3. Palm", "",
          "1. A base servo with a cradle of its own (`f<finger>_cradle`, `hinge_cradle`) is put together on the "
          "bench: the servo into the cradle and its link onto its horns as the table's order says (step 4). "
          "Then stand the cradle in its window in the plate, pads on the plate's front face, and fix it from "
          "behind with 2 x M3 x 8. These go on first, while nothing is behind the plate.",
          "2. Screw the mount's column to the back of `palm` through the plate from the front (M3 x 12), "
          "before any servo the palm itself holds goes in (turn the links of step 1 out of the way).",
          "3. Each servo the palm itself holds goes into its cradle as the table's order says (step 4), and its "
          "link onto its horns (step 2); the cable leaves through the side wall's window.", ""]
    if hand.palm_joints:
        h = d.servos.get(("hinge",))
        where = ("`hinge_cradle` (step 1: `palm_beyond` goes on its horns on the bench)" if h is not None and h.holder
                 else "a cradle of the main palm")
        L += [f"4. The hinge servo sits in {where}; `palm_beyond` is fitted to its horns as in step 2.", ""]
    if steps:
        L += ["## Pictures", "",
              "Each joint: the part holding the servo, the servo (horns, centre screws, case screws in gold) and "
              "the next link pulled off the way it slides on, with its arm screws.", ""]
        for sp in sorted(d.servos.values(), key=servo_id):
            if sp.key in steps:
                L += [f"**Servo {'/'.join(map(str, sp.key))}** (ID {servo_id(sp)}, "
                      f"{(orders or {}).get(sp.key) or 'servo first'}): `{names[sp.holder or sp.parent]}` holds it, "
                      f"`{names[sp.child]}` goes on its horns.", "", f"![servo {sp.key}](steps/{steps[sp.key]})", ""]
    L += ["## 4. Fingers, from the palm out", "",
          "For each joint, in the order the table gives:", "",
          "- **servo first:** slide the servo into its housing from the front (its horns pass through the "
          "slots), push it to the back wall and screw the case to the side plates (4 of the servo's box screws, "
          "through the far holes). Then fit the next link to its horns (step 2).",
          "- **link first:** fit the next link to the servo's horns on the bench (step 2), then slide the "
          "servo, link and all, into its housing from the front and screw the case to the side plates.", "",
          "## 5. Cables", "",
          "Each finger's servos are chained on the finger's back: board port -> base servo -> next servo. Leave "
          "about 25 mm of slack across each joint and tie the cable to the links. The palm's cables run over "
          "its back face into the mount's bore.", "",
          "| Finger | Board port |", "|---|---|"]
    for i in range(len(hand.fingers)):
        L.append(f"| {i} | {ports[i]} |")
    L += ["", "## 6. Board and arm", "",
          "1. Bolt `bay` to the arm's flange through its floor slots (4 button heads with washers: M5 on the "
          "31.5 mm circle, M6 on the 50 mm circle). Check the slot and pin angles against your flange "
          "drawing (`ISO_BOLT_ANGLES`, `ISO_PIN_ANGLE` in `hand.scad`).",
          "2. Screw the board to its standoffs (4 x M2 x 6), USB-C and power toward the wall openings.",
          "3. Thread the finger cables down the mount's bore and plug them in as in the table above.",
          "4. Lower the hand onto the bay, the lid's skirt over the cup's rim, and fix it with 4 x M3 x 8 "
          "through the skirt's short sides.", "",
          "## 7. Calibration", "",
          "The horns' 8 holes every 45 degrees and the 25-tooth spline put each link within about a degree of "
          "its zero pose; set each servo's zero offset so that position 2048 is the finger straight. A "
          "joint's positive direction in the grammar may be either way round on the servo: check each one "
          "with a small move before running the hand.", ""]
    return "\n".join(L)
