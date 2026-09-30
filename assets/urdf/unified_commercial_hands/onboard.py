"""Onboard a commercial hand: vendor URDF -> unified URDF + measured spec JSON + report.

What a hand needs to become a RobotSpec, measured from its URDF instead of by hand
(allegro_handonly.py documents every one of these for the first commercial hand):

  unify     re-root at the palm (drop mounts, forearm, wrist) -> <hand>/<hand>_<side>.urdf
  joints    the revolute joints under the palm, per finger proximal -> distal
  home      0 clamped inside each joint's limits (a zero outside the range twitches at reset)
  frame     our palm convention -- x = grasp normal, y = width, z = wrist -> fingertip --
            as a proper rotation in the palm body's frame:
              x = the palm collider's thinnest axis, signed toward where the fingers flex,
              z = the extended non-thumb fingers' direction at home (root -> tip), orthogonal to x,
              y = z cross x   (always a rotation; the width's sign then follows),
            each axis snapped onto a palm-body axis when within SNAP_DEG (recorded in the report)
  palm box  the palm body's collision hull, axis-aligned in our convention (centre + extents)
  pads      each fingertip's pad point: the vendor's fixed tip frame if it has one, else the
            centroid of the distal-most 15% of the fingertip mesh
  filters   self-collision pairs to filter: every jointed pair, plus any pair of bodies whose
            convex hulls overlap at the home pose (they would start every episode in contact)
  gains     uniform stiffness / damping / armature unless the hand overrides (see GAINS)

Everything is written to <hand>/<hand>_<side>.spec.json, which unified_hands.py turns into a RobotSpec,
and a human-readable <hand>/<hand>_<side>.report.md with the checks. Kit-free (yourdfpy + trimesh).

    .venv_isaacsim/bin/python assets/urdf/unified_commercial_hands/onboard.py <hand> [...]
"""
from __future__ import annotations

import itertools, json, math, sys, xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# Per hand: which body is the palm and which bodies are fingertips (last MOVING link of each finger,
# since merge_fixed_joints folds tip frames into them), and how to tell the thumb.
HANDS = {
    "allegro_right": dict(vendor="allegro/allegro.urdf", palm="palm_link",
                          tips=["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"], thumb="thumb"),
    "allegro": dict(vendor="allegro/vendor_left/allegro_left.urdf", palm="base_link",
                    tips=["link_3.0", "link_7.0", "link_11.0", "link_15.0"], thumb="link_1[2-5]"),
    "leap": dict(vendor="leap/vendor_left/leap_left.urdf", palm="palm_lower_left",
                 tips=["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"], thumb="thumb"),
    "shadow": dict(vendor="shadow/vendor_left/shadow_left.urdf", palm="palm",
                   tips=["ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal"], thumb="^th"),
    "dex3": dict(vendor="dex3/vendor_left/dex3_left.urdf", palm="left_hand_palm_link",
                 tips=["left_hand_thumb_2_link", "left_hand_middle_1_link", "left_hand_index_1_link"], thumb="thumb"),
    "tesollo": dict(vendor="tesollo/vendor_left/tesollo_left.urdf", palm="ll_dg_palm",
                    tips=[f"ll_dg_{k}_4" for k in range(1, 6)], thumb="ll_dg_1_"),
    "wuji2": dict(vendor="wuji2/vendor_left/wuji2_left.urdf", palm="l_wrist",
                  tips=["l_thumb_distal", "l_index_finger_distal", "l_middle_finger_distal",
                        "l_ring_finger_distal", "l_pinky_distal"], thumb="thumb"),
    "xhand": dict(vendor="xhand/vendor_left/xhand_left.urdf", palm="left_hand_link",
                  tips=["left_hand_thumb_rota_link2", "left_hand_index_rota_link2", "left_hand_mid_link2",
                        "left_hand_ring_link2", "left_hand_pinky_link2"], thumb="thumb"),
}
# Default drive gains: Isaac Lab's Allegro (stiffness 3.0, damping 0.1) and our armature 0.001. A hand
# whose vendor or Isaac Lab publishes its own goes here.
SNAP_DEG = 10.0
# Palms that are genuinely thicker than 60 mm, after checking the collider has no stray piece.
THICK_PALM_OK = {
    "tesollo": "one continuous 64 mm collider (DG-5F carries actuators in the palm); the only other pieces "
               "are a 3-vertex sliver and a 1 mm plate inside that span",
}
GAINS_DEFAULT = dict(stiffness=3.0, damping=0.1, armature=0.001)
GAINS = {}

import re

def usd_name(name: str) -> str:
    """A valid USD prim name: [A-Za-z_][A-Za-z0-9_]*. Isaac's URDF importer renames anything else on its
    own (Allegro's 'link_0.0' became 'link_0_0') or, for bare-number joint names (LEAP's '0', '1', ...),
    fails to build the articulation at all -- LEAP came out as ONE rigid body. Renaming here keeps the
    unified URDF, the spec and the converted USD in agreement."""
    out = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return out if re.match(r"[A-Za-z_]", out) else f"j{out}"


def unify(hand: str, cfg: dict, side: str) -> Path:
    """Re-root the vendor URDF at the palm; keep only the palm's subtree. Mesh paths stay valid."""
    src = HERE / cfg["vendor"]
    if "vendor_left" not in cfg["vendor"]:          # an already-unified hand (allegro right): use as is
        return src
    root = ET.parse(src).getroot()
    joints = root.findall("joint"); links = {l.get("name"): l for l in root.findall("link")}
    kids = {}
    for j in joints:
        kids.setdefault(j.find("parent").get("link"), []).append(j)
    keep_links, keep_joints, stack = {cfg["palm"]}, [], [cfg["palm"]]
    while stack:
        for j in kids.get(stack.pop(), []):
            keep_joints.append(j); c = j.find("child").get("link"); keep_links.add(c); stack.append(c)
    out = ET.Element("robot", root.attrib)
    out.append(ET.Comment(f" GENERATED by onboard.py from {cfg['vendor']}: re-rooted at {cfg['palm']} "
                          "(ancestors: mounts, forearm and wrist, dropped). Do not hand-edit. "))
    for name in [l.get("name") for l in root.findall("link")]:
        if name in keep_links:
            out.append(links[name])
    for j in joints:
        if j in keep_joints:
            out.append(j)
    renamed = {}
    for el in list(out.iter("link")) + list(out.iter("joint")):
        n = el.get("name"); u = usd_name(n)
        if u != n:
            renamed[n] = u; el.set("name", u)
    for tag in ("parent", "child"):
        for el in out.iter(tag):
            el.set("link", usd_name(el.get("link")))
    for el in out.iter("mimic"):
        el.set("joint", usd_name(el.get("joint")))
    if renamed:
        out.insert(1, ET.Comment(" renamed to valid USD identifiers: " + ", ".join(f"{a} to {b}" for a, b in renamed.items()) + " "))
    for m in out.iter("mesh"):                       # meshes live in vendor_left/meshes
        m.set("filename", "vendor_left/" + m.get("filename"))
    dst = HERE / hand / f"{hand}_{side}.urdf"
    ET.ElementTree(out).write(dst, xml_declaration=True, encoding="utf-8")
    return dst

def body_meshes(robot, body: str, merged: dict) -> list[trimesh.Trimesh]:
    """Collision meshes of a post-merge body (its own + fixed descendants'), in the BODY's frame, as hulls."""
    out = []
    for link in merged[body]:
        T_link = robot.get_transform(link, body)
        for c in robot.link_map[link].collisions:
            g = c.geometry; T0 = c.origin if c.origin is not None else np.eye(4)
            if g.mesh is not None:
                m = trimesh.load(robot._filename_handler(g.mesh.filename), force="mesh")
                if g.mesh.scale is not None:
                    m.apply_scale(g.mesh.scale)
            elif g.box is not None:
                m = trimesh.creation.box(extents=g.box.size)
            elif g.cylinder is not None:
                m = trimesh.creation.cylinder(radius=g.cylinder.radius, height=g.cylinder.length)
            elif g.sphere is not None:
                m = trimesh.creation.icosphere(radius=g.sphere.radius)
            else:
                continue
            m = m.convex_hull; m.apply_transform(T_link @ T0); out.append(m)
    return out

def measure(hand: str, cfg: dict, urdf: Path, side: str) -> dict:
    robot = yourdfpy.URDF.load(str(urdf), build_collision_scene_graph=True, load_collision_meshes=True)
    unified_names = "vendor_left" in cfg["vendor"]
    nm = usd_name if unified_names else (lambda x: x)
    palm = nm(cfg["palm"]); tips = [nm(t) for t in cfg["tips"]]; thumb = re.compile(cfg["thumb"])
    # post-merge bodies: a link is its own body unless its parent joint is fixed
    parent_joint = {j.child: j for j in robot.joint_map.values()}
    def body_of(link):
        while link in parent_joint and parent_joint[link].type == "fixed":
            link = parent_joint[link].parent
        return link
    merged = {}
    for l in robot.link_map:
        merged.setdefault(body_of(l), []).append(l)
    bodies = [b for b in merged if b == palm or b in parent_joint]
    # joints, finger by finger in URDF order, proximal -> distal
    order = []
    def dfs(link):
        for j in robot.joint_map.values():
            if j.parent == link:
                if j.type in ("revolute", "continuous"):
                    order.append(j.name)
                dfs(j.child)
    dfs(palm)
    lim = {n: (robot.joint_map[n].limit.lower, robot.joint_map[n].limit.upper) for n in order}
    margin = 1e-3
    home = {n: float(min(max(0.0, lo + margin), hi - margin)) for n, (lo, hi) in lim.items()}
    q_home = [home[n] for n in order]
    def fk(q):
        robot.update_cfg(dict(zip(order, q)))
        return {b: robot.get_transform(b, palm) for b in bodies}
    P0 = fk(q_home)
    # which tip belongs to the thumb: its chain has a link or joint matching the thumb pattern
    def chain(b):
        out = [b]
        while out[-1] in parent_joint and out[-1] != palm:
            j = parent_joint[out[-1]]; out.append(j.parent)
            out.append(j.name)
        return out
    is_thumb = {t: any(thumb.search(x) for x in chain(t)) for t in tips}
    assert sum(is_thumb.values()) == 1, f"{hand}: thumb match {is_thumb}"
    fingers = [t for t in tips if not is_thumb[t]]
    # finger roots: the first moving joint of each finger chain (its child's origin)
    def root_of(t):
        b = t
        while parent_joint[b].parent != palm and body_of(parent_joint[b].parent) != palm:
            b = body_of(parent_joint[b].parent)
        return b
    palm_hulls = body_meshes(robot, palm, merged)
    palm_pts = np.vstack([m.vertices for m in palm_hulls])
    centre_guess = palm_pts.mean(0)
    roots = np.array([P0[root_of(t)][:3, 3] for t in fingers])
    # grasp normal = the palm slab's THINNEST axis (oriented bounding box of the palm collider), signed by
    # which way the non-thumb fingertips move when the fingers flex. (Averaging the flex displacement itself
    # is biased: a flexing fingertip sweeps an arc, and its mean motion leans back toward the palm -- that
    # tilted Allegro's frame by ~15 deg in validation.)
    obb = trimesh.util.concatenate(palm_hulls).convex_hull.bounding_box_oriented
    R_obb = obb.primitive.transform[:3, :3]; ext_obb = obb.primitive.extents
    x = R_obb[:, int(np.argmin(ext_obb))].copy()
    q_flex = list(q_home)
    for i, n in enumerate(order):
        lo, hi = lim[n]; h = home[n]
        q_flex[i] = h + 0.5 * ((hi - h) if (hi - h) >= (h - lo) else (lo - h))
    P1 = fk(q_flex)
    disp = np.mean([P1[t][:3, 3] - P0[t][:3, 3] for t in fingers], 0)
    if disp @ x < 0:
        x = -x
    # fingers = the extended non-thumb fingers' own direction at home (root -> tip), made orthogonal to the
    # normal. (Palm centre -> finger roots was biased on palms whose hull bulges toward the thumb: LEAP,
    # Shadow and Tesollo came out 13-21 deg off any palm axis.)
    zc = np.mean([P0[t][:3, 3] - P0[root_of(t)][:3, 3] for t in fingers], 0)
    z = zc - (zc @ x) * x; z /= np.linalg.norm(z)
    # Vendor palm frames are nearly always aligned with the palm: snap an axis onto a palm-body axis when
    # within SNAP_DEG of it, and record that it was snapped.
    snapped = []
    def snap(v, tag):
        k = int(np.argmax(np.abs(v))); e = np.zeros(3); e[k] = np.sign(v[k])
        if math.degrees(math.acos(min(1.0, abs(float(v @ e))))) < SNAP_DEG:
            snapped.append(f"{tag}->{'+-'[int(e[k] < 0)]}{'xyz'[k]}"); return e
        return v
    x = snap(x, "x"); z = snap(z - (z @ x) * x, "z"); z /= np.linalg.norm(z)
    y = np.cross(z, x)
    F = np.stack([x, y, z], 1)                   # columns: our axes in the palm body's frame
    # palm box in our convention
    v_ours = palm_pts @ F
    lo_b, hi_b = v_ours.min(0), v_ours.max(0)
    centre_ours = (lo_b + hi_b) / 2; ext_ours = hi_b - lo_b
    centre_palm = F @ centre_ours
    # pads
    pads = []
    for t in tips:
        fixed_kids = [l for l in merged[t] if l != t]
        tipframe = [l for l in fixed_kids if "tip" in l.lower()]
        if tipframe:
            pads.append(robot.get_transform(tipframe[0], t)[:3, 3].tolist()); continue
        pts = np.vstack([m.vertices for m in body_meshes(robot, t, merged)])
        axis = pts.mean(0); axis = axis / (np.linalg.norm(axis) + 1e-12)
        s = pts @ axis; far = pts[s >= np.quantile(s, 0.85)]
        pads.append(far.mean(0).tolist())
    # self-collision: jointed pairs + pairs overlapping at home
    hulls = {b: body_meshes(robot, b, merged) for b in bodies}
    def posed(b, P):
        return [m.copy().apply_transform(P[b]) for m in hulls[b]]
    def overlap(a, b, P):
        best = 0.0
        for ma in posed(a, P):
            for mb in posed(b, P):
                if (ma.bounds[1] < mb.bounds[0]).any() or (mb.bounds[1] < ma.bounds[0]).any():
                    continue
                pts = np.vstack([ma.vertices, ma.sample(200)])
                best = max(best, float(trimesh.proximity.signed_distance(mb, pts).max()))
        return best
    jointed = {frozenset((body_of(parent_joint[b].parent), b)) for b in bodies if b in parent_joint}
    at_home = {}
    for a, b in itertools.combinations(bodies, 2):
        if frozenset((a, b)) in jointed:
            continue
        d = max(overlap(a, b, P0), overlap(b, a, P0))
        if d > 5e-4:
            at_home[(a, b)] = d
    filt = {frozenset(p) for p in jointed} | {frozenset(p) for p in at_home}
    adj = {}
    for p in filt:
        a, b = sorted(p); adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    tip_pos_ours = {t: (F.T @ (P0[t][:3, 3] - centre_palm)).round(4).tolist() for t in tips}
    g = GAINS.get(hand, GAINS_DEFAULT)
    return dict(
        hand=hand, side=side, urdf=str(urdf.relative_to(REPO)), palm_body=palm,
        hand_joint_names=order, joint_limits={n: list(lim[n]) for n in order},
        hand_default_joint_pos=home, fingertip_body_names=tips, fingertip_offsets=pads,
        thumb_tip=[t for t in tips if is_thumb[t]][0],
        palm_frame=F.tolist(), palm_frame_det=float(np.linalg.det(F)), palm_frame_snapped=snapped,
        palm_centre_in_palm_body=centre_palm.tolist(), palm_extents_ours=ext_ours.tolist(),
        palm_centre_ours=centre_ours.tolist(),
        adjacent_links={k: sorted(v) for k, v in sorted(adj.items())},
        overlap_at_home_mm={f"{a} x {b}": round(d * 1000, 2) for (a, b), d in sorted(at_home.items(), key=lambda kv: -kv[1])},
        fingertip_pos_home_ours=tip_pos_ours,
        stiffness=g["stiffness"], damping=g["damping"], armature=g["armature"],
    )

def report(s: dict) -> str:
    F = np.array(s["palm_frame"])
    tips = s["fingertip_pos_home_ours"]
    rows = "\n".join(f"| {t} | {' thumb' if t == s['thumb_tip'] else ''} | {p[0]:+.3f} | {p[1]:+.3f} | {p[2]:+.3f} |"
                     for t, p in tips.items())
    ext = s["palm_extents_ours"]
    checks = [
        ("palm frame is a proper rotation (det +1)", abs(s["palm_frame_det"] - 1) < 1e-6),
        ("palm thickness < width and < length (a slab)", ext[0] < ext[1] and ext[0] < ext[2]),
        ("palm thickness < 60 mm (no stray collider geometry)"
         + (f" -- WAIVED: {THICK_PALM_OK[s['hand']]}" if s['hand'] in THICK_PALM_OK else ""),
         ext[0] < 0.06 or s['hand'] in THICK_PALM_OK),
        ("every non-thumb fingertip is beyond the palm centre along z", all(p[2] > 0 for t, p in tips.items() if t != s["thumb_tip"])),
        ("home pose inside every limit", all(lo < s["hand_default_joint_pos"][n] < hi for n, (lo, hi) in s["joint_limits"].items())),
    ]
    ck = "\n".join(f"- {'PASS' if ok else '**FAIL**'} {name}" for name, ok in checks)
    ov = "\n".join(f"| {k} | {v} |" for k, v in s["overlap_at_home_mm"].items()) or "| (none) | |"
    return f"""# {s['hand']} {s['side']} — onboarding report (generated by onboard.py)

URDF `{s['urdf']}`, palm body `{s['palm_body']}`, {len(s['hand_joint_names'])} joints, fingertips {s['fingertip_body_names']}.

## Checks
{ck}

## Palm frame (columns = our x grasp-normal, y width, z fingers, in `{s['palm_body']}`'s frame)
```
{np.array2string(F, precision=3, suppress_small=True)}
```
Axes snapped to palm-body axes: {s['palm_frame_snapped'] or 'none'}.
Palm box in our convention: extents {np.round(ext, 4).tolist()} m (thickness, width, length), centre in
palm body {np.round(s['palm_centre_in_palm_body'], 4).tolist()}.

## Fingertips at home, from the palm centre, in our convention (m)
| body | | x (normal) | y (width) | z (fingers) |
|---|---|---|---|---|
{rows}

## Bodies overlapping at the home pose (filtered)
| pair | overlap mm |
|---|---|
{ov}

Gains: stiffness {s['stiffness']}, damping {s['damping']}, armature {s['armature']} (uniform).
"""

if __name__ == "__main__":
    for hand in sys.argv[1:]:
        cfg = HANDS[hand]; side = "right" if hand.endswith("_right") else "left"
        name = hand.replace("_right", "")
        urdf = unify(name, cfg, side)
        spec = measure(hand, cfg, urdf, side)
        out = HERE / name / f"{name}_{side}"
        out.with_suffix(".spec.json").write_text(json.dumps(spec, indent=1))
        out.with_suffix(".report.md").write_text(report(spec))
        print(f"{hand}: {len(spec['hand_joint_names'])} joints, det {spec['palm_frame_det']:+.3f}, "
              f"palm extents {np.round(spec['palm_extents_ours'], 3).tolist()}, "
              f"{len(spec['overlap_at_home_mm'])} overlapping pairs at home -> {out.with_suffix('.spec.json').relative_to(REPO)}")
