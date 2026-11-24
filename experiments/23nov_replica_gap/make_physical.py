"""Vendor hands with the grammar's physical build (23 Nov 2026), on top of <hand>_left_canonclip_centred (canonical
signs, grammar-range limits, the grammar's palm centre):
  <hand>_left_mass   every finger link's and the palm's mass / inertia scaled to the exact replica's matching body
                     (vendor finger links weigh ~2x the grammar's, thumbs up to 2.6x; all hands share one set of PD gains)
  <hand>_left_gram   _mass + every finger link a 20 mm box (the boxlinks URDF) + a 20 mm plate palm (the plate URDF)
Registered as <hand>_left_uniform_handonly_<tag>.

    .venv_isaacsim/bin/python experiments/23nov_replica_gap/make_physical.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HANDS = ("wuji2", "tesollo", "xhand", "leap", "dex3", "allegro")
D = REPO / "assets/urdf/unified_dynamics_commercial_hands"


def link_block(txt: str, name: str) -> re.Match:
    m = re.search(r'<link\s+name="%s"\s*>.*?</link>' % re.escape(name), txt, re.S)
    assert m, name
    return m


def scale_inertial(txt: str, link: str, k: float) -> str:
    m = link_block(txt, link)
    blk = m.group(0)
    new = re.sub(r'(<mass\s+value=")([^"]+)(")', lambda a: f'{a.group(1)}{float(a.group(2)) * k:.6g}{a.group(3)}', blk)
    new = re.sub(r'\b(i(?:xx|xy|xz|yy|yz|zz)=")([^"]+)(")', lambda a: f'{a.group(1)}{float(a.group(2)) * k:.6g}{a.group(3)}', new)
    return txt.replace(blk, new)


def swap_collisions(txt: str, src: str, link: str) -> str:
    dst_blk = link_block(txt, link).group(0)
    src_cols = re.findall(r'<collision\b.*?</collision>', link_block(src, link).group(0), re.S)
    stripped = re.sub(r'\s*<collision\b.*?</collision>', '', dst_blk, flags=re.S)
    new = stripped.replace("</link>", "\n    " + "\n    ".join(src_cols) + "\n  </link>")
    return txt.replace(dst_blk, new)


def main() -> None:
    import yourdfpy
    from isaacsimenvs.common.robots import get_robot_spec
    for h in HANDS:
        base = json.loads((D / h / f"{h}_left_canonclip_centred.spec.json").read_text())
        rep = get_robot_spec(f"debug_outputs/coevolve/v9/commercial_exact_axes_canon/{h}_v7/{h}_v7.spec.json")
        ven = get_robot_spec(f"{h}_left_uniform_handonly_canonclip_centred")
        ur = yourdfpy.URDF.load(rep.urdf_path, load_meshes=False, build_collision_scene_graph=False)
        uv = yourdfpy.URDF.load(ven.urdf_path, load_meshes=False, build_collision_scene_graph=False)
        mr = {l.name: l.inertial.mass for l in ur.robot.links if l.inertial is not None}
        mv = {l.name: l.inertial.mass for l in uv.robot.links if l.inertial is not None}
        ratios = {ven.palm_body_name: mr[rep.palm_body_name] / mv[ven.palm_body_name]}
        for bv, br in zip(ven.joint_link_bodies, rep.joint_link_bodies):
            ratios[bv] = mr[br] / mv[bv]
        for tag in ("mass", "gram"):
            if tag == "gram":
                bl = D / h / f"{h}_left_boxlinks_canonclip.urdf"
                pl = D / h / f"{h}_left_plate_canonclip.urdf"
                if not (bl.exists() and pl.exists()):
                    print(f"{h:8s} {tag}: no boxlinks / plate URDF, skipped")
                    continue
                txt = swap_collisions(bl.read_text(), pl.read_text(), ven.palm_body_name)
            else:
                txt = (REPO / base["urdf"]).read_text()
            for link, k in ratios.items():
                txt = scale_inertial(txt, link, k)
            u = D / h / f"{h}_left_{tag}.urdf"
            u.write_text(txt)
            s = dict(base, urdf=str(u.relative_to(REPO)))
            s[f"notes_{tag}"] = (f"23nov_replica_gap/make_physical.py: masses scaled to the exact replica's "
                                 f"({sum(mv.values()) * 1000:.0f} g -> ~{sum(mr.values()) * 1000:.0f} g)"
                                 + ("; 20 mm box links + 20 mm plate palm" if tag == "gram" else ""))
            (D / h / f"{h}_left_{tag}.spec.json").write_text(json.dumps(s, indent=1))
            print(f"{h:8s} {tag}: {s[f'notes_{tag}']}")


if __name__ == "__main__":
    main()
