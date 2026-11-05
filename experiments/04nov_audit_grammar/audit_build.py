"""Audit the 29 Oct evolution runs' built hands: does each hand's spec and URDF say what its design says?

    .venv_isaacsim/bin/python experiments/04nov_audit_grammar/audit_build.py [--runs map_elites baseline] [--rebuild 60]

Kit-free. Per hand in the run's archive (its stored hands/<id>/ files):
  design     the spec's embedded design equals the archive's design
  joints     the spec's hand joints are f<i>_j<j>, exactly the design's fingers and joint counts (+ palm0_joint iff cut)
  limits     each joint's limits are its kind's (flexion -30..90, abduction +-30, roll +-90 degrees; the cut +-30)
  axes       each URDF joint's axis is its kind's axis in the link frame (flexion z, abduction y, roll x), up to sign
  tips       one fingertip body per finger, the last link of that finger
  urdf       the URDF's revolute joints are the spec's joints, every one with effort/velocity as the spec says
And --rebuild N hands (spread over the run) are built again from their designs into a temp dir: the URDF must come out
byte-identical (paths aside) and the spec equal. Writes experiments/04nov_audit_grammar/results/<run>_build.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "results"
KIND = {(0, 0): "flexion", (90, 0): "abduction", (0, 90): "roll"}
LIMITS_DEG = {"flexion": (-30, 90), "abduction": (-30, 30), "roll": (-90, 90), "palm": (-30, 30)}
AXIS = {"flexion": 2, "abduction": 1, "roll": 0}       # the link-frame axis index (x, y, z) per kind


def _check(rec: dict) -> list:
    spec_path = REPO / rec["spec"]
    spec = json.loads(spec_path.read_text())
    d = rec["design"]
    bad = []
    sd = spec["design"] if isinstance(spec["design"], dict) else json.loads(spec["design"])
    if sd != d:
        bad.append(("design", None))
    want = [f"f{i}_j{j}" for i, f in enumerate(d["fingers"]) for j in range(len(f["joints"]))]
    have = list(spec["hand_joint_names"])
    cut = bool(d.get("palm_joints"))
    if [n for n in have if n != "palm0_joint"] != want or (("palm0_joint" in have) != cut):
        bad.append(("joints", {"want": want + (["palm0_joint"] if cut else []), "have": have}))
    lim = spec["joint_limits"]
    for i, f in enumerate(d["fingers"]):
        for j, jt in enumerate(f["joints"]):
            k = KIND.get(tuple(jt["axis"]))
            lo, hi = (math.degrees(v) for v in lim.get(f"f{i}_j{j}", (math.nan, math.nan)))
            if k is None or abs(lo - LIMITS_DEG[k][0]) > 1e-6 or abs(hi - LIMITS_DEG[k][1]) > 1e-6:
                bad.append(("limits", (f"f{i}_j{j}", k, round(lo, 3), round(hi, 3))))
    if cut:
        lo, hi = (math.degrees(v) for v in lim["palm0_joint"])
        if abs(lo + 30) > 1e-6 or abs(hi - 30) > 1e-6:
            bad.append(("limits", ("palm0_joint", "palm", lo, hi)))
    tips_want = [f"f{i}_link{len(f['joints']) - 1}" for i, f in enumerate(d["fingers"])]
    if list(spec["fingertip_body_names"]) != tips_want:
        bad.append(("tips", {"want": tips_want, "have": spec["fingertip_body_names"]}))
    root = ET.parse(REPO / spec["urdf"]).getroot()
    joints = {j.get("name"): j for j in root.findall("joint") if j.get("type") == "revolute"}
    if sorted(joints) != sorted(have):
        bad.append(("urdf", {"urdf_revolute": sorted(joints), "spec": sorted(have)}))
    for name, j in joints.items():
        l = j.find("limit")
        if float(l.get("effort")) != float(spec["effort_limit"]) or float(l.get("velocity")) != float(spec["velocity_limit"]):
            bad.append(("urdf_limit", name))
        if name.startswith("f"):
            i, jj = (int(x) for x in name[1:].split("_j"))
            k = KIND.get(tuple(d["fingers"][i]["joints"][jj]["axis"]))
            ax = [float(v) for v in j.find("axis").get("xyz").split()]
            if k is not None and not (abs(abs(ax[AXIS[k]]) - 1.0) < 1e-6 and sum(abs(v) for v in ax) < 1.0 + 1e-6):
                bad.append(("axes", (name, k, ax)))
    return bad


def _rebuild(rec: dict) -> list:
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar.urdf_export import build
    spec = json.loads((REPO / rec["spec"]).read_text())
    with tempfile.TemporaryDirectory() as tmp:
        row = build(gh.hand_from_dict(rec["design"]), rec["id"], Path(tmp) / rec["id"])
        new_spec = json.loads(Path(row["spec"]).read_text()) if Path(row["spec"]).is_absolute() else \
            json.loads((REPO / row["spec"]).read_text())
        new_urdf = Path(new_spec["urdf"]) if Path(new_spec["urdf"]).is_absolute() else REPO / new_spec["urdf"]
        a = (REPO / spec["urdf"]).read_text().replace(str(Path(spec["urdf"]).parent), "<dir>")
        b = new_urdf.read_text().replace(str(new_urdf.parent), "<dir>").replace(
            str(new_urdf.parent.relative_to(REPO)) if new_urdf.is_relative_to(REPO) else "\0", "<dir>")
        out = []
        if a != b:
            out.append(("rebuild_urdf", None))
        skip = {"urdf", "hand"}
        diff = sorted(k for k in set(spec) | set(new_spec) if k not in skip and spec.get(k) != new_spec.get(k))
        if diff:
            out.append(("rebuild_spec", diff))
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["map_elites", "baseline"])
    ap.add_argument("--rebuild", type=int, default=60)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        recs = json.loads((REPO / "debug_outputs/evolution" / run / "archive.json").read_text())["hands"]
        fails = {}
        for r in recs:
            b = _check(r)
            if b:
                fails[r["id"]] = b
        step = max(1, len(recs) // max(1, args.rebuild))
        rebuilt = 0
        for r in recs[::step][:args.rebuild]:
            b = _rebuild(r)
            rebuilt += 1
            if b:
                fails.setdefault(r["id"], []).extend(b)
        kinds = collections.Counter(k for v in fails.values() for k, _ in v)
        (OUT / f"{run}_build.json").write_text(json.dumps({"run": run, "hands": len(recs), "rebuilt": rebuilt,
                                                           "failures": dict(kinds), "failed": fails}, indent=1,
                                                          default=str))
        print(f"[build] {run}: {len(recs)} hands checked, {rebuilt} rebuilt; failures {dict(kinds) or 'none'}",
              flush=True)
        for hid, v in list(fails.items())[:5]:
            print(f"    {hid}: {v[:3]}", flush=True)


if __name__ == "__main__":
    main()
