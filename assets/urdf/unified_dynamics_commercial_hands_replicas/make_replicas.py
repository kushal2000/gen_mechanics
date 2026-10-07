"""Commercial hands rebuilt in the hand grammar: one replica per vendor hand that hand_sampler.commercial fits.

For each hand in commercial.HANDS (LEAP, Wuji v2, MIDAS, Allegro), commercial.fit(name) reads the vendor URDF and
returns the closest legal grammar design plus notes on every way it is not the vendor's. Written per hand:

  <hand>/<stem>_replica.json      one-design population: what TRAINING reads, as ROBOT_SPEC=handonly:<abs path>
  <hand>/<stem>_replica.urdf      hand_sampler.build.urdf_for_viewing: the viewer's URDF, NOT for simulation
  <hand>/<stem>_replica_palm.obj  its palm mesh (referenced relatively from the URDF)
  <hand>/<stem>_replica.notes.md  the fit's notes
  <hand>/<stem>_replica.joint_map.json  vendor joint name -> grammar slot (f<finger>_j<depth>)

<stem> is the vendor URDF's (leap_left, wuji2_left, midas, allegro_left). Dynamics are the uniform actuator:
generated joints take hand_sampler.robot_param_constants.gen_joint_drive(), which since 2026-10-07 is the
make_uniform.py actuator every commercial hand trains with (0.5 N.m, 5 rad/s, kp 3.0, kd 0.0775, armature
0.00058) -- hence "unified_dynamics". Links are the grammar's capsules, not the vendor's meshes or masses.

    .venv_isaacsim/bin/python assets/urdf/unified_dynamics_commercial_hands_replicas/make_replicas.py [hand ...]
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from hand_sampler import build, commercial, population_io, validate_design  # noqa: E402
from hand_sampler import design_space as D  # noqa: E402
from hand_sampler import robot_param_constants as rpc  # noqa: E402


def make(name: str) -> Path:
    hand, notes = commercial.fit(name)
    problems = validate_design.check(hand)
    if problems:
        raise SystemExit(f"{name}: not a legal design:\n  " + "\n  ".join(problems))
    vendor = commercial.urdf_of(name)
    stem = f"{vendor.stem}_replica"
    out = HERE / name
    pop = population_io.save_population(
        [hand], out / f"{stem}.json", name=stem,
        provenance=population_io.provenance(method="commercial.fit", source="hand_sampler.commercial",
                                            vendor_urdf=str(vendor.relative_to(REPO)), notes=notes,
                                            gen_joint_drive=list(rpc.gen_joint_drive())))
    urdf = build.urdf_for_viewing(hand, out / f"{stem}.urdf")
    # urdf_for_viewing names its palm mesh by absolute path (it is written for a scratch directory); a tracked
    # asset should not carry this checkout's path, and the viewers resolve a bare name against the URDF's folder.
    text = urdf.read_text()
    text = re.sub(r'filename="[^"]*/(' + re.escape(stem) + r'_palm\.obj)"', r'filename="\1"', text)
    urdf.write_text(text)
    build.write_palm_mesh(hand, out / f"{stem}_palm.obj")
    # Which grammar slot replicates which vendor joint: fit() builds finger i from (row + [thumb])[i] of the
    # straightened digits, segment j from that digit's j-th joint. Checked against the fitted hand's joint counts.
    row, thumb = commercial._split([commercial._straighten(d) for d in commercial.digits(name)])
    order = row + [thumb]
    assert [len(d.joints) for d in order] == [len(f.segments) for f in hand.fingers], name
    joint_map = {j: f"f{i}_j{k}" for i, d in enumerate(order) for k, j in enumerate(d.joints)}
    (out / f"{stem}.joint_map.json").write_text(json.dumps(joint_map, indent=1) + "\n")
    extents = tuple(round(v * 1000) for v in D.palm_extents(hand))
    (out / f"{stem}.notes.md").write_text(
        f"# {stem}\n\n`hand_sampler.commercial.fit('{name}')` of `{vendor.relative_to(REPO)}`: "
        f"{hand.n_fingers} fingers, {hand.n_joints} joints, palm {extents[0]} x {extents[1]} x {extents[2]} mm.\n\n"
        "How it differs from the vendor's hand:\n\n" + "".join(f"* {n}\n" for n in notes))
    print(f"{name:8s} {hand.n_fingers} fingers {hand.n_joints} joints palm {extents} mm -> {pop.relative_to(REPO)}")
    return pop


if __name__ == "__main__":
    for name in sys.argv[1:] or commercial.HANDS:
        make(name)
