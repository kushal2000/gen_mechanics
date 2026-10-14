"""Measure 1000 random Feetech hands (rebuilt grammar, 2026-10-14):

    PYTHONPATH=. python3 project-notes/grammar/experiments/E16_grammar_rebuild/ours/measure_feetech.py 1000 \
        > project-notes/grammar/experiments/E16_grammar_rebuild/ours/feetech_1000.json

Seed 20261014; numbers in feetech_1000.json. The gravity torque is about each
finger's first joint axis at the zero pose with the palm up (gravity along
-x of the palm), from the link boxes' masses (1750 kg/m^3)."""
import json, math, sys, time, collections
import numpy as np
from hand_sampler.grammar import derive as dv, viability as V, operators as ops, motor as mt
from hand_sampler.grammar.hand import FEETECH_RULES, JOINT_EFFORT_NM, PLATE_HALF_MM, parameter_count, KIND_AXIS, finger_section
from hand_sampler.grammar import build_conformed as BC

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rng = np.random.default_rng(20261014)
t0 = time.time()
hands, c2ok = [], []
for _ in range(N):
    h = ops.random_hand(rng, FEETECH_RULES)
    hands.append(h)
gen_s = (time.time() - t0) / N
c1 = [V.c1_self_overlap(h).ok for h in hands]
c2 = [V.c2_workspace_overlap(h).ok for h in hands]
nums = [parameter_count(h)["numbers"] for h in hands]
area = [dv.polygon_area(dv.outline_mm(h)) / 100 for h in hands]
radii = collections.Counter(round(math.hypot(f.y, f.z), 1) for h in hands for f in h.fingers)
nf = collections.Counter(len(h.fingers) for h in hands)
nj = collections.Counter(len(f.joints) for h in hands for f in h.fingers)
cuts = sum(bool(h.palm_joints) for h in hands)
beyond = [sum(finger_section(h, i) == 0 for i in range(len(h.fingers))) for h in hands if h.palm_joints]
links = [jt.length for h in hands for f in h.fingers for jt in f.joints[:-1]]
zeros = sum(L == 0 for L in links) / max(1, len(links))
flen = [sum(j.length for j in f.joints) for h in hands for f in h.fingers]
# protrusion behind the back face: lowest point of every base motor's housing
prot = []
for h in hands:
    low = min(min(b.grown(mt.WALL_MM).corners()[:, 0].min() for b in dv.base_motor_boxes_mm(f)) for f in h.fingers)
    prot.append(max(0.0, -PLATE_HALF_MM - low))
# object start height above the front face (mm), cube half 30 and 36
ostart = [dv.object_start_m(h, 0.03)[0] * 1e3 - PLATE_HALF_MM - 35 for h in hands]
ostart36 = [dv.object_start_m(h, 0.036)[0] * 1e3 - PLATE_HALF_MM - 41 for h in hands]
# finger masses and the gravity torque about the base joint with the finger flat (palm up)
masses, torques = [], []
for h in hands:
    for i, f in enumerate(h.fingers):
        m_tot = 0.0; tq = 0.0
        pose = dv.fk(h)
        for j in range(len(f.joints)):
            m, com, _ = dv.link_mass_props(h, i, j)
            T = pose.links[i][j]
            p = T[:3, :3] @ com + T[:3, 3]
            base = dv.mount_point_m(f)
            r = p - base
            # gravity along -x (palm up); torque about the first joint's axis
            a = pose.links[i][0][:3, :3] @ dv.joint_axis(h, i, 0)
            tq += abs(float(np.cross(r, np.array([-9.81 * m, 0, 0])) @ a))
            m_tot += m
        masses.append(m_tot); torques.append(tq)
recs = BC.load()
carea = [dv.polygon_area(dv.outline_mm(r["hand"])) / 100 for r in recs.values()]
out = {
    "N": N, "seconds_per_hand": round(gen_s, 3), "c1_ok": float(np.mean(c1)), "c2_ok": float(np.mean(c2)),
    "random_hands_per_viable_hand": round(1 / max(np.mean(c2), 1e-9), 2),
    "numbers": {"mean": float(np.mean(nums)), "min": int(min(nums)), "max": int(max(nums))},
    "palm_area_cm2": {"median": float(np.median(area)), "p10": float(np.percentile(area, 10)), "p90": float(np.percentile(area, 90))},
    "commercial_palm_area_cm2_median_same_outline_rule": float(np.median(carea)),
    "base_radii": dict(sorted((str(k), v) for k, v in radii.items())),
    "fingers": dict(sorted(nf.items())), "joints_per_finger": dict(sorted(nj.items())),
    "cut_share": cuts / N, "fingers_beyond_cut": dict(sorted(collections.Counter(beyond).items())),
    "zero_link_share": zeros, "finger_length_mm": {"median": float(np.median(flen)), "max": int(max(flen))},
    "rear_protrusion_mm": {"share_any": float(np.mean(np.array(prot) > 1e-6)), "median": float(np.median(prot)),
                            "max": float(max(prot))},
    "object_raised_mm_half30": {"share_raised": float(np.mean(np.array(ostart) > 1e-6)),
                                "median_raise_when_raised": float(np.median([x for x in ostart if x > 1e-6] or [0])),
                                "max": float(max(ostart))},
    "object_raised_mm_half36": {"share_raised": float(np.mean(np.array(ostart36) > 1e-6)), "max": float(max(ostart36))},
    "finger_mass_g": {"median": float(np.median(masses)) * 1e3, "max": float(max(masses)) * 1e3},
    "flat_finger_gravity_torque_nm": {"median": float(np.median(torques)), "p90": float(np.percentile(torques, 90)),
                                       "max": float(max(torques)), "share_over_0_5": float(np.mean(np.array(torques) > 0.5)),
                                       "effort_nm": JOINT_EFFORT_NM,
                                       "share_over_effort": float(np.mean(np.array(torques) > JOINT_EFFORT_NM))},
}
print(json.dumps(out, indent=1))
