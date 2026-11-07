"""Kit-free: structural features that the joint-transformer policy is sensitive to, for the commercial hands and for a
co-evolution population. Per hand: finger count, largest angular gap between fingers (palm plane), mean finger
length, the first joint's link length per finger (min / mean; a 2-DOF knuckle shows as a short stub), joints per
finger, and how many fingers have a short (< 40 mm) first link.

    python experiments/06nov_coevolve_v2/commercial_coverage.py [--run debug_outputs/coevolve/v2/k5_none --gen 4]
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
src = Path(__file__).with_name("audit") / "population_morph.py"
sys.argv, _a = sys.argv[:1] + ["/dev/null"], sys.argv[1:]
exec(compile(src.read_text().split("m = json.load(open(sys.argv[1]))")[0], str(src), "exec"))
from hand_sampler.design_space import joint_axes_and_link_vectors
from isaacsimenvs.common.robots import get_robot_spec
from isaacsimenvs.common.robots.finger_layout import finger_layout


def features(name):
    s = get_robot_spec(name)
    st = stats(name)
    _, lv = joint_axes_and_link_vectors(s.urdf_path, s.hand_joint_names, s.fingertip_body_names, s.fingertip_offsets)
    L = np.linalg.norm(lv, axis=1) * 1000
    lay = finger_layout(s)
    first = [L[f[0]] for f in lay.fingers if len(f)]
    return dict(n=int(st["n"]), gap=float(st["max_gap"]), len=float(st["len_mm"]), first_min=float(min(first)), first_mean=float(np.mean(first)),
                short=int(sum(x < 40 for x in first)), jpf=float(np.mean([len(f) for f in lay.fingers])))


def row(lbl, f):
    return (f"{lbl:34s} {f['n']:3d} {f['gap']:6.0f} {f['len']:6.0f} {f['first_min']:7.0f} {f['first_mean']:7.0f} "
            f"{f['short']:5d} {f['jpf']:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="debug_outputs/coevolve/v2/k5_none")
    ap.add_argument("--gen", type=int, default=4)
    ap.add_argument("--sample", type=int, default=600)
    a = ap.parse_args(_a)
    hdr = f"{'hand':34s} {'fin':>3s} {'gap':>6s} {'len':>6s} {'1st_min':>7s} {'1st_avg':>7s} {'short':>5s} {'j/f':>5s}"
    print(hdr)
    for h in ("allegro", "leap", "shadow", "dex3", "tesollo", "wuji2", "xhand"):
        try:
            print(row(f"{h}_left_uniform_handonly", features(f"{h}_left_uniform_handonly")))
        except Exception as e:                              # noqa: BLE001
            print(f"{h}: {type(e).__name__} {str(e)[:80]}")
    m = json.loads((REPO / a.run / f"gen_{a.gen:04d}/manifest.json").read_text())
    random.seed(0)
    F = []
    for h in random.sample(m["hands"], min(a.sample, len(m["hands"]))):
        try:
            F.append(features(h["spec"]))
        except Exception:                                    # noqa: BLE001
            pass
    print(f"\n{a.run} gen {a.gen}: {len(F)} sampled designs -- percentiles p5 / p50 / p95")
    for k in ("n", "gap", "len", "first_min", "first_mean", "short", "jpf"):
        v = np.array([f[k] for f in F], float)
        print(f"   {k:10s} {np.percentile(v, 5):7.1f} {np.percentile(v, 50):7.1f} {np.percentile(v, 95):7.1f}   "
              f"frac first_min<40mm: {np.mean([f['first_min'] < 40 for f in F]):.3f}" if k == "first_min" else
              f"   {k:10s} {np.percentile(v, 5):7.1f} {np.percentile(v, 50):7.1f} {np.percentile(v, 95):7.1f}")
    print(f"   designs with gap > 240: {np.mean([f['gap'] > 240 for f in F]):.3f};  with >= 3 short-first-link fingers: "
          f"{np.mean([f['short'] >= 3 for f in F]):.3f}")


if __name__ == "__main__":
    main()
