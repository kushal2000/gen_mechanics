"""How many mutations separate each design from gen-SHARPA?

A LOWER BOUND on the number of grammar mutations that would turn a design into
`hand_sampler.sharpa_capsule`, computed from what each operator can do in one
step:

  perturb_length   moves EVERY link by one 5 mm quantum, each independently up
                   or down -> fixing all lengths costs max_link |dL| / 5 mm,
                   not the sum
  perturb_axis     the same for every joint's theta, 15 degree quantum
  perturb_offset   the same for every joint's rest offset, 15 degrees
  move_mount       ONE finger's mount, up to 5 mm in u AND 5 mm in v at once
                   (so Chebyshev on the face, in metres) -> summed over fingers
  perturb_palm     ONE palm dimension per step, 10 mm -> summed over dimensions
  split_link /     one joint added to / removed from one finger
  merge_links
  add_finger /     a new finger arrives with ONE joint, and only a one-joint
  remove_finger    finger can be removed, so an n-joint finger costs n either way

Fingers are matched by brute force over assignments (a design has at most five,
gen-SHARPA has five) and segments are aligned from the mount outward. The
result is a lower bound because the whole-hand operators must also carry the
links that topology edits create, and because a mount crossing a face edge
travels further than the straight line between the two positions.

PALM THICKNESS IS NOT MUTABLE. A design seeded at 20 mm can never become
gen-SHARPA's 25 mm slab, so reachability is reported separately from distance.

    .venv_isaacsim/bin/python experiments/17sep_coevolution/analysis/edit_distance.py [label ...]
"""
import itertools, json, math, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hand_sampler import design_space as ds, population_io
from hand_sampler.sharpa_capsule import sharpa_capsule

LINK_Q, ANG_Q, PALM_STEP, MOUNT_STEP = ds.LINK_QUANTUM, ds.ANGLE_QUANTUM, ds.PALM_STEP, 0.005


def _steps(delta: float, quantum: float) -> int:
    return int(round(abs(delta) / quantum))


def _finger_pair_cost(fa, fb, palm_a, palm_b):
    """``(topology, mount, max theta steps, max length steps, max offset steps)``
    for putting finger ``fa`` where ``fb`` is."""
    na, nb = fa.n_joints, fb.n_joints
    topo = abs(na - nb)                                   # split_link / merge_links
    n = min(na, nb)
    th = max((_steps(fa.segments[i].joint.theta - fb.segments[i].joint.theta, ANG_Q) for i in range(n)), default=0)
    ln = max((_steps(fa.segments[i].length - fb.segments[i].length, LINK_Q) for i in range(n)), default=0)
    of = max((_steps(fa.segments[i].joint.offset - fb.segments[i].joint.offset, ANG_Q) for i in range(n)), default=0)
    # the mount, in metres on the palm surface
    if fa.mount.face == fb.mount.face:
        _, _, _, _, span_u, span_v = ds.face_frame(fa.mount.face, palm_b)
        du = abs(fa.mount.u - fb.mount.u) * span_u; dv = abs(fa.mount.v - fb.mount.v) * span_v
        mount = math.ceil(max(du, dv) / MOUNT_STEP - 1e-9)
    else:   # a straight line under-counts a path that has to cross an edge
        pa = ds.mount_position(fa.mount, palm_a); pb = ds.mount_position(fb.mount, palm_b)
        mount = math.ceil(float(np.linalg.norm(pa - pb)) / MOUNT_STEP - 1e-9)
    return topo, mount, th, ln, of


def distance(hand, target=None) -> dict:
    """Lower bound on mutations from ``hand`` to ``target`` (gen-SHARPA), and
    the terms it is made of."""
    B = target if target is not None else sharpa_capsule()
    na, nb = hand.n_fingers, B.n_fingers
    # a finger present in one hand and not the other costs its joint count
    pair = [[_finger_pair_cost(fa, fb, hand.palm, B.palm) for fb in B.fingers] for fa in hand.fingers]
    best = None
    for assign in itertools.permutations(range(nb), min(na, nb)):
        used_b = set(assign)
        topo = mount = 0; th = ln = of = 0
        for ia, ib in enumerate(assign):
            t, m, a, l, o = pair[ia][ib]
            topo += t; mount += m; th = max(th, a); ln = max(ln, l); of = max(of, o)
        for ia in range(min(na, nb), na):                 # extra fingers on the design: merge down, remove
            topo += hand.fingers[ia].n_joints
        for ib in range(nb):                              # fingers the design lacks: add, then split up
            if ib not in used_b:
                topo += B.fingers[ib].n_joints
        total = topo + mount + th + ln + of
        if best is None or total < best["total"]:
            best = dict(total=total, topology=topo, mount=mount, theta=th, length=ln, offset=of,
                        assignment=assign)
    palm = _steps(hand.palm.width - B.palm.width, PALM_STEP) + _steps(hand.palm.length - B.palm.length, PALM_STEP)
    best["palm"] = palm; best["total"] += palm
    best["thickness_ok"] = abs(hand.palm.thickness - B.palm.thickness) < 1e-9
    return best


def main() -> None:
    labels = sys.argv[1:] or ["coevolution_v2_gen2k"]
    S = sharpa_capsule()
    print(f"target: gen-SHARPA, {S.n_fingers} fingers, {S.n_joints} joints, "
          f"palm {tuple(round(v*1000) for v in S.palm.extents)} mm\n")
    for label in labels:
        P = pathlib.Path("assets/populations") / label
        gens = sorted(int(p.name.split("_")[1]) for p in P.glob("gen_*") if (p / "population.json").exists())
        print(f"=== {label}")
        print(" gen | closest design      | edits | topo mount theta len off palm | reachable (25 mm palm) | population median")
        rows = []
        for g in gens:
            hands = population_io.load_population(P / f"gen_{g}/population.json")
            ds_ = [distance(h, S) for h in hands]
            tot = np.array([d["total"] for d in ds_])
            ok = np.array([d["thickness_ok"] for d in ds_])
            # the best design that could actually get there, and the best overall
            i_ok = int(np.where(ok, tot, 10**6).argmin()) if ok.any() else None
            i = int(tot.argmin()); d = ds_[i]
            rows.append(dict(gen=g, best=i, total=int(tot[i]), median=float(np.median(tot)),
                             reachable=int(ok.sum()), best_reachable=None if i_ok is None else int(tot[i_ok]),
                             **{k: int(d[k]) for k in ("topology", "mount", "theta", "length", "offset", "palm")}))
            print(f" {g:3d} | #{i:<4d} {hands[i].n_fingers}f {hands[i].n_joints:2d}j      | {tot[i]:5.0f} | "
                  f"{d['topology']:4d} {d['mount']:5d} {d['theta']:5d} {d['length']:3d} {d['offset']:3d} {d['palm']:4d} | "
                  f"{ok.sum():4d} / {len(hands)}  best {'' if i_ok is None else int(tot[i_ok])} | {np.median(tot):5.0f}")
        out = pathlib.Path("debug_outputs/17sep_coevo_analysis"); out.mkdir(parents=True, exist_ok=True)
        (out / f"edit_distance_{label}.json").write_text(json.dumps(rows, indent=1))
        print(f"wrote {out}/edit_distance_{label}.json\n")


if __name__ == "__main__":
    main()
