"""Final analysis of one design: full-resolution metrics (3 sampling seeds), geometry numbers, comparison with the 100
random coarse hands, and figures.

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_design/analyze_design.py \
        [--design <design.json>] [--alt <design.json> --alt-label "..."]

-> agent_design/final_metrics.json, plots/agent_design/*.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pen_metrics as pm  # noqa: E402
import render  # noqa: E402
from hand_sampler.grammar import derive as dv  # noqa: E402
from hand_sampler.grammar import hand as gh  # noqa: E402
from hand_sampler.grammar import operators as gops  # noqa: E402
from hand_sampler.grammar import viability as gvb  # noqa: E402

RULES = gh.FEETECH_COARSE_RULES
PLOTS = HERE.parent / "plots/agent_design"
METRICS = ["pickup", "hold3", "pivot", "pivot_h", "gait_all", "reset_clean", "dof", "score"]
LABELS = {"pickup": "pickup (pen flat, any yaw)", "hold3": "hold any goal, 3 contacts", "pivot": "pivot, any axis",
          "pivot_h": "pivot, near-horizontal axis", "gait_all": "gait: lift any finger, keep 3",
          "reset_clean": "reset pose clear of pen/table", "dof": "joints (DoF)", "score": "composite score"}
BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#a9a8a3", "#0b0b0b", "#52514e"


def full_metrics(h, seeds=(0, 1, 2)):
    runs = []
    for s in seeds:
        S = [pm.finger_samples(h, i, pm.N_SAMPLES, seed=s) for i in range(len(h.fingers))]
        m = pm.evaluate(h, n_goals=96, n_axes=96, seed=s, samples=S, details=(s == 0))
        m["reset_clean"] = pm.reset_clean(S, seed=s)
        m.update(pm.gait_hold(h, S, n_goals=96, seed=s))
        m["score"] = pm.score(m)
        runs.append(m)
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    out = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    out["min"] = {k: float(np.min([r[k] for r in runs])) for k in keys}
    out["details"] = runs[0]["details"]
    return out


def jac_force(h, q, i, n_dir):
    """Largest force the fingertip of finger i can press along n_dir (palm frame, unit) at joint vector q, with every
    joint at most the effort limit (0.5 N m): min over joints of 0.5 / |J_j . n|."""
    ds = dv.dofs(h)
    cols = [k for k, d in enumerate(ds) if d.finger == i]
    eps = 1e-5
    p0 = dv.fk(h, q).tips[i]
    tau = []
    for k in cols:
        dq = q.copy()
        dq[k] += eps
        J = (dv.fk(h, dq).tips[i] - p0) / eps            # m / rad
        tau.append(abs(float(J @ n_dir)))
    return 0.5 / max(max(tau), 1e-9)


def pick_sample(fs, target, R, c, max_x, sweep=None):
    cand = pm.candidates(fs, target[None])[0]
    cand = cand[cand >= 0]
    if not len(cand):
        return None
    ok = pm._clear(fs.pts[cand], fs.rad, np.repeat(R[None], len(cand), 0), np.repeat(c[None], len(cand), 0),
                   np.full(len(cand), max_x), None if sweep is None else np.repeat(sweep[None], len(cand), 0))
    if not ok.any():
        return None
    good = cand[ok]
    return int(good[np.argmin(np.linalg.norm(fs.tip[good] - target, axis=1))])


def park_sample(fs, R, c, max_x, sweep=None):
    for s in fs.park:
        if pm._clear(fs.pts[s][None], fs.rad, R[None], c[None], np.array([max_x]),
                     None if sweep is None else sweep[None])[0]:
            return int(s)
    return None


def pivot_pose(h, S, u, pinch=(0, 1)):
    """A pivot about u (palm frame): (q, pen centre) with fingers pinch[0] / pinch[1] at c +- FACE_OFF u, everyone
    clear of the swept disc; None if not found on the centre line."""
    ext = pm.SWEEP_R * math.sqrt(max(0.0, 1 - u[0] ** 2)) + pm.PEN_HALF[1] * abs(u[0])
    for depth in np.arange(pm.PLATE_FACE + pm.GAP + ext, pm.WRIST_Z[1] - ext + 1, 2.0):
        for lat in [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (0.0, -10.0), (-10.0, 0.0)]:
            c = np.array([depth, *lat])
            a = pick_sample(S[pinch[0]], c + pm.FACE_OFF * u, np.eye(3), c, pm.WRIST_Z[1], u)
            b = pick_sample(S[pinch[1]], c - pm.FACE_OFF * u, np.eye(3), c, pm.WRIST_Z[1], u)
            if a is None or b is None:
                continue
            q = S[pinch[0]].q[a] + S[pinch[1]].q[b]
            ok = True
            for k, fs in enumerate(S):
                if k in pinch:
                    continue
                p = park_sample(fs, np.eye(3), c, pm.WRIST_Z[1], u)
                if p is None:
                    ok = False
                    break
                q = q + fs.q[p]
            if ok:
                return q, c
    return None


def geometry(h, S):
    """Cage, reach and opposition numbers."""
    g = {}
    zero = np.concatenate([fs.pts[0] for fs in S])
    rad = np.concatenate([fs.rad for fs in S])
    radial = np.hypot(zero[:, 1], zero[:, 2]) - rad
    below = zero[:, 0] > pm.PLATE_FACE                    # below the palm's front face
    g["cage_clear_radius_mm"] = float(radial[below].min())
    g["pen_sweep_radius_mm"] = pm.SWEEP_R
    g["pen_plan_half_diagonal_mm"] = float(np.hypot(pm.PEN_HALF[0], pm.PEN_HALF[1]))
    tips = [fs.tip for fs in S]
    g["tip_depth_range_mm"] = [float(min(t[:, 0].min() for t in tips)), float(max(t[:, 0].max() for t in tips))]
    bands = {}
    for d0 in range(30, 151, 20):
        vals = []
        for t in tips:
            m = (t[:, 0] >= d0) & (t[:, 0] < d0 + 20)
            vals.append(float(np.hypot(t[m, 1], t[m, 2]).min()) if m.any() else None)
        bands[f"{d0}-{d0 + 20}"] = vals
    g["tip_min_radial_mm_by_depth"] = bands         # tip sphere centre's closest approach to the palm axis
    g["pinch_needs_radial_mm"] = pm.FACE_OFF
    # opposite pairs: closest the two tip spheres come (touching: <= 2 r)
    pairs = []
    F = len(h.fingers)
    for a in range(F):
        for b in range(a + 1, F):
            fa, fb = h.fingers[a], h.fingers[b]
            ang = abs((math.degrees(math.atan2(fa.y, fa.z)) - math.degrees(math.atan2(fb.y, fb.z)) + 180) % 360 - 180)
            ta, tb = S[a].tip[::6], S[b].tip[::6]
            d = np.sqrt(((ta[:, None] - tb[None]) ** 2).sum(-1)).min()
            pairs.append({"fingers": [a, b], "bearing_gap_deg": round(ang, 1), "closest_tip_centres_mm": float(d)})
    g["finger_pairs"] = pairs
    g["link_masses_g"] = [[round(dv.link_mass_props(h, i, j)[0] * 1e3, 1) for j in range(len(f.joints))]
                          for i, f in enumerate(h.fingers)]
    return g


def strip_compare(rand, des, alt, alt_label, path):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    rng = np.random.default_rng(0)
    for ax, k in zip(axes.ravel(), METRICS):
        v = np.array([r[k] for r in rand], dtype=float)
        jit = rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(jit, v, s=16, color=GREY, alpha=0.8, linewidths=0, label="100 random hands")
        ax.scatter([0], [des[k]], s=90, color=BLUE, edgecolors="white", linewidths=1.5, zorder=3, label="claude_design_0")
        ax.annotate(f"{des[k]:.2f}" if k != "dof" else f"{des[k]:.0f}", (0.04, des[k]), xytext=(14, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
        if alt is not None:
            ax.scatter([0.3], [alt[k]], s=60, marker="D", color=ORANGE, edgecolors="white", linewidths=1.2, zorder=3,
                       label=alt_label)
        ax.set_title(LABELS[k], fontsize=9.5, color=INK)
        ax.set_xticks([])
        ax.set_xlim(-0.45, 0.55)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Static pen-repose metrics: the design vs the 100 random coarse hands (fractions; higher is better, "
                 "except DoF)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def workspace_plot(h, S, path):
    """One finger's fingertip reach in its radial-depth plane, with the pen and its sweep."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    f0 = h.fingers[0]
    u = np.array([f0.y, f0.z], dtype=float) / math.hypot(f0.y, f0.z)
    t = S[0].tip
    r_signed = t[:, 1:] @ u
    off = np.abs(t[:, 1:] @ np.array([-u[1], u[0]]))
    m = off < 10.0                                        # samples near the finger's own vertical plane
    ax.scatter(r_signed[m], -t[m, 0], s=3, color=BLUE, alpha=0.35, linewidths=0, label="fingertip centre (finger 0)")
    for k in range(1, len(S)):
        fk = h.fingers[k]
        if abs(math.atan2(fk.y, fk.z) - math.atan2(f0.y, f0.z)) % (2 * math.pi) and \
                abs(abs(math.atan2(fk.y, fk.z) - math.atan2(f0.y, f0.z)) - math.pi) < 0.1:
            t2 = S[k].tip
            m2 = np.abs(t2[:, 1:] @ np.array([-u[1], u[0]])) < 10.0
            ax.scatter(t2[m2, 1:] @ u, -t2[m2, 0], s=3, color=ORANGE, alpha=0.35, linewidths=0,
                       label=f"fingertip centre (finger {k}, opposite)")
    depth = 90.0
    th = np.linspace(0, 2 * math.pi, 200)
    ax.plot(pm.SWEEP_R * np.cos(th), -depth + pm.SWEEP_R * np.sin(th), color=INK2, lw=1.2, ls="--",
            label="pen sweep disc (r 52 mm), centre 90 mm down")
    ax.add_patch(plt.Rectangle((-50, -depth - 15), 100, 30, color="#2860c8", alpha=0.35, label="pen (100 x 30)"))
    for sgn in (1, -1):
        ax.add_patch(plt.Circle((sgn * pm.FACE_OFF, -depth), pm.TIP_R, fill=False, color=INK, lw=1))
    ax.axhline(-pm.PLATE_FACE, color=INK, lw=2)
    ax.text(-150, -pm.PLATE_FACE + 3, "palm front face", fontsize=8, color=INK2)
    ax.set_aspect("equal")
    ax.set_xlim(-160, 160)
    ax.set_ylim(-200, 20)
    ax.set_xlabel("mm along the finger pair's line (palm centre at 0)", fontsize=9)
    ax.set_ylabel("mm (up = toward the palm)", fontsize=9)
    ax.set_title("Fingertip reach of an opposite pair, vs the pen and the disc it sweeps", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default=str(HERE / "seeds/cross80_sbb_0_70_70.json"))
    ap.add_argument("--alt", default=None)
    ap.add_argument("--alt-label", default="alternative")
    args = ap.parse_args()
    PLOTS.mkdir(parents=True, exist_ok=True)
    h = gh.hand_from_dict(json.loads(Path(args.design).read_text()))
    h = gops.canonical(h, RULES)
    checks = {"rules": gh.check(h, RULES), "c1": bool(gvb.clear(h))}
    c2 = gvb.c2_workspace_overlap(h)
    checks["c2"] = {"ok": bool(c2.ok), "best_mm": float(c2.best_mm), "pair": list(c2.pair) if c2.pair else None}
    print("checks", checks, flush=True)

    des = full_metrics(h)
    print({k: round(v, 3) for k, v in des.items() if isinstance(v, float)}, flush=True)
    S = [pm.finger_samples(h, i) for i in range(len(h.fingers))]
    geo = geometry(h, S)
    print(json.dumps(geo, indent=0)[:1500], flush=True)

    alt = None
    if args.alt:
        ha = gh.hand_from_dict(json.loads(Path(args.alt).read_text()))
        alt = full_metrics(ha)
        alt.pop("details")

    # random hands
    rand = json.loads((HERE / "random_metrics.json").read_text())
    gait = json.loads((HERE / "random_gait.json").read_text())
    for r in rand:
        r.update(gait[r["name"]])
        r["score"] = pm.score(r)
    comp = {}
    for k in METRICS:
        v = np.array([r[k] for r in rand], dtype=float)
        comp[k] = {"design": des[k], "random_mean": float(v.mean()), "random_median": float(np.median(v)),
                   "random_max": float(v.max()), "random_best_by_score": None,
                   "share_of_random_at_or_above_design": float((v >= des[k] - 1e-9).mean())}
    best = max(rand, key=lambda r: r["score"])
    for k in METRICS:
        comp[k]["random_best_by_score"] = float(best[k])
    comp["best_random_by_score"] = best["name"]
    dominated = [r["name"] for r in rand if all(r[k] >= des[k] - 1e-9 for k in METRICS if k not in ("dof",))
                 and r["dof"] <= des["dof"]]
    comp["random_hands_matching_or_beating_design_on_every_metric"] = dominated

    # pinch force at a horizontal pivot about the line of fingers 0 and the one opposite it
    f0 = h.fingers[0]
    opp = min(range(1, len(h.fingers)), key=lambda k: math.hypot(h.fingers[k].y + f0.y, h.fingers[k].z + f0.z))
    u = np.array([0.0, f0.y, f0.z]) / math.hypot(f0.y, f0.z)
    pv = pivot_pose(h, S, u, pinch=(0, opp))
    forces = None
    if pv is not None:
        q, c = pv
        forces = {"pivot_centre_depth_mm": float(c[0]),
                  "finger0_inward_N": jac_force(h, q, 0, -u), f"finger{opp}_inward_N": jac_force(h, q, opp, u)}
        # renders: the pivot sequence
        e1 = np.cross(u, [1.0, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        poses = []
        for deg in (0, 45, 90, 135):
            a = math.radians(deg)
            x = math.cos(a) * e1 + math.sin(a) * e2
            R = np.column_stack([x, u, np.cross(x, u)])
            poses.append((f"pivot {deg} deg about the pinch line", q, (R, c), None, (5, -90 + 8)))
        render.render_hand(h, PLOTS / "design_pivot.png",
                           f"Pivot: fingers 0 and {opp} pinch the pen across its width, the other two parked "
                           f"(pen centre {c[0]:.0f} mm below the palm centre)", poses)
    # pickup render: pen flat at yaw 30 deg on the table
    yaw = math.radians(30)
    xl = np.array([0.0, math.cos(yaw), math.sin(yaw)])
    zl = np.array([-1.0, 0, 0])
    Rp = np.column_stack([xl, np.cross(zl, xl), zl])
    pick = None
    for depth in np.arange(29, 186, 4.0):
        c = np.array([depth, 0.0, 0.0])
        root = depth + pm.REST_Z
        tg = [c + Rp @ np.array([0, pm.FACE_OFF, 0]), c + Rp @ np.array([0, -pm.FACE_OFF, 0])]
        best_pair = None
        for a in range(len(S)):
            for b in range(len(S)):
                if a == b:
                    continue
                sa, sb = pick_sample(S[a], tg[0], Rp, c, root), pick_sample(S[b], tg[1], Rp, c, root)
                if sa is None or sb is None:
                    continue
                q = S[a].q[sa] + S[b].q[sb]
                ok = True
                for k in range(len(S)):
                    if k in (a, b):
                        continue
                    pk = park_sample(S[k], Rp, c, root)
                    if pk is None:
                        ok = False
                        break
                    q = q + S[k].q[pk]
                if ok:
                    best_pair = (a, b, q)
                    break
            if best_pair:
                break
        if best_pair:
            pick = (depth, root, best_pair)
            break
    if pick:
        depth, root, (a, b, q) = pick
        render.render_hand(h, PLOTS / "design_pickup.png",
                           f"Pickup: pen flat on the table at yaw 30 deg, fingers {a} and {b} pinch it "
                           f"(wrist root {root:.0f} mm above the table)",
                           [("pickup pinch", q, (Rp, np.array([depth, 0.0, 0.0])), root),
                            ("same, side view", q, (Rp, np.array([depth, 0.0, 0.0])), root, (0, -90))])
    render.render_hand(h, PLOTS / "design_views.png", "claude_design_0 at its zero (start) pose, palm down",
                       [("zero pose", None, None, None)])
    strip_compare(rand, des, alt, args.alt_label, PLOTS / "compare_random.png")
    workspace_plot(h, S, PLOTS / "workspace_pair.png")

    out = {"design": gh.hand_to_dict(h), "checks": checks, "metrics": {k: v for k, v in des.items() if k != "details"},
           "alt_metrics": alt, "geometry": geo, "pivot_force": forces, "comparison": comp,
           "pivot_detail": {"axes": des["details"]["pivot_axes"], "ok": des["details"]["pivot_ok"]}}
    (HERE / "final_metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"comparison": comp, "pivot_force": forces}, indent=1))


if __name__ == "__main__":
    main()
