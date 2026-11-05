"""Study 6: a design value function against the beam search of study 5 -- how does it judge the designs the beam
found, and can a beam search run on the value function alone (no simulation until the end)?

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_vf_beam.py --phase judge  [--labels eval4]
    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_vf_beam.py --phase search [--labels ...]
    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_vf_beam.py --phase validate

Labels (the value net's training data: the iteration's 4,096-hand population, nothing from the beam):
    eval4       the loop's own eval of that iteration (gen_NNNN/fitness.json: 4 envs, one episode each)
    trainlike   the training-like rollout (eval_training_like.py, all SAPG blocks, ~70 episodes per design) in
                it<N>_design_value/training_like<--trainlike-suffix>/
The net: study_value.py's DeepSets over fingers, an ensemble of 5 bootstrap members (study_design_value.fit_net);
the prediction is the members' mean, its spread the sd.

judge     every design the beam search scored (study_beam.py's state.json, 100 envs each): the net's prediction
          against its score -- overall, within each round's children, and along the best design's lineage.
search    the same beam search with the net as the only judge: from the same start, every valid one-step child of the
          beam (canonical form, as the study-5 beam deployed them), the beam = the --beam best by prediction (elitist),
          --rounds rounds. Writes vf_beam_<labels>.json.
validate  every design that was ever in the net's beam, scored with 100 envs (study_beam_eval.sub), so each round's
          predicted best / beam mean can be set against its true one, and against the study-5 beam.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments/04nov_audit_grammar"))
import study_beam as sb  # noqa: E402
import study_design_value as sdv  # noqa: E402
import study_value as sv  # noqa: E402

RUN = REPO / "debug_outputs/evolution/noevolution_4096"
GRID_FINGERS, GRID_SLOTS = 6, 5


def out_dir(args) -> Path:
    return REPO / "debug_outputs/noevolution/studies" / f"beam_{args.start}_it{args.iteration}"


def labels(args) -> dict[str, float]:
    """spec -> label for the iteration's population."""
    g = RUN / f"gen_{args.iteration:04d}"
    if args.labels == "eval4":
        return {s: v[sv.FIT] for s, v in json.loads((g / "fitness.json").read_text())["hands"].items()}
    if args.labels == "env100":                              # every population design at 100 envs (it<N>_env100/)
        out = {}
        for f in sorted((REPO / "debug_outputs/noevolution/studies" / f"it{args.iteration}_env100").glob("c*_fitness.json")):
            out.update({s: v[sv.FIT] for s, v in json.loads(f.read_text())["hands"].items()})
        if len(out) < 4000:
            raise SystemExit(f"only {len(out)} population designs have a 100-env score yet")
        return out
    d = REPO / "debug_outputs/noevolution/studies" / f"it{args.iteration}_design_value" / f"training_like{args.trainlike_suffix}"
    if not (d / "done.json").exists():
        raise SystemExit(f"no finished training-like rollout in {d}")
    meta = np.load(d / "meta.npz")
    hands = json.loads((d / "hands.json").read_text())
    hoe = meta["hand_of_env"].astype(int)
    dumps = sorted(d.glob("step_*.npz"))[1:]                # the first dump's episodes were cut by the stagger
    env = np.concatenate([np.load(f)["env"] for f in dumps]).astype(int)
    goals = np.concatenate([np.load(f)["goals"] for f in dumps])
    ep = np.bincount(hoe[env], minlength=len(hands))
    gl = np.bincount(hoe[env], goals, minlength=len(hands))
    print(f"[vf] trainlike labels: {ep.mean():.0f} episodes per design over {len(dumps)} dumps", flush=True)
    return {s: float(gl[h] / ep[h]) for h, s in enumerate(hands) if ep[h] > 0}


_POP = {}


def fit(args, extra: list | None = None, quiet: bool = False):
    """The ensemble's predict(designs) -> (mean, sd), fitted to the population labels (plus extra (design, label))."""
    import torch
    g = RUN / f"gen_{args.iteration:04d}"
    if "pop" not in _POP:
        man = json.loads((g / "manifest.json").read_text())
        L = labels(args)
        hs = [h for h in man["hands"] if h["spec"] in L]
        _POP["pop"] = ([sv._load_design(h["spec"]) for h in hs], [L[h["spec"]] for h in hs])
    designs, ys = list(_POP["pop"][0]), list(_POP["pop"][1])
    for d, v in extra or []:
        designs.append(d)
        ys.append(v)
    hs = designs
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, M, H = (t.to(dev) for t in sv._batch(designs))
    y = torch.tensor(ys, dtype=torch.float32, device=dev)
    members = [sdv.fit_net(X, M, H, y, seed=s, members=1) for s in range(args.members)]
    if not quiet:
        print(f"[vf] fitted {args.members} members on {len(hs)} designs ({args.labels} labels"
              f"{f' + {len(extra)} beam designs' if extra else ''}, mean {float(y.mean()):.3f})", flush=True)

    def predict(designs: list[dict]):
        out = []
        for k in range(0, len(designs), 2048):
            Xc, Mc, Hc = (t.to(dev) for t in sv._batch(designs[k:k + 2048]))
            out.append(torch.stack([p(Xc, Mc, Hc) for p in members]))
        P = torch.cat(out, 1)
        return P.mean(0).numpy(), P.std(0).numpy()
    return predict


def _spear(a, b):
    return sv._spearman(list(a), list(b))


def judge(args) -> None:
    S = json.loads((out_dir(args) / "state.json").read_text())
    D = {i: r for i, r in S["designs"].items() if r["scores"]}
    ids = list(D)
    predict = fit(args)
    mu, sd = predict([D[i]["design"] for i in ids])
    P = dict(zip(ids, mu))
    T = {i: st.mean(D[i]["scores"]) for i in ids}
    a, b = [P[i] for i in ids], [T[i] for i in ids]
    print(f"\n[vf] {len(ids)} beam-search designs with a 100-env score: prediction vs score pearson "
          f"{st.correlation(a, b):.2f}, spearman {_spear(a, b):.2f}; mean prediction {st.mean(a):.3f} vs "
          f"mean score {st.mean(b):.3f}")
    print(f"  {'round':>5s} {'children':>8s} {'spearman':>8s} {'pred best -> its true':>22s} {'true best':>9s} "
          f"{'net picks true top-10?':>22s}")
    for r in range(1, len(S["rounds"])):
        kids = [i for i in ids if D[i]["round"] == r]
        if len(kids) < 5:
            continue
        if args.with_beam:                                    # train on everything scored before this round
            prior = [(D[i]["design"], T[i]) for i in ids if D[i]["round"] < r]
            pr = fit(args, prior, quiet=True)
            m_r, _ = pr([D[i]["design"] for i in kids])
            P.update(zip(kids, m_r))
        pb = max(kids, key=lambda i: P[i])
        tb = max(kids, key=lambda i: T[i])
        top_t = set(sorted(kids, key=lambda i: -T[i])[:10])
        top_p = set(sorted(kids, key=lambda i: -P[i])[:10])
        print(f"  {r:5d} {len(kids):8d} {_spear([P[i] for i in kids], [T[i] for i in kids]):8.2f} "
              f"{P[pb]:10.2f} -> {T[pb]:.2f}   {T[tb]:9.2f} {len(top_t & top_p):>14d} of 10")
    best = S["rounds"][-1]["best"]
    chain = []
    i = best
    while i is not None:
        chain.append(i)
        i = S["designs"][i]["parent"]
    print("\n  the best design's lineage: round, design, 100-env score, prediction (+- ensemble sd)")
    for i in chain[::-1]:
        k = ids.index(i)
        print(f"    {D[i]['round']:2d} {i:16s} {T[i]:.3f}   {mu[k]:.3f} +- {sd[k]:.3f}   "
              f"{D[i]['operator'] or 'start'}")
    (out_dir(args) / f"vf_judge_{args.labels}{'_withbeam' if args.with_beam else ''}.json").write_text(json.dumps(
        {"labels": args.labels, "pred": {i: float(P[i]) for i in ids}, "sd": dict(zip(ids, map(float, sd)))}))


def _children(design: dict):
    from children_of import LEVEL, describe
    from coevolution.evolution.variation import design_key
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import operators as gops
    rules = getattr(gh, sb.RULES)
    parent = gh.hand_from_dict(design)
    out = []
    for op in gops.OPERATORS:
        if op.name not in LEVEL:
            continue
        for mv, child in gops.iter_valid(parent, op, rules, canonical_children=True):   # as study 5 deployed them
            d = gh.hand_to_dict(child)
            if len(d["fingers"]) > GRID_FINGERS or any(len(f["joints"]) > GRID_SLOTS for f in d["fingers"]):
                continue
            out.append((design_key(child, rules), d, op.name, describe(op.name, mv, parent)))
    return out


def search(args) -> None:
    from multiprocessing import Pool
    S = json.loads((out_dir(args) / "state.json").read_text())
    start = S["designs"][args.start]
    predict = fit(args)
    mu0, sd0 = predict([start["design"]])
    V = {args.start: {"design": start["design"], "key": start["key"], "parent": None, "operator": None, "edit": None,
                      "round": 0, "pred": float(mu0[0]), "sd": float(sd0[0])}}
    rounds = [{"beam": [args.start], "best": args.start, "best_pred": float(mu0[0])}]
    print(f"[vf] search from {args.start}: predicted {mu0[0]:.3f} (its 100-env score {st.mean(start['scores']):.3f})",
          flush=True)
    for r in range(1, args.rounds + 1):
        beam = rounds[-1]["beam"]
        seen = {v["key"] for v in V.values()}
        with Pool(args.workers) as pool:
            kids = pool.map(_children, [V[b]["design"] for b in beam], chunksize=1)
        new = {}
        for pid, cs in zip(beam, kids):
            for key, d, op, edit in cs:
                if key not in seen and key not in new:
                    new[key] = (pid, d, op, edit)
        if not new:
            break
        items = list(new.items())
        mu, sd = predict([d for _, (_, d, _, _) in items])
        for k, ((key, (pid, d, op, edit)), m, s) in enumerate(zip(items, mu, sd)):
            V[f"v{r:02d}_{k:04d}"] = {"design": d, "key": key, "parent": pid, "operator": op, "edit": edit,
                                     "round": r, "pred": float(m), "sd": float(s)}
        pool_ids = beam + [f"v{r:02d}_{k:04d}" for k in range(len(items))]
        score = (lambda i: V[i]["pred"] + args.kappa * V[i]["sd"])
        nb = sorted(pool_ids, key=lambda i: -score(i))[:args.beam]
        rounds.append({"beam": nb, "best": nb[0], "best_pred": V[nb[0]]["pred"], "children": len(items),
                       "beam_pred_mean": float(np.mean([V[i]["pred"] for i in nb]))})
        b = V[nb[0]]
        print(f"[vf] round {r}: {len(items)} children; best {nb[0]} predicted {b['pred']:.3f} +- {b['sd']:.3f} "
              f"({b['operator']}); beam predicted mean {rounds[-1]['beam_pred_mean']:.3f}; "
              f"{sum(V[i]['round'] == r for i in nb)} new in the beam", flush=True)
    (out_dir(args) / f"vf_beam_{args.labels}.json").write_text(json.dumps({"args": vars(args), "designs": V,
                                                                          "rounds": rounds}))


def validate(args) -> None:
    from multiprocessing import Pool
    od = out_dir(args)
    F = od / f"vf_beam_{args.labels}.json"
    VB = json.loads(F.read_text())
    V, R = VB["designs"], VB["rounds"]
    ids = sorted({i for rd in R for i in rd["beam"]})
    todo = [i for i in ids if "spec" not in V[i]]
    hands = od / "vf_hands"
    hands.mkdir(exist_ok=True)
    with Pool(args.workers) as pool:
        built = pool.map(sb._build, [(i, V[i]["design"], str(hands)) for i in todo], chunksize=4)
    for i, spec, ok in built:
        V[i]["spec"] = spec
    if args.start in V and "spec" not in V[args.start]:
        V[args.start]["spec"] = json.loads((od / "state.json").read_text())["designs"][args.start]["spec"]
    rows = [{"id": i, "spec": V[i]["spec"], "scores": V[i].get("scores", [])} for i in ids]
    g = RUN / f"gen_{args.iteration:04d}"
    gman = json.loads((g / "manifest.json").read_text())
    train_run = sorted(glob.glob(str(REPO / "debug_outputs/train_logs/evolution/noevolution_4096" / g.name / "*/")))[0]
    eargs = argparse.Namespace(chunk_hands=256, iteration=args.iteration, envs=100, table_clearance=0.02)
    (od / "rounds").mkdir(exist_ok=True)
    need = [r for r in rows if not r["scores"]]
    if need:
        sb.evaluate(od, 90 + len(args.labels), need, eargs, gman, train_run)       # rounds/r9x_*: the validation
    for r in rows:
        V[r["id"]]["scores"] = r["scores"]
    F.write_text(json.dumps(VB))
    S = json.loads((od / "state.json").read_text())
    print(f"\n[vf] the net's beam search ({args.labels} labels), each round's beam scored with 100 envs, "
          f"against the study-5 beam (simulated every child)")
    print(f"  {'round':>5s} {'net: predicted best':>19s} {'its true score':>14s} {'beam true mean':>14s} "
          f"{'true best in beam':>17s} | {'study-5 best':>12s}")
    for r, rd in enumerate(R):
        tb = [st.mean(V[i]["scores"]) for i in rd["beam"] if V[i].get("scores")]
        b = V[rd["best"]]
        s5 = S["rounds"][r]["best_score"] if r < len(S["rounds"]) else float("nan")
        print(f"  {r:5d} {rd['best_pred']:19.3f} {st.mean(b['scores']) if b.get('scores') else float('nan'):14.3f} "
              f"{st.mean(tb) if tb else float('nan'):14.3f} {max(tb) if tb else float('nan'):17.3f} | {s5:12.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("judge", "search", "validate"), required=True)
    ap.add_argument("--start", default="f4_m12-13_126")
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--labels", choices=("eval4", "trainlike", "env100"), default="eval4")
    ap.add_argument("--with-beam", action="store_true",
                    help="judge: before round r, also train on the beam designs scored in rounds < r (100 envs)")
    ap.add_argument("--trainlike-suffix", default="_fixreset")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--kappa", type=float, default=0.0, help="beam score = prediction + kappa x ensemble sd")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    {"judge": judge, "search": search, "validate": validate}[args.phase](args)


if __name__ == "__main__":
    main()
