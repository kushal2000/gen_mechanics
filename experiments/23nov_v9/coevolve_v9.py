"""v9 (23 Nov 2026): the v7 script under the v9 grammar (configure("v9"): v7 + the outward-finger rule).

v7 (18 Nov 2026): the v5 script under the v7 grammar (hand_sampler/grammar/v3.py configure("v7")).

v5 co-evolution (14 Nov 2026): v4 (experiments/08nov_v4) with the v5 palm-width rule -- 3+ finger palms at least
50 mm across at their narrowest (hand_sampler/grammar/v3.py configure("v5"); layout cells labelled by facing spread).

v4 co-evolution (8 Nov 2026): experiments/07nov_v3/coevolve_v3.py under the v4 grammar (hand_sampler/grammar/v3.py
configure("v4"): every joint -30..90 deg, abduction too; each finger's last joint a flexion), its own feasible cells
(debug_outputs/coevolve/v9/feasible_cells.json) and seeds; the policy trains without joint limits in its tokens, joint
angles and actions on one fixed range (experiments/23nov_v9/coevolve_v9.sub). Everything else as v3:

v3 co-evolution (7 Nov 2026): one shared policy, an archive of 16 designs in each of the v3 MAP-Elites cells
(coevolution/evolution/descriptors_v3.py; the niche list debug_outputs/coevolve/v9/feasible_cells.json), link-only hands
(hand_sampler/grammar/v3.py) and the v3 operators (hand_sampler/grammar/v3_ops.py).

Fingers may roll about their own length (v3.ROLLS); designs are read and written with v3.hand_to_dict / hand_from_dict.
Each iteration trains a stratified 4,096 of the archive (an equal share from every occupied cell: the top of each cell
by the value net's prediction; gen 0: the seed script's stratified sample). Then: labels from the training logs -> the
design value net (v3 tokens, coevolution/evolution/value_tokens_v3.py) -> K rounds, each applying ALL ten operators to
EVERY archive member, the children de-duplicated and predicted, every cell keeping its top 16 (old and new) -> the
new members built -> the next training sample. K = 0 trains the same sample again (House of Dextra / Locoformer).

    python experiments/23nov_v9/coevolve_v9.py init --run <dir> --seeds debug_outputs/coevolve/v9/seeds [--k 5]
    python experiments/23nov_v9/coevolve_v9.py trained --run <dir> --train-run <dir> --checkpoint <pth>
    python experiments/23nov_v9/coevolve_v9.py next --run <dir>
One SLURM job per iteration (experiments/23nov_v9/coevolve_v9.sub): train -> trained -> next -> resubmit.
Run folder: config.json state.json designs.json hands/ ; gen_XXXX/{archive.json (cell -> ids, the 16 per cell),
manifest.json (the training sample), labels.json, value_net.pt, policy.pth, search/round_RR_*}.
Children landing outside the niche list are counted and logged (search/round_RR_offmap.json), never admitted.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/05nov_noevolution/studies"))
from hand_sampler.grammar import v3 as _v3  # noqa: E402
_v3.configure("v9")                          # this process and its forked workers (Pool): the v4 grammar
PER_CELL = 16
TRAIN = 4096
LABEL_FRACTION = 0.25
CELLS_FILE = REPO / "debug_outputs/coevolve/v9/feasible_cells.json"


def gen_dir(run: Path, g: int) -> Path:
    return run / f"gen_{g:04d}"


def load(run: Path, name: str):
    return json.loads((run / name).read_text())


def save(run: Path, name: str, obj) -> None:
    tmp = run / f".{name}.tmp"
    tmp.write_text(json.dumps(obj))
    tmp.replace(run / name)


def key_of(design: dict) -> str:
    return json.dumps(design, sort_keys=True)


def stratified(archive: dict, pred: dict, total: int, rng) -> list[str]:
    """`total` ids, an equal share from every non-empty cell (the remainder one by one, cells in random order); within
    a cell the top by prediction."""
    cells = sorted(c for c, v in archive.items() if v)
    quota = {c: 0 for c in cells}
    left = min(total, sum(len(archive[c]) for c in cells))
    while left > 0:
        moved = False
        for c in rng.permutation(cells):
            if left == 0:
                break
            if quota[c] < len(archive[c]):
                quota[c] += 1
                left -= 1
                moved = True
        if not moved:
            break
    out = []
    for c in cells:
        out += sorted(archive[c], key=lambda i: -pred.get(i, 0.0))[:quota[c]]
    return out


def init(args) -> None:
    run = Path(args.run)
    if (run / "state.json").exists():
        raise SystemExit(f"{run} already initialised")
    sd = REPO / args.seeds
    allm = json.loads((sd / "manifest.json").read_text())
    train = json.loads((sd / "train_manifest.json").read_text())
    if args.cell_regex:       # targeted map (16 Nov 2026): only the cells matching it, its own stratified training sample
        rx = re.compile(args.cell_regex)
        allm = {**allm, "hands": [h for h in allm["hands"] if rx.search(h["niche"])]}
        by = defaultdict(list)
        for h in allm["hands"]:
            by[h["niche"]].append(h["id"])
        rng0 = np.random.default_rng(args.seed + 11)
        pick = set(stratified(by, {h["id"]: float(rng0.random()) for h in allm["hands"]}, TRAIN, rng0))
        train = {**train, "hands": [h for h in allm["hands"] if h["id"] in pick]}
    gd = gen_dir(run, 0)
    gd.mkdir(parents=True)
    reg = {h["id"]: {"design": json.loads((REPO / h["spec"]).read_text())["design"], "niche": h["niche"], "born": 0,
                     "parent": None, "operator": None, "edit": None, "spec": h["spec"]} for h in allm["hands"]}
    archive = defaultdict(list)
    for h in allm["hands"]:
        archive[h["niche"]].append(h["id"])
    (gd / "archive.json").write_text(json.dumps(archive))
    (gd / "manifest.json").write_text(json.dumps({"generation": 0, "grid": train["grid"], "hands": train["hands"]}))
    save(run, "designs.json", reg)
    save(run, "config.json", {"k": args.k, "rule": args.rule, "seeds": args.seeds, "seed": args.seed,
                              "per_cell": PER_CELL, "train": TRAIN, "label_fraction": LABEL_FRACTION,
                              "cells": str(CELLS_FILE.relative_to(REPO)), "grid": train["grid"],
                              "expl_reward_coef_scale": args.expl_scale, "anchor_fraction": args.anchor_fraction,
                              "child_margin_std": args.child_margin_std, "max_new_per_cell": args.max_new_per_cell,
                              "cell_regex": args.cell_regex})
    save(run, "state.json", {"generation": 0, "trained": False, "checkpoint": None, "train_run": None, "history": []})
    print(f"[coevolve_v9] {run.name}: gen 0 = {len(reg)} designs in {len(archive)} cells, training {len(train['hands'])} "
          f"(K {args.k}, rule {args.rule})", flush=True)


def trained(args) -> None:
    run = Path(args.run)
    S = load(run, "state.json")
    S.update({"trained": True, "checkpoint": args.checkpoint, "train_run": args.train_run})
    S.setdefault("train_runs", {})[str(S["generation"])] = args.train_run
    save(run, "state.json", S)


def labels_from_logs(train_run: Path, manifest: dict) -> dict:
    d = train_run / "design_stats"
    dumps = sorted(d.glob("step_*.npz"))
    if not dumps:
        raise SystemExit(f"no design_stats under {d}")
    use = dumps[-max(1, math.ceil(LABEL_FRACTION * len(dumps))):]
    hoe = np.load(d / "meta.npz")["hand_of_env"].astype(int)
    hands = json.loads((d / "hands.json").read_text())
    env = np.concatenate([np.load(f)["env"] for f in use]).astype(int)
    goals = np.concatenate([np.load(f)["goals"] for f in use])
    ep = np.bincount(hoe[env], minlength=len(hands))
    gl = np.bincount(hoe[env], goals, minlength=len(hands))
    id_of = {h["spec"]: h["id"] for h in manifest["hands"]}
    return {id_of[s]: [float(gl[k] / ep[k]), int(ep[k])] for k, s in enumerate(hands) if ep[k] > 0 and s in id_of}


def _children(job):
    """Every operator applied once to one parent: [(operator, design, move)]."""
    pid, design, seed = job
    from hand_sampler.grammar import v3, v3_ops
    rng = np.random.default_rng(seed)
    h = v3.hand_from_dict(design)
    out = []
    for op in v3_ops.OPERATORS:
        res = v3_ops.apply_operator(h, op.name, rng)
        if res is not None:
            out.append((op.name, v3.hand_to_dict(res[0]), list(res[1]) if isinstance(res[1], tuple) else res[1]))
    return pid, out


def _cell(design: dict):
    from hand_sampler.grammar import v3
    from coevolution.evolution.descriptors_v3 import cell
    return cell(v3.design_features(v3.hand_from_dict(design)))


def _build(job):
    rid, design, out_dir = job
    from hand_sampler.grammar import v3
    row = v3.build(v3.hand_from_dict(design), rid, Path(out_dir) / rid)
    return rid, row["spec"]


def next_gen(args) -> None:
    run = Path(args.run)
    cfg, S, reg = load(run, "config.json"), load(run, "state.json"), load(run, "designs.json")
    g = S["generation"]
    if not S["trained"]:
        raise SystemExit(f"gen {g} is not trained yet")
    t0 = time.time()
    gd, nd = gen_dir(run, g), gen_dir(run, g + 1)
    man = json.loads((gd / "manifest.json").read_text())
    archive = json.loads((gd / "archive.json").read_text())
    lab = labels_from_logs(Path(S["train_run"]), man)
    (gd / "labels.json").write_text(json.dumps(lab))
    nd.mkdir(parents=True, exist_ok=True)
    hist = {"generation": g, "labels_mean": float(np.mean([v[0] for v in lab.values()])), "rounds": []}
    if cfg["k"] == 0:
        (nd / "archive.json").write_text(json.dumps(archive))
        (nd / "manifest.json").write_text(json.dumps({**man, "generation": g + 1}))
    else:
        hist.update(search(run, cfg, reg, man, archive, lab, g, nd))
    S["history"].append({**hist, "propose_s": time.time() - t0})
    S.update({"generation": g + 1, "trained": False, "checkpoint": str(gd / "policy.pth"), "train_run": None})
    save(run, "state.json", S)
    print(f"[coevolve_v9] {run.name}: gen {g + 1} ready ({time.time() - t0:.0f} s)", flush=True)


def search(run: Path, cfg: dict, reg: dict, man: dict, archive: dict, lab: dict, g: int, nd: Path) -> dict:
    import torch
    import design_value_net as dvn
    from coevolution.evolution import value_tokens_v3
    value_tokens_v3.install()
    niches = set(json.loads((REPO / cfg["cells"]).read_text())["cells"])
    if cfg.get("cell_regex"):                            # targeted map: children outside it are off the map
        niches = {c for c in niches if re.search(cfg["cell_regex"], c)}
    gd = gen_dir(run, g)
    sd = gd / "search"
    sd.mkdir(exist_ok=True)
    ids = [h["id"] for h in man["hands"] if h["id"] in lab]
    t = time.time()
    net = dvn.fit_transformer([reg[i]["design"] for i in ids], [lab[i][0] for i in ids], seed=cfg["seed"] + g)
    torch.save(net.net.state_dict(), gd / "value_net.pt")
    fit_s = time.time() - t
    flat = cfg["rule"] == "global"                       # the no-MAP-Elites ablation: one pool, no per-cell quota
    # child_margin_std (15 Nov 2026): a child displaces an archive member only if predicted better by this many
    # standard deviations of the generation's labels -- v4's archive turned over 100% every generation on the value
    # net's say-so, drifting to designs only the co-adapted policy could use
    margin = float(cfg.get("child_margin_std", 0.0)) * float(np.std([lab[i][0] for i in ids]))
    size = sum(len(v) for v in archive.values())
    members = [i for v in archive.values() for i in v]
    niche_of = {i: reg[i]["niche"] for i in members}
    pred = dict(zip(members, net([reg[i]["design"] for i in members]).tolist()))
    keys = {key_of(r["design"]) for r in reg.values()}
    kids_all, rounds = {}, []
    before = {c: list(v) for c, v in archive.items()}    # the archive as it entered (the rounds edit `work` in place)
    work = archive if not flat else {"all": members}
    for r in range(1, cfg["k"] + 1):
        parents = [i for v in work.values() for i in v]
        seeds = [cfg["seed"] * 1_000_003 + g * 100_003 + r * 10_007 + k for k in range(len(parents))]
        t1 = time.time()
        with Pool(cfg.get("workers", 12)) as pool:
            res = pool.map(_children, [(p, (reg.get(p) or kids_all[p])["design"], s) for p, s in zip(parents, seeds)],
                           chunksize=16)
        mut_s = time.time() - t1
        kids, n_raw, n_dup, off = [], 0, 0, Counter()
        flat_kids = [(p, op, d, mv) for p, cs in res for op, d, mv in cs]
        with Pool(cfg.get("workers", 12)) as pool:
            cells = pool.map(_cell, [d for _, _, d, _ in flat_kids], chunksize=64)
        for (p, op, d, mv), c in zip(flat_kids, cells):
            n_raw += 1
            k = key_of(d)
            if k in keys:
                n_dup += 1
                continue
            if c not in niches:
                off[str(c)] += 1
                continue
            keys.add(k)
            kids.append((p, op, mv, d, c))
        P = net([d for *_, d, _ in kids]).tolist() if kids else []
        cids = []
        for j, ((p, op, mv, d, c), v) in enumerate(zip(kids, P)):
            cid = f"g{g + 1:04d}r{r:02d}_{j:06d}"
            kids_all[cid] = {"design": d, "niche": c, "born": g + 1, "parent": p, "operator": op, "edit": mv,
                             "spec": None}
            pred[cid] = v
            niche_of[cid] = c
            (work["all"] if flat else work.setdefault(c, [])).append(cid)
            cids.append(cid)
        for c in work:                                   # children rank margin below their prediction (child_margin)
            work[c] = sorted(work[c], key=lambda i: -(pred[i] - (margin if i in kids_all else 0.0)))[
                : (size if flat else PER_CELL)]
        inside = {i for v in work.values() for i in v}
        with gzip.open(sd / f"round_{r:02d}_children.jsonl.gz", "wt") as fh:
            for cid in cids:
                ch = kids_all[cid]
                fh.write(json.dumps({"id": cid, "parent": ch["parent"], "operator": ch["operator"],
                                     "niche": ch["niche"], "pred": pred[cid], "admitted": cid in inside,
                                     "design": ch["design"]}) + "\n")
        if off:
            (sd / f"round_{r:02d}_offmap.json").write_text(json.dumps(off))
        pm = float(np.mean([pred[i] for i in inside]))
        ops_in = Counter(kids_all[i]["operator"] for i in inside if i in kids_all)
        rounds.append({"round": r, "children_raw": n_raw, "duplicates": n_dup, "off_map": sum(off.values()),
                       "children": len(cids), "admitted": sum(c in inside for c in cids), "pred_mean": pm,
                       "occupied_cells": len({niche_of[i] for i in inside}), "admitted_by_operator": dict(ops_in),
                       "mutate_s": mut_s})
        print(f"[coevolve_v9] gen {g} round {r}/{cfg['k']}: {len(cids)} children ({rounds[-1]['admitted']} admitted, "
              f"{n_dup} duplicates, {sum(off.values())} off the map), predicted mean {pm:.3f}, "
              f"{rounds[-1]['occupied_cells']} cells ({mut_s:.0f} s mutating)", flush=True)
    cap = int(cfg.get("max_new_per_cell", 0))
    if cap and not flat:      # turnover cap (15 Nov 2026): per cell at most `cap` new members, the rest stay old ones
        for c in work:
            old = [i for i in before.get(c, []) if i not in kids_all]
            new = [i for i in work[c] if i in kids_all][:cap]
            keep_old = sorted(old, key=lambda i: -pred.get(i, 0.0))[: max(0, PER_CELL - len(new))]
            if len(new) + len(keep_old) < PER_CELL:          # a cell short of old members: more new ones fill it
                extra = [i for i in work[c] if i in kids_all and i not in new]
                new += extra[: PER_CELL - len(new) - len(keep_old)]
            work[c] = sorted(new + keep_old, key=lambda i: -pred.get(i, 0.0))
    final = {i for v in work.values() for i in v}
    by_cell = defaultdict(list)
    for i in final:
        by_cell[niche_of[i]].append(i)
    new = [i for i in final if i not in reg]
    for i in new:
        reg[i] = kids_all[i]
    t2 = time.time()
    with Pool(cfg.get("workers", 12)) as pool:
        for i, spec in pool.map(_build, [(i, reg[i]["design"], str(run / "hands")) for i in new], chunksize=8):
            reg[i]["spec"] = spec
    rng_t = np.random.default_rng(cfg["seed"] * 7919 + g)
    n_anchor = int(round(cfg.get("anchor_fraction", 0.0) * cfg["train"]))
    train = stratified(by_cell, pred, cfg["train"] - n_anchor, rng_t)
    anchors = anchor_sample(reg, set(train), n_anchor, rng_t, cfg.get("cell_regex", "")) if n_anchor else []
    niche_of.update({i: reg[i]["niche"] for i in anchors})
    train = train + anchors
    (nd / "archive.json").write_text(json.dumps({c: sorted(v, key=lambda i: -pred[i]) for c, v in by_cell.items()}))
    (nd / "manifest.json").write_text(json.dumps({"generation": g + 1, "grid": man["grid"], "hands": [
        {"id": i, "spec": reg[i]["spec"], "niche": niche_of[i]} for i in sorted(train, key=lambda i: (niche_of[i], i))]}))
    save(run, "designs.json", reg)
    return {"rounds": rounds, "fit_s": fit_s, "build_s": time.time() - t2, "new_designs": len(new),
            "archive": len(final), "cells": len(by_cell), "train": len(train), "anchors": len(anchors)}


def anchor_sample(reg: dict, exclude: set, n: int, rng, cell_regex: str = "") -> list[str]:
    """n gen-0 seed designs (config anchor_fraction, 15 Nov 2026), an equal share from every cell, at random within a
    cell, none already in the sample: the training sample keeps a fixed share of ordinary hands every generation, so
    the shared policy cannot specialise to the archive's drift (v4's MAP-Elites policy scored 0 on its own gen-0
    designs by gen 48). Anchors are trained and labelled (the value net sees them) but never enter the archive."""
    by = defaultdict(list)
    for i, r in reg.items():
        if (r.get("born", 0) == 0 and r.get("parent") is None and i not in exclude
                and (not cell_regex or re.search(cell_regex, r["niche"]))):
            by[r["niche"]].append(i)
    return stratified(by, {i: float(rng.random()) for v in by.values() for i in v}, n, rng)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("init", "trained", "next"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="debug_outputs/coevolve/v9/seeds")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rule", choices=("none", "global"), default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expl-scale", type=float, default=0.001)
    ap.add_argument("--cell-regex", default="", help="targeted map: only the cells matching this regex (16 Nov)")
    ap.add_argument("--max-new-per-cell", type=int, default=0,
                    help="at most this many new members per cell per generation (0: no cap; 15 Nov)")
    ap.add_argument("--child-margin-std", type=float, default=0.0,
                    help="children need a predicted lead of this many label standard deviations to enter (15 Nov)")
    ap.add_argument("--anchor-fraction", type=float, default=0.0,
                    help="share of every training sample drawn from the gen-0 seeds (stratified by cell; 15 Nov)")
    ap.add_argument("--train-run")
    ap.add_argument("--checkpoint")
    args = ap.parse_args()
    {"init": init, "trained": trained, "next": next_gen}[args.phase](args)


if __name__ == "__main__":
    main()
