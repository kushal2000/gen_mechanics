"""The generation step's CLI (Kit-free); experiments/29oct_evolution_from_scratch/generation.sh calls it.

    python -m coevolution.evolution.generation init    --run-dir D --config C.json --seed-manifest M
                                                       [--seed-fitness F --checkpoint P]
    python -m coevolution.evolution.generation propose --run-dir D --fitness F
    python -m coevolution.evolution.generation admit   --run-dir D --checkpoint P [--screen-fitness S]
    python -m coevolution.evolution.generation status  --run-dir D

A run directory holds ``config.json``, ``archive.json``, ``state.json`` (generation, checkpoint, the MAP-Elites map,
per-generation summaries), ``hands/<id>/`` (every child built) and ``gen_NNNN/``: ``manifest.json`` (the population
trained that generation), ``fitness.json`` (its evaluation after training), ``children_manifest.json`` (the children
proposed) and, for screened strategies, ``screen.json`` (their zero-shot evaluation). Manifests pin the slot grid, so
the network keeps one shape for the whole run.

Generation g:
  1. the trainer trains ``gen_g/manifest.json`` (from ``state.checkpoint``; from scratch at g = 0 with none)
  2. eval_hands.py scores it with the new checkpoint -> ``gen_g/fitness.json``
  3. ``propose``: records those scores; MAP-Elites settles its challengers against their niches' elites; parents are
     chosen on the CURRENT generation's fitness; children are built -> ``gen_g/children_manifest.json``
  4. screened strategies: eval_hands.py scores the children zero-shot -> ``gen_g/screen.json``
  5. ``admit``: archive / greedy / random keep the protected newcomers and the fittest survivors and admit the best
     screened children; map_elites makes every child a challenger (trained next generation, then judged); writes
     ``gen_{g+1}/manifest.json``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import descriptors
from .archive import Archive, HandRecord
from .config import EvolutionConfig
from .select import select_parents
from .variation import design_key, make_child

FITNESS = "goals_per_episode"      # the scalar selection uses: eval_hands.py, each env's first episode
# The training reward per step breaks ties among hands with no goals: |reward per step| is ~0.2 or less, so at 0.1 it
# moves a hand by at most ~0.02 goals/episode -- it ranks the zeros and never outweighs one goal in 10 episodes.
SHAPING_WEIGHT = 0.1


# --- run-directory files ----------------------------------------------------------------------------------------

def _gen_dir(run: Path, g: int) -> Path:
    d = run / f"gen_{g:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state(run: Path) -> dict:
    return json.loads((run / "state.json").read_text())


def _save_state(run: Path, state: dict) -> None:
    tmp = run / "state.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, run / "state.json")


def write_manifest(path: Path, records: list[HandRecord], cfg: EvolutionConfig, generation: int) -> None:
    path.write_text(json.dumps({"generation": generation, "grid": cfg.grid,
                                "hands": [{"id": r.id, "spec": r.spec} for r in records]}, indent=1))


def _metrics(fitness: dict, spec: str) -> dict:
    """eval_hands.py's entry for a hand, keyed by its spec path; ``value`` is the selection scalar."""
    try:
        m = dict(fitness["hands"][spec])
    except KeyError:
        raise KeyError(f"{spec}: not in the fitness file") from None
    m["value"] = float(m[FITNESS]) + SHAPING_WEIGHT * float(m.get("reward_per_step", 0.0))
    return m


def _trained(r: HandRecord) -> dict | None:
    return r.latest(screen=False)


def best_per_niche(records) -> dict[str, HandRecord]:
    """The best trained-on (or seed-evaluated) hand per niche: the archive's map, for summaries."""
    best: dict[str, HandRecord] = {}
    for r in records:
        e = _trained(r)
        if e is not None and (r.niche not in best or e["value"] > _trained(best[r.niche])["value"]):
            best[r.niche] = r
    return best


def _summary(g: int, checkpoint: str, active: list[HandRecord], niche_best: dict, extra: dict) -> dict:
    goals = [_trained(r)[FITNESS] for r in active if _trained(r) and _trained(r)["generation"] == g]
    elite_goals = [_trained(e)[FITNESS] for e in niche_best.values() if _trained(e)]
    return {"generation": g, "checkpoint": checkpoint, "active": len(active),
            "best": max(goals) if goals else None, "mean_active": float(np.mean(goals)) if goals else None,
            "niches_filled": len(niche_best), "niches_with_goals": sum(1 for v in elite_goals if v > 0),
            "qd_score": float(sum(elite_goals)), **extra}


# --- commands ---------------------------------------------------------------------------------------------------

def init(run: Path, cfg: EvolutionConfig, seed_manifest: Path, seed_fitness: Path | None, checkpoint: str) -> None:
    """The archive from a seed manifest. From scratch (no fitness): every seed hand is the first population. With a
    seed evaluation: the ``population`` fittest. map_elites: each seed hand starts as its niche's elite."""
    from hand_sampler import resolve
    from hand_sampler.grammar import hand as gh
    if (run / "archive.json").exists():
        raise FileExistsError(f"{run} already holds a run")
    run.mkdir(parents=True, exist_ok=True)
    cfg.save(run / "config.json")
    fit = json.loads(Path(seed_fitness).read_text()) if seed_fitness else None
    rules = cfg.rule_set()
    archive = Archive()
    for h in json.loads(Path(seed_manifest).read_text())["hands"]:
        spec = h["spec"]
        design = json.loads(resolve(spec).with_name(Path(spec).name.replace(".spec.json", ".design.json")).read_text())
        rid = h.get("id") or Path(spec).name.replace(".spec.json", "")
        archive.add(HandRecord(id=rid, spec=spec, design=design, born=0, status="retired", operator="seed",
                               key=design_key(gh.hand_from_dict(design), rules)))
        if fit is not None:
            archive.record(rid, 0, _metrics(fit, spec), screen=False)
    seeds = list(archive.records.values())
    if fit is not None:
        seeds.sort(key=lambda r: -r.current())
    if cfg.strategy == "map_elites":
        emap: dict[str, str] = {}
        for r in seeds:                                   # fittest first (or manifest order): the first per niche
            emap.setdefault(r.niche, r.id)
        first = [archive[i] for i in emap.values()]
    else:
        emap = {}
        first = seeds if fit is None else seeds[:cfg.population]
    for r in first:
        r.status, r.admitted = "active", 0
    archive.save(run / "archive.json")
    write_manifest(_gen_dir(run, 0) / "manifest.json", first, cfg, 0)
    _save_state(run, {"generation": 0, "checkpoint": checkpoint or "", "map": emap, "history": []})
    print(f"[evolution] {run}: {len(archive)} seed hands, first population {len(first)}"
          + (f", {len(emap)} niches" if emap else "") + ("" if fit else " (from scratch)"))


def _settle_challengers(archive: Archive, emap: dict, g: int) -> dict:
    """MAP-Elites replacement: every challenger that has trained a generation (admitted at g) against its niche's
    elite, both scored in this generation's evaluation pass. It takes the niche if the niche is empty or it scores
    higher (no margin: noisy early rankings are fine); the loser retires to the archive."""
    chall = [r for r in archive.with_status("active") if r.operator != "seed" and r.admitted == g
             and r.id not in emap.values()]
    won = lost = 0
    for c in sorted(chall, key=lambda r: -_trained(r)["value"]):
        inc = emap.get(c.niche)
        if inc is None or _trained(c)["value"] > _trained(archive[inc])["value"]:
            if inc is not None:
                archive[inc].status = "retired"
            emap[c.niche] = c.id
            won += 1
        else:
            c.status = "retired"
            lost += 1
    return {"won": won, "lost": lost}


def propose(run: Path, fitness_path: Path) -> None:
    cfg = EvolutionConfig.load(run / "config.json")
    state = _state(run)
    g = state["generation"]
    archive = Archive.load(run / "archive.json")
    gdir = _gen_dir(run, g)
    if archive.with_status("candidate"):
        # a resubmitted generation (the job died after propose): keep its children, so nothing is made twice
        if not (gdir / "children_manifest.json").exists():
            raise RuntimeError(f"generation {g} has candidates but no children manifest")
        print(f"[evolution] generation {g}: children already proposed ({len(archive.with_status('candidate'))})")
        return
    fit = json.loads(Path(fitness_path).read_text())
    active = archive.with_status("active")
    for r in active:
        archive.record(r.id, g, _metrics(fit, r.spec), screen=False)
    settled = _settle_challengers(archive, state["map"], g) if cfg.strategy == "map_elites" else {}
    rng = np.random.default_rng([cfg.seed, g])
    taken = archive.keys()
    made, attempts = [], 0
    while len(made) < cfg.children and attempts < 4 * cfg.children:
        parent = select_parents(archive, cfg, rng, 1, elites=state.get("map"))[0]
        attempts += 1
        cid = f"g{g:04d}_c{len(made):02d}"
        child = make_child(None if parent is None else archive[parent].design, cid, run / "hands", cfg, rng, taken)
        if child is None:
            continue
        archive.add(HandRecord(id=cid, born=g, status="candidate", parent=parent, **child))
        made.append(archive[cid])
    write_manifest(gdir / "children_manifest.json", made, cfg, g)
    state["pending"] = settled
    archive.save(run / "archive.json")
    _save_state(run, state)
    ops: dict[str, int] = {}
    for r in made:
        ops[r.operator] = ops.get(r.operator, 0) + 1
    print(f"[evolution] generation {g}: recorded {len(active)} hands; "
          + (f"challengers {settled['won']} won, {settled['lost']} lost; " if settled else "")
          + f"{len(made)} children from {attempts} attempts ({ops})")


def _screen_order(c: HandRecord, archive: Archive, cfg: EvolutionConfig) -> tuple:
    """Admission key for screened strategies, best first: passes the floor, zero-shot fitness, novelty."""
    screen = c.current() or 0.0
    parent = archive[c.parent].current() if c.parent else None
    # screen_floor 0 (the default) means no floor at all: zero-goal fitness can be slightly negative (the reward
    # tie-breaker), and "screen >= 0 x parent" would then refuse every child (the 29 Oct dry run admitted none)
    passes = cfg.screen_floor <= 0 or parent is None or screen >= cfg.screen_floor * parent
    others = [descriptors.genetic_vector(r.design) for r in archive.records.values()
              if r.id != c.id and r.current() is not None and r.status != "candidate"]
    nov = descriptors.novelty(descriptors.genetic_vector(c.design), others, cfg.novelty_k)
    return (not passes, -screen, -nov)


def admit(run: Path, checkpoint: str, screen_path: Path | None) -> None:
    cfg = EvolutionConfig.load(run / "config.json")
    state = _state(run)
    g = state["generation"]
    archive = Archive.load(run / "archive.json")
    cands = archive.with_status("candidate")
    active = archive.with_status("active")
    if cfg.strategy == "map_elites":
        for c in cands:                                   # every child a challenger, trained before it is judged
            c.status, c.admitted = "active", g + 1
        nxt = [archive[i] for i in state["map"].values()] + cands
        niche_best = {n: archive[i] for n, i in state["map"].items()}
        extra = {"admitted": [c.id for c in cands], **state.pop("pending", {})}
    else:
        if screen_path is None:
            raise ValueError(f"strategy {cfg.strategy} admits by a zero-shot screen: pass --screen-fitness")
        scr = json.loads(Path(screen_path).read_text())
        for c in cands:
            archive.record(c.id, g, _metrics(scr, c.spec), screen=True)
        ranked = sorted(cands, key=lambda c: _screen_order(c, archive, cfg))
        passing = [c for c in ranked if not _screen_order(c, archive, cfg)[0]]
        protected = [r for r in active if r.operator != "seed" and r.admitted is not None
                     and g - r.admitted < cfg.protect_generations]
        new = passing[:max(0, min(cfg.admitted, len(passing), cfg.population - len(protected)))]
        rest = sorted((r for r in active if r not in protected), key=lambda r: -r.current())
        keep = protected + rest[:max(0, cfg.population - len(new) - len(protected))]
        for r in active:
            if r not in keep:
                r.status = "retired"
        for c in cands:
            if c in new:
                c.status, c.admitted = "active", g + 1
            else:
                c.status = "unadmitted"
        nxt = keep + new
        niche_best = best_per_niche(archive.records.values())
        extra = {"admitted": [c.id for c in new]}
        state.pop("pending", None)
    write_manifest(_gen_dir(run, g + 1) / "manifest.json", nxt, cfg, g + 1)
    summary = _summary(g, checkpoint, active, niche_best, extra)
    state["history"].append(summary)
    state["generation"], state["checkpoint"] = g + 1, checkpoint
    archive.save(run / "archive.json")
    _save_state(run, state)
    b, m = summary["best"], summary["mean_active"]
    print(f"[evolution] generation {g} done: best {b if b is None else round(b, 2)}, mean "
          f"{m if m is None else round(m, 2)} {FITNESS}; {len(summary['admitted'])} children in; "
          f"{summary['niches_filled']} niches ({summary['niches_with_goals']} with goals), QD score "
          f"{summary['qd_score']:.1f}; next population {len(nxt)}")


def status(run: Path) -> None:
    state = _state(run)
    archive = Archive.load(run / "archive.json")
    print(f"{run}: generation {state['generation']}, {len(archive)} hands in the archive"
          + (f", {len(state['map'])} niches in the map" if state.get("map") else ""))
    for h in state["history"]:
        b, m = h["best"], h["mean_active"]
        print(f"  gen {h['generation']:3d}: best {b if b is None else round(b, 2)}  mean "
              f"{m if m is None else round(m, 2)}  niches {h['niches_filled']:2d} ({h['niches_with_goals']} with "
              f"goals)  QD {h['qd_score']:.1f}  children in {len(h['admitted'])}")
    top = sorted((r for r in archive.records.values() if _trained(r)), key=lambda r: -_trained(r)["value"])[:10]
    for r in top:
        print(f"  {r.id:16s} {r.status:10s} {r.niche:16s} {_trained(r)[FITNESS]:6.2f}  parent {r.parent}  "
              f"op {r.operator}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init")
    a.add_argument("--run-dir", required=True)
    a.add_argument("--config", required=True)
    a.add_argument("--seed-manifest", required=True)
    a.add_argument("--seed-fitness", default=None)
    a.add_argument("--checkpoint", default="")
    p = sub.add_parser("propose")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--fitness", required=True)
    b = sub.add_parser("admit")
    b.add_argument("--run-dir", required=True)
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--screen-fitness", default=None)
    sub.add_parser("status").add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    if args.cmd == "init":
        init(run, EvolutionConfig.load(args.config), Path(args.seed_manifest),
             Path(args.seed_fitness) if args.seed_fitness else None, args.checkpoint)
    elif args.cmd == "propose":
        propose(run, Path(args.fitness))
    elif args.cmd == "admit":
        admit(run, args.checkpoint, Path(args.screen_fitness) if args.screen_fitness else None)
    else:
        status(run)


if __name__ == "__main__":
    main()
