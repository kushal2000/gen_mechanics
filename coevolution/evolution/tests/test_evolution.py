"""The evolution bookkeeping end to end, Kit-free: init from scratch, then three generations of (fake) evaluate /
propose / screen / admit for every strategy, over real grammar hands (children built with the real URDF exporter)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from coevolution.evolution import descriptors, generation
from coevolution.evolution.archive import Archive, HandRecord
from coevolution.evolution.config import EvolutionConfig
from coevolution.evolution.select import dgm_weights

ROOT = Path(__file__).resolve().parents[3]
GRAMMAR = ROOT / "assets/urdf/27oct_grammar_random_hands"
SEEDS = [f"coarse_s0_{i:03d}" for i in (0, 5, 13, 25, 26, 40)]


def _seed_manifest(tmp: Path) -> Path:
    specs = [str((GRAMMAR / n / f"{n}.spec.json").relative_to(ROOT)) for n in SEEDS]
    p = tmp / "seed_manifest.json"
    p.write_text(json.dumps({"hands": [{"spec": s} for s in specs]}))
    return p


def _fake_eval(run: Path, manifest: Path, out: Path, g: int) -> Path:
    """A fitness file for a manifest: more joints, more goals, plus a generation-dependent nudge."""
    arch = Archive.load(run / "archive.json")
    by_spec = arch.by_spec()
    hands = json.loads(manifest.read_text())["hands"]
    out.write_text(json.dumps({"hands": {h["spec"]: {
        "goals_per_episode": float(descriptors.summary(by_spec[h["spec"]].design)["joints"]) + 0.1 * g,
        "reward_per_step": 0.01} for h in hands}}))
    return out


def test_dgm_weights_favour_fitness_and_penalise_children():
    cfg = EvolutionConfig()
    a = HandRecord(id="a", spec="a", design={}, born=0, status="active", fitness=[{"value": 10.0}])
    b = HandRecord(id="b", spec="b", design={}, born=0, status="active", fitness=[{"value": 2.0}])
    c = HandRecord(id="c", spec="c", design={}, born=0, status="active", fitness=[{"value": 10.0}],
                   children=["x", "y", "z"])
    w = dgm_weights([a, b, c], cfg)
    assert w[0] > w[1] > 0 and w[0] == pytest.approx(4 * w[2])


def test_niche_and_vector_read_the_design():
    d = json.loads((GRAMMAR / "coarse_s0_000/coarse_s0_000.design.json").read_text())
    assert descriptors.niche(d) == "f6_cut_j1"                     # 6 fingers, a palm cut, 19 joints: 3.2 a finger
    v = descriptors.genetic_vector(d)
    assert v.shape == (10,) and descriptors.novelty(v, [v]) == 0.0


@pytest.mark.parametrize("strategy", ["archive", "map_elites", "greedy", "random", "static"])
def test_three_generations_from_scratch(tmp_path, strategy):
    cfg = EvolutionConfig(strategy=strategy, population=6, children=4, admitted=2, seed=1)
    cfg_path = tmp_path / "config.json"
    cfg.save(cfg_path)
    run = tmp_path / "run"
    generation.main(["init", "--run-dir", str(run), "--config", str(cfg_path),
                     "--seed-manifest", str(_seed_manifest(tmp_path))])
    state = json.loads((run / "state.json").read_text())
    assert state["checkpoint"] == "" and state["generation"] == 0
    first = json.loads((run / "gen_0000/manifest.json").read_text())["hands"]
    if strategy == "map_elites":                    # each seed hand its niche's elite (the 6 seeds: 5 niches)
        assert len(state["map"]) == len(first) == len({descriptors.niche(Archive.load(run / "archive.json")[h["id"]].design)
                                                       for h in first})
    else:
        assert len(first) == len(SEEDS)
    for g in range(3):
        gdir = run / f"gen_{g:04d}"
        fit = _fake_eval(run, gdir / "manifest.json", gdir / "fitness.json", g)
        generation.main(["propose", "--run-dir", str(run), "--fitness", str(fit)])
        generation.main(["propose", "--run-dir", str(run), "--fitness", str(fit)])   # resubmitted: a no-op
        arch = Archive.load(run / "archive.json")
        cands = arch.with_status("candidate")
        if strategy == "static":                    # RL only: no children, the population unchanged
            assert not cands
        else:
            assert 1 <= len(cands) <= 4 and all((c.parent is None) == (strategy == "random") for c in cands)
        assert len({r.key for r in arch.records.values()}) == len(arch)            # never a duplicate
        args = ["admit", "--run-dir", str(run), "--checkpoint", f"gen{g}.pth"]
        if cfg.screened:
            scr = _fake_eval(run, gdir / "children_manifest.json", gdir / "screen.json", g)
            args += ["--screen-fitness", str(scr)]
        generation.main(args)
        arch = Archive.load(run / "archive.json")
        state = json.loads((run / "state.json").read_text())
        active = arch.with_status("active")
        nxt = json.loads((run / f"gen_{g + 1:04d}/manifest.json").read_text())["hands"]
        assert sorted(h["id"] for h in nxt) == sorted(r.id for r in active) and not arch.with_status("candidate")
        if strategy == "static":
            assert sorted(h["id"] for h in nxt) == sorted(h["id"] for h in first)
        elif strategy == "map_elites":
            # the population is the map's elites plus this generation's challengers; elites are one per niche
            elites = set(state["map"].values())
            assert elites <= {r.id for r in active}
            assert len({arch[i].niche for i in elites}) == len(elites)
            assert {r.id for r in active} - elites == {c.id for c in cands}
        else:
            assert len(active) == cfg.population if len(arch) >= cfg.population else True
    assert state["generation"] == 3 and state["checkpoint"] == "gen2.pth" and len(state["history"]) == 3
    if strategy == "map_elites":                    # challengers were judged: some won or lost every generation
        assert all(h.get("won", 0) + h.get("lost", 0) > 0 for h in state["history"][1:])
    generation.main(["status", "--run-dir", str(run)])


def test_a_zero_floor_admits_children_with_negative_zero_shot_fitness(tmp_path):
    """The 29 Oct dry run: zero-goal children score slightly negative (reward tie-breaker); floor 0 must still admit."""
    cfg = EvolutionConfig(strategy="archive", population=6, children=4, admitted=2, seed=1)
    cfg_path = tmp_path / "config.json"
    cfg.save(cfg_path)
    run = tmp_path / "run"
    generation.main(["init", "--run-dir", str(run), "--config", str(cfg_path),
                     "--seed-manifest", str(_seed_manifest(tmp_path))])
    def neg(manifest: Path, out: Path) -> Path:
        hands = json.loads(manifest.read_text())["hands"]
        out.write_text(json.dumps({"hands": {h["spec"]: {"goals_per_episode": 0.0, "reward_per_step": -0.02}
                                             for h in hands}}))
        return out
    g0 = run / "gen_0000"
    generation.main(["propose", "--run-dir", str(run), "--fitness", str(neg(g0 / "manifest.json", g0 / "fitness.json"))])
    generation.main(["admit", "--run-dir", str(run), "--checkpoint", "g0.pth", "--screen-fitness",
                     str(neg(g0 / "children_manifest.json", g0 / "screen.json"))])
    assert len(json.loads((run / "state.json").read_text())["history"][0]["admitted"]) == 2
