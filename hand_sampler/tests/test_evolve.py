"""Selection keeps the best, mutates each survivor once, and says what it did."""

from __future__ import annotations

import json

import pytest

from hand_sampler import evolve, gen_init_pop, validate_design


@pytest.fixture(scope="module")
def hands():
    return gen_init_pop.seed_population(0, 64)


def _rewards(n, fn):
    return {i: {"return_mean": fn(i), "episodes": 24, "success_rate": 0.0} for i in range(n)}


def test_the_best_survive_and_the_worst_do_not(hands):
    rewards = _rewards(64, lambda i: float(i))          # design 63 is best
    new, sel = evolve.next_generation(hands, rewards, keep=32, seed=0)
    assert len(new) == 64
    assert sel["survivors"] == list(range(63, 31, -1))
    assert set(sel["culled"]) == set(range(32))
    assert new[:32] == [hands[i] for i in sel["survivors"]]


def test_every_child_is_one_mutation_of_its_survivor(hands):
    rewards = _rewards(64, lambda i: float(-i))
    new, sel = evolve.next_generation(hands, rewards, keep=32, seed=1)
    children = new[32:]
    assert len(children) == 32
    for k, child in enumerate(children):
        parent = hands[sel["children"][k]["parent"]]
        assert sel["children"][k]["parent"] == sel["survivors"][k]
        if not sel["children"][k]["copied"]:
            assert child != parent, f"child {k} is its parent but was not marked copied"
        assert not validate_design.check(child)


def test_unscored_designs_rank_last(hands):
    rewards = _rewards(64, lambda i: 1.0)
    del rewards[5], rewards[40]
    _, sel = evolve.next_generation(hands, rewards, keep=32, seed=0)
    assert sel["ranking"][-2:] == [5, 40]
    assert 5 in sel["culled"] and 40 in sel["culled"]
    assert sel["unscored"] == [5, 40]


def test_it_is_deterministic_in_the_seed(hands):
    rewards = _rewards(64, lambda i: (i * 7919) % 64)
    a, _ = evolve.next_generation(hands, rewards, keep=32, seed=42)
    b, _ = evolve.next_generation(hands, rewards, keep=32, seed=42)
    c, _ = evolve.next_generation(hands, rewards, keep=32, seed=43)
    assert a == b
    assert a != c


def test_the_selection_record_is_complete(hands):
    rewards = _rewards(64, lambda i: float(i % 10))
    _, sel = evolve.next_generation(hands, rewards, keep=32, seed=0)
    assert sorted(sel["survivors"] + sel["culled"]) == list(range(64))
    assert len(sel["children"]) == 32
    assert sel["summary"]["survivor_mean"] >= sel["summary"]["culled_mean"]
    json.dumps(sel)                                          # serialisable


def test_merge_sums_across_ranks(tmp_path):
    from coevolution.design_rewards import merge_rank_files

    a = {"designs": {"0": {"episodes": 2, "return_sum": 10.0, "goals_sum": 3.0, "succeeded": 1},
                     "1": {"episodes": 1, "return_sum": 3.0, "goals_sum": 0.0, "succeeded": 0}}}
    b = {"designs": {"0": {"episodes": 2, "return_sum": 6.0, "goals_sum": 2.0, "succeeded": 1}}}
    (tmp_path / "r0.json").write_text(json.dumps(a))
    (tmp_path / "r1.json").write_text(json.dumps(b))
    m = merge_rank_files([tmp_path / "r0.json", tmp_path / "r1.json"])
    assert m[0]["episodes"] == 4 and m[0]["return_mean"] == pytest.approx(4.0)
    assert m[0]["success_rate"] == pytest.approx(0.5)          # 2 of 4 episodes reached a goal
    assert m[0]["goals_per_episode"] == pytest.approx(1.25)   # 5 goals over 4 episodes
    assert m[1]["return_mean"] == pytest.approx(3.0)
