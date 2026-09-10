"""A stored population must be the population, exactly.

The file exists so a design can be looked at instead of re-derived, which is
only worth anything if what comes back is what went in -- a design that
round-trips almost right sends you to inspect a hand the run never trained.
"""

from __future__ import annotations

import json

import pytest

from hand_sampler import design_space, gen_init_pop, population_io


@pytest.fixture(scope="module")
def hands():
    return gen_init_pop.seed_population(0, 64)


def test_round_trip_is_exact(hands):
    """Frozen dataclasses compare by value, so equality is the whole check."""
    for i, hand in enumerate(hands):
        assert population_io.hand_from_dict(population_io.hand_to_dict(hand)) == hand, i


def test_round_trip_through_a_file(hands, tmp_path):
    path = population_io.save_population(hands, tmp_path / "p.json", name="gen_s0_n64")
    assert population_io.load_population(path) == hands
    population_io.verify(path, hands)          # raises if it drifted


def test_verify_catches_a_drifted_file(hands, tmp_path):
    path = population_io.save_population(hands, tmp_path / "p.json", name="gen_s0_n64")
    data = json.loads(path.read_text())
    data["designs"][7]["fingers"][0]["segments"][0]["length"] += 0.001
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="design 7"):
        population_io.verify(path, hands)


def test_a_truncated_file_is_not_silently_short(hands, tmp_path):
    path = population_io.save_population(hands, tmp_path / "p.json", name="gen_s0_n64")
    data = json.loads(path.read_text())
    del data["designs"][-1]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="header says"):
        population_io.load_population(path)


def test_a_future_format_is_refused(hands, tmp_path):
    path = population_io.save_population(hands, tmp_path / "p.json", name="gen_s0_n64")
    data = json.loads(path.read_text())
    data["format"] = population_io.FORMAT_VERSION + 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="format"):
        population_io.load_population(path)


def test_optional_fields_survive(tmp_path):
    """An imported hand carries what a generated one leaves None; both must keep it."""
    rich = design_space.Hand(
        palm=design_space.Palm(0.025, 0.06, 0.06),
        fingers=(design_space.Finger(
            mount=design_space.Mount("+z", 0.0, 0.0),
            segments=(design_space.Segment(
                joint=design_space.Joint(theta=0.3, phi=1.1, offset=0.2,
                                         axis_override=(0.0, 1.0, 0.0),
                                         limits=(-0.5, 0.9),
                                         drive=(1.0, 2.0, 3.0, 4.0, 5.0)),
                length=0.04,
                cross_section=(0.018, 0.02),
                meshes=("a.stl", "b.stl"),
                token_box=((0.0, 0.0, 0.0), (0.04, 0.0, 0.0),
                           (0.0, 0.018, 0.0), (0.0, 0.0, 0.02)),
            ),),
        ),),
    )
    assert population_io.hand_from_dict(population_io.hand_to_dict(rich)) == rich


def test_defaults_may_be_omitted_by_hand(hands):
    """The file is meant to be read and edited, so absent means default."""
    minimal = {"palm": {"thickness": 0.025, "width": 0.06, "length": 0.06},
               "fingers": [{"mount": {"face": "+z", "u": 0.0, "v": 0.0},
                            "segments": [{"joint": {"theta": 0.0}, "length": 0.04}]}]}
    hand = population_io.hand_from_dict(minimal)
    joint = hand.fingers[0].segments[0].joint
    assert (joint.phi, joint.offset, joint.limits, joint.drive) == (
        design_space.Joint(0.0).phi, 0.0, None, None)


# --- a population as data, not as a function of the code ---------------------

def test_a_file_and_its_name_give_the_same_population(hands, tmp_path):
    from hand_sampler import robot_spec

    path = population_io.save_population(hands, tmp_path / "gen_s0_n64.json", name="gen_s0_n64")
    robot_spec._POPULATION_CACHE.clear()
    from_file = robot_spec.population_from_ref(str(path))
    from_name = robot_spec.population_from_ref("gen_s0_n64")
    assert from_file.hands == from_name.hands
    assert from_file.spec.joint_names_canonical == from_name.spec.joint_names_canonical
    assert from_file.n_designs == from_name.n_designs


def test_a_file_pins_the_population_against_the_sampler(tmp_path):
    """The point of the file: it stops moving when the code moves.

    seed_population rejection-samples, so a change to validate_design or the
    SEED_ constants yields different hands under the same NAME. A file is data.
    """
    from hand_sampler import robot_spec

    edited = list(gen_init_pop.seed_population(0, 8))
    edited[3] = design_space.Hand(
        palm=design_space.Palm(0.02, 0.05, 0.05),
        fingers=(design_space.Finger(
            mount=design_space.Mount("+z", 0.0, 0.0),
            segments=(design_space.Segment(design_space.Joint(0.0), 0.031),)),))
    path = population_io.save_population(edited, tmp_path / "pinned.json", name="pinned")

    robot_spec._POPULATION_CACHE.clear()
    loaded = robot_spec.population_from_ref(str(path))
    assert loaded.hands[3] == edited[3]                      # the file wins
    assert loaded.hands[3] != gen_init_pop.seed_population(0, 8)[3]


def test_a_missing_file_is_fatal_not_a_silent_resample(tmp_path):
    """Falling back to sampling would train the very thing the file pins."""
    from hand_sampler import robot_spec

    with pytest.raises(FileNotFoundError, match="population_io"):
        robot_spec.population_from_ref(str(tmp_path / "absent.json"))


def test_a_path_is_recognised_as_a_population_reference():
    from hand_sampler import robot_spec

    assert robot_spec.is_population_ref("gen_s0_n64")
    assert robot_spec.is_population_ref("assets/populations/gen_s0_n64.json")
    assert robot_spec.is_population_file("/abs/path/p.json")
    assert not robot_spec.is_population_file("gen_s0_n64")
    assert not robot_spec.is_population_ref("sharpa_iiwa14")
