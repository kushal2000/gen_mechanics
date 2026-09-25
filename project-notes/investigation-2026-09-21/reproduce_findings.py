"""Read-only CPU probes for the 2026-09-21 project investigation.

Run from the repository root:
    python ../project-notes/investigation-2026-09-21/reproduce_findings.py

These probes report current behavior; they do not change the grammar or policy.
The normalization probe uses real repository methods with a minimal token layout
to avoid importing Isaac Lab. It is not a simulation or checkpoint evaluation.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2] / "gen_mechanics"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "rl_games"))

import numpy as np

from hand_sampler import design_space as ds
from hand_sampler import evolve, gen_init_pop, population_io, validate_design
from hand_sampler import robot_param_constants as rpc


def selection_records() -> list[dict]:
    hands = gen_init_pop.seed_population(0, 8)
    rewards = {i: {"return_mean": float(i)} for i in range(8)}
    rows = []
    for keep in (2, 4, 6, 8):
        new, selection = evolve.next_generation(hands, rewards, keep=keep, seed=0)
        rows.append({
            "population": len(new), "keep": keep,
            "actual_children": len(new) - keep,
            "recorded_children": len(selection["children"]),
            "recorded_parents": [c["parent"] for c in selection["children"]],
        })
    return rows


def axis_wrap() -> dict:
    palm = ds.Palm(0.025, 0.060, 0.060)
    mount = ds.Mount("+z", 0.5, 0.5)

    def finger(theta, offset=60):
        return ds.Finger(mount, (ds.Segment(ds.Joint(
            math.radians(theta), offset=math.radians(offset)), 0.05),))

    points = {a: ds.fingertip(finger(a), palm) for a in (0, 150, 165, 180)}
    distance = lambda a, b: float(np.linalg.norm(points[a] - points[b]) * 1000)
    other = ds.Finger(ds.Mount("-y", 0.5, 0.5), (ds.Segment(ds.Joint(0), 0.03),))
    return {
        "link_mm": 50, "offset_degrees": 60,
        "ordinary_150_to_165_tip_displacement_mm": distance(150, 165),
        "wrapped_165_to_0_tip_displacement_mm": distance(165, 0),
        "unwrapped_165_to_180_tip_displacement_mm": distance(165, 180),
        "negating_offset_at_wrap_matches_unwrapped_mm": float(np.linalg.norm(
            ds.fingertip(finger(0, -60), palm) - points[180]) * 1000),
        "before_validation_errors": validate_design.check(ds.Hand(palm, (finger(165), other))),
        "after_validation_errors": validate_design.check(ds.Hand(palm, (finger(0), other))),
    }


def population_validation() -> dict:
    hand = gen_init_pop.seed_population(0, 1)[0]
    invalid = replace(hand, palm=replace(hand.palm, width=0.123))
    with tempfile.TemporaryDirectory(prefix="hand-load-probe-") as directory:
        path = Path(directory) / "invalid.json"
        population_io.save_population([invalid], path, name="validation_probe")
        loaded = population_io.load_population(path)
    return {"loaded": True, "validation_errors": validate_design.check(loaded[0])}


def normalized_mask() -> dict:
    import torch
    from rl_games.algos_torch.models import BaseModelNetwork
    from coevolution.networks.joint_transformer import JointTransformerNet

    # Only layout and projection metadata are supplied manually. norm_obs and
    # _trunk execute the repository implementations without replacing methods.
    net = JointTransformerNet.__new__(JointTransformerNet)
    torch.nn.Module.__init__(net)
    net.net_type = "simple"
    net.n_hand, net.token_dim, net.enabled_col = 3, 1, 0
    net.register_buffer("token_gather", torch.tensor([0, 1, 2]))
    net.register_buffer("global_index", torch.tensor([3]))
    net.token_proj = torch.nn.Linear(1, 4)
    net.global_proj = torch.nn.Linear(1, 4)
    net.layers = torch.nn.ModuleList()
    net.final_norm = False
    model = BaseModelNetwork(obs_shape=(4,), normalize_value=False,
                             normalize_input=True, value_size=1,
                             extra_info_start_idx=None)
    raw = torch.ones(4096, 4)
    normalized = model.norm_obs(raw)
    with torch.no_grad():
        return {
            "batch_rows": len(raw),
            "raw_enabled": raw[0, :3].tolist(),
            "normalized_enabled": normalized[0, :3].tolist(),
            "valid_before_normalization": net._trunk(raw)[3][0].tolist(),
            "valid_after_normalization": net._trunk(normalized)[3][0].tolist(),
        }


def main() -> None:
    git = "/opt/homebrew/bin/git" if Path("/opt/homebrew/bin/git").exists() else "git"
    commit = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    versions = {}
    for package in ("numpy", "torch", "pytest", "scipy", "trimesh", "yourdfpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "commit": commit, "python": platform.python_version(), "versions": versions,
        "selection_records": selection_records(), "axis_wrap": axis_wrap(),
        "population_validation": population_validation(),
        "radius_mm": {"grammar_validator_tokens": ds.CAPSULE_RADIUS * 1000,
                      "authored_collider": rpc.GEN_LINK_RADIUS_M * 1000},
    }
    try:
        # RunningMeanStd announces its size; keep stdout as valid JSON.
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            result["normalized_mask"] = normalized_mask()
    except ImportError as exc:
        result["normalized_mask"] = {"not_run_missing_dependency": str(exc)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
