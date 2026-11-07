"""Campaign-level config pins: keep a running campaign on the settings it started with when a yaml default changes.

A co-evolution run trains each generation on `env.hand.robot_spec=multi:template:<campaign>/<run>/gen_XXXX/manifest.json`.
If `<run>/TRAIN_PINS.json` or `<campaign>/TRAIN_PINS.json` exists (the run's file wins), its {dotted.key: value} entries
are applied to the agent config -- unless the key was set explicitly on the Hydra command line, which always wins.
Only `agent.*` keys are supported.

6 Nov 2026: SAPG's expl_reward_coef_scale default went 0.002 -> 0.001 (v2); debug_outputs/coevolve/v1/TRAIN_PINS.json
keeps the v1 campaign at 0.002 so its arms stay comparable.
"""
from __future__ import annotations

import json
from pathlib import Path


def pin_files(robot_spec: str) -> list[Path]:
    """TRAIN_PINS.json files that apply to this robot spec, campaign first then run (later ones win)."""
    if not isinstance(robot_spec, str) or not robot_spec.startswith("multi:template:"):
        return []
    manifest = Path(robot_spec.split(":", 2)[2])
    run = manifest.parent.parent
    return [f for f in (run.parent / "TRAIN_PINS.json", run / "TRAIN_PINS.json") if f.exists()]


def apply_pins(robot_spec: str, agent_cfg: dict, cli_overrides: list[str]) -> dict:
    """Apply the pins to agent_cfg in place; returns {key: value} actually applied."""
    pins: dict = {}
    for f in pin_files(robot_spec):
        pins.update(json.loads(f.read_text()))
    explicit = {o.split("=", 1)[0].lstrip("+~") for o in cli_overrides if "=" in o}
    applied = {}
    for key, value in pins.items():
        if not key.startswith("agent."):
            raise ValueError(f"TRAIN_PINS: only agent.* keys are supported, got {key}")
        if key in explicit:
            continue
        node = agent_cfg
        parts = key.split(".")[1:]
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"TRAIN_PINS: {key} is not in the agent config")
        node[parts[-1]] = value
        applied[key] = value
    return applied
