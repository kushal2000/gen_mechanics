"""Rebuild a training run's env config, so an eval scores what actually trained.

Every tool in this package has to reproduce the environment some checkpoint was
trained in. Getting that wrong is not a subtle error: the observation width
changes and the checkpoint refuses to load, or -- worse -- it loads and the
policy reads a differently-shaped world.

One field list, imported by every consumer, because the failure this prevents
already happened. The viewer built its config from ``PoseReachMultiEnvCfg``
defaults while the population eval copied the run's, and the two disagreed by 22
observation dimensions (307 vs 329); the checkpoint's state_dict simply would
not load. Two places deciding what "the run's config" means is one place too
many.

What is copied and what is NOT is the whole design:

* **Copied**: assets, actuation, observation layout, reward shaping. These
  define the embodiment, its object and the control pipeline -- the things that
  must match training or the checkpoint is being evaluated on a different task.
* **Not copied**: goals, resets, termination, domain randomization. These are the
  eval PROTOCOL, and they come from ``genmech/eval/suites.py`` so that results
  sit alongside everything else in ``results/``. Taking them from the run would
  score each checkpoint under its own private conditions and make runs
  incomparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

RUN_FIELDS = (
    # assets: the embodiment, its object pool, and their physical properties
    "assets.handle_head_types",
    "assets.num_assets_per_type",
    "assets.object_seed",
    "assets.shuffle_assets",
    "assets.object_density_scale",
    "assets.object_friction",
    "assets.object_restitution",
    "assets.modify_asset_frictions",
    "assets.author_object_usds",
    "assets.author_which",
    "assets.author_robot_usds",
    "assets.robot_population_path",
    "assets.robot_population_seed",
    "assets.robot_population_count",
    "assets.robot_friction",
    "assets.finger_tip_friction",
    # actuation: what an action means
    "action.arm_moving_average",
    "action.hand_moving_average",
    "action.dof_speed_scale",
    # observation layout: what the network's input vector IS
    "obs.state_list",
    "obs.obs_list",
    "obs.clamp_abs_observations",
    # reward shaping: not used for scoring, but it sets object_base_size, which
    # the object_scales observation is normalized by
    "reward.keypoint_rew_scale",
    "reward.keypoint_scale",
    "reward.object_base_size",
    # episode length bounds the rollout; success_tolerance is the curriculum
    # START, which eval_success_tolerance later pins over
    "termination.episode_length",
    "termination.success_tolerance",
)


def set_by_path(cfg: Any, dotted: str, value: Any) -> None:
    """Set a nested configclass field, failing loudly on a typo."""
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not hasattr(node, part):
            raise KeyError(f"no config section {part!r} in {dotted!r}")
        node = getattr(node, part)
    if not hasattr(node, parts[-1]):
        raise KeyError(f"no config field {dotted!r}")
    setattr(node, parts[-1], value)


def get_by_path(node: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if isinstance(node, dict):
            if part not in node:
                raise KeyError(dotted)
            node = node[part]
        else:
            node = getattr(node, part)
    return node


def load_run_config(run_dir: Path):
    """A run's hydra config as an OmegaConf node, resolvers registered.

    Kept as OmegaConf rather than a plain dict because the rl_games block
    interpolates back out to ``env`` (``num_actors`` is
    ``${....env.scene.num_envs}``), so the two halves only resolve together.
    """
    from genmech.eval import rl_player_utils  # noqa: F401  registers resolvers
    from omegaconf import OmegaConf

    path = Path(run_dir) / ".hydra" / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"no {path}; is {run_dir} a hydra run directory? Runs that died "
            "during scene setup often have no .hydra at all.")
    return OmegaConf.load(path)


def run_env_dict(run_cfg) -> dict:
    """The run's ``env`` section, interpolations resolved."""
    from omegaconf import OmegaConf
    return OmegaConf.to_container(run_cfg.env, resolve=True)


def apply_run_fields(cfg: Any, run_env: dict, verbose: bool = True) -> list[str]:
    """Copy :data:`RUN_FIELDS` from a run's env config onto ``cfg``.

    Returns the fields that were absent from the run config, which is normal for
    older runs and worth printing rather than swallowing.
    """
    missing = []
    for field in RUN_FIELDS:
        try:
            value = get_by_path(run_env, field)
        except (KeyError, AttributeError):
            missing.append(field)
            continue
        if isinstance(value, list):
            value = tuple(value)
        set_by_path(cfg, field, value)
    if missing and verbose:
        print(f"[run_config] run has no {missing}; keeping cfg defaults")
    return missing


def synthesise_policy_config(run_cfg, out_path: Path, num_actors: int) -> str:
    """Write the rl_games config RlPlayer wants, from the run's own agent block.

    ``RlPlayer`` reads ``cfg["train"]``; a run stores the identical block under
    ``agent``. Resolve against the WHOLE config, not the agent block alone --
    ``num_actors`` interpolates out to ``env.scene.num_envs``, so an agent-only
    save leaves a dangling key that fails minutes later, inside a Kit boot.
    """
    from omegaconf import OmegaConf

    resolved = OmegaConf.to_container(run_cfg, resolve=True)
    agent = resolved["agent"]
    # The run's value is its training env count; an eval usually uses fewer.
    agent["params"]["config"]["num_actors"] = num_actors
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"train": agent}), str(out_path))
    return str(out_path)


def eval_protocol(dr_profile: str, goals_per_episode: int,
                  success_tolerance: float | None = None) -> dict:
    """The frozen eval protocol, with the knobs an embodiment sweep varies.

    ``success_tolerance`` pins ``termination.eval_success_tolerance``, which
    overrides the curriculum outright (``termination_utils.py``: "Eval pins the
    success criterion"). The suite's own value is 0.01 m -- the curriculum FLOOR,
    the tightest bar training ever sets. A mid-training checkpoint has usually
    not tightened that far, so scoring it there collapses most designs to zero
    and destroys the ranking resolution an embodiment sweep exists to produce.
    """
    from genmech.eval.suites import DR_PROFILES, NOMINAL, resolve_overrides

    protocol = {k: v for k, v in resolve_overrides(NOMINAL).items()
                if not k.startswith("domain_randomization.")}
    protocol.update(DR_PROFILES[dr_profile])
    protocol["termination.max_consecutive_successes"] = goals_per_episode
    if success_tolerance is not None:
        protocol["termination.eval_success_tolerance"] = float(success_tolerance)
    return protocol


__all__ = [
    "RUN_FIELDS",
    "apply_run_fields",
    "eval_protocol",
    "get_by_path",
    "load_run_config",
    "run_env_dict",
    "set_by_path",
    "synthesise_policy_config",
]
