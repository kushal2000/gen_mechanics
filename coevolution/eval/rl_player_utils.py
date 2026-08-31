from dataclasses import dataclass
from typing import Optional

import yaml
from gym import spaces
from omegaconf import DictConfig, OmegaConf

from coevolution.utils.reformat import omegaconf_to_dict, print_dict


def _register_legacy_resolvers() -> None:
    """Register the OmegaConf resolvers that training-run configs interpolate.

    rl_games checkpoints ship the Hydra config they were trained under, and
    those files contain interpolations like ``${resolve_default:...}`` and
    ``${eval:"1/60"}``. In simtoolreal the resolvers behind them were registered
    as a side effect of importing ``isaacgymenvs``; genmech does not import that
    package, so reading such a config raises UnsupportedInterpolationType at the
    first ``.items()``.

    Registering them here keeps every historical checkpoint loadable. The
    definitions are copied from isaacgymenvs/__init__.py.
    """
    from omegaconf import OmegaConf

    resolvers = {
        "eq": lambda x, y: x.lower() == y.lower(),
        "contains": lambda x, y: x.lower() in y.lower(),
        "if": lambda pred, a, b: a if pred else b,
        "resolve_default": lambda default, arg: default if arg == "" else arg,
        "eval": lambda x: eval(x),  # noqa: S307 - matches the upstream definition
    }
    for name, fn in resolvers.items():
        OmegaConf.register_new_resolver(name, fn, replace=True)


_register_legacy_resolvers()


@dataclass
class DummyEnv:
    observation_space: spaces.Box
    action_space: spaces.Box

    def get_env_info(self) -> dict:
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
        }


def read_cfg(config_path: str, device: Optional[str] = None) -> dict:
    # Read in config
    # Change device if needed

    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Convert to OmegaConf to process interpolations
    omegaconf_cfg = OmegaConf.create(raw_cfg)

    # Need to convert back to dict
    cfg = omegaconf_to_dict(omegaconf_cfg)

    # Modify device manually
    if device is not None:
        if (
            "train" in cfg
            and "params" in cfg["train"]
            and "config" in cfg["train"]["params"]
        ):
            # OLD
            cfg["train"]["params"]["config"]["device"] = device
            cfg["train"]["params"]["config"]["device_name"] = device
        else:
            # NEW
            cfg["rl_device"] = device
            cfg["sim_device"] = device

    print("-" * 80)
    print("CONFIGURATION")
    print_dict(cfg)
    print("-" * 80 + "\n")

    return cfg


def read_cfg_omegaconf(config_path: str, device: Optional[str] = None) -> DictConfig:
    # Read in config
    # Change device if needed

    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # Convert to OmegaConf to process interpolations
    omegaconf_cfg = OmegaConf.create(raw_cfg)

    # Modify device manually
    if device is not None:
        if (
            "train" in omegaconf_cfg
            and "params" in omegaconf_cfg.train
            and "config" in omegaconf_cfg.train.params
        ):
            # OLD
            omegaconf_cfg.train.params.config.device = device
            omegaconf_cfg.train.params.config.device_name = device
            omegaconf_cfg.rl_device = device
            omegaconf_cfg.sim_device = device
        else:
            # NEW
            omegaconf_cfg.rl_device = device
            omegaconf_cfg.sim_device = device

    print("-" * 80)
    print("CONFIGURATION")
    print_dict(omegaconf_cfg)
    print("-" * 80 + "\n")

    return omegaconf_cfg
