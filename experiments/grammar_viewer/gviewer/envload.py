"""Load the simulator's numpy-only envelope modules without Isaac.

`grammar_envelope.py` (viability oracle), `palm_calibration.py` (its one
constant) and `evolution/archive.py` (MAP-Elites descriptor bins) import only
numpy, the stdlib and `hand_sampler`. Importing them the normal way runs
`isaacsimenvs/__init__.py`, which imports every task and therefore Isaac Lab.
This module loads the three files by path instead, under their real dotted
names, after placing stub package entries for `isaacsimenvs`,
`isaacsimenvs.inhand_reorient`, `.scene` and `.evolution` in `sys.modules`.
No package `__init__` runs, and the relative import
`from ..palm_calibration import ...` inside `grammar_envelope.py` resolves to
the module loaded here.

If the real `isaacsimenvs` package is already imported (an Isaac process),
the normal import is used instead.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
_ISAAC_PKG = REPO_ROOT / "isaacsimenvs"

_STUB_PACKAGES = {
    "isaacsimenvs": _ISAAC_PKG,
    "isaacsimenvs.inhand_reorient": _ISAAC_PKG / "inhand_reorient",
    "isaacsimenvs.inhand_reorient.scene": _ISAAC_PKG / "inhand_reorient" / "scene",
    "isaacsimenvs.inhand_reorient.evolution": _ISAAC_PKG / "inhand_reorient" / "evolution",
}

_FILES = {
    "isaacsimenvs.inhand_reorient.palm_calibration": _ISAAC_PKG / "inhand_reorient" / "palm_calibration.py",
    "isaacsimenvs.inhand_reorient.scene.grammar_envelope": _ISAAC_PKG / "inhand_reorient" / "scene" / "grammar_envelope.py",
    "isaacsimenvs.inhand_reorient.evolution.archive": _ISAAC_PKG / "inhand_reorient" / "evolution" / "archive.py",
}


@dataclass(frozen=True)
class EnvModules:
    grammar_envelope: types.ModuleType
    palm_calibration: types.ModuleType
    archive: types.ModuleType
    stubbed: bool


_CACHE: dict = {}


def _real_package_loaded() -> bool:
    mod = sys.modules.get("isaacsimenvs")
    return mod is not None and not getattr(mod, "__gviewer_stub__", False) and getattr(mod, "__file__", None)


def _ensure_stub(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    stub = types.ModuleType(name)
    stub.__path__ = [str(path)]  # a package: submodules may be found under it
    stub.__file__ = None
    stub.__gviewer_stub__ = True
    stub.__doc__ = f"grammar_viewer stub for {name}: its __init__ was deliberately not run (no Isaac Lab)."
    sys.modules[name] = stub
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, stub)


def _load_by_path(name: str, path: Path) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, module)
    return module


def load_env_modules() -> EnvModules:
    """Return the three modules, loading them once per process."""
    if "mods" in _CACHE:
        return _CACHE["mods"]
    if _real_package_loaded():
        mods = EnvModules(
            grammar_envelope=importlib.import_module("isaacsimenvs.inhand_reorient.scene.grammar_envelope"),
            palm_calibration=importlib.import_module("isaacsimenvs.inhand_reorient.palm_calibration"),
            archive=importlib.import_module("isaacsimenvs.inhand_reorient.evolution.archive"),
            stubbed=False,
        )
    else:
        for name, path in _STUB_PACKAGES.items():
            _ensure_stub(name, path)
        pc = _load_by_path("isaacsimenvs.inhand_reorient.palm_calibration",
                           _FILES["isaacsimenvs.inhand_reorient.palm_calibration"])
        ge = _load_by_path("isaacsimenvs.inhand_reorient.scene.grammar_envelope",
                           _FILES["isaacsimenvs.inhand_reorient.scene.grammar_envelope"])
        ar = _load_by_path("isaacsimenvs.inhand_reorient.evolution.archive",
                           _FILES["isaacsimenvs.inhand_reorient.evolution.archive"])
        mods = EnvModules(grammar_envelope=ge, palm_calibration=pc, archive=ar, stubbed=True)
    _CACHE["mods"] = mods
    return mods
