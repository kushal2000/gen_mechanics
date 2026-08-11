"""Import every module, so a stale reference fails in seconds instead of hours.

The SHARPA training run died 22 minutes in -- after the full 24576-env scene
build -- on a one-line stale import in the pose viewer that no other test
touched. Kit is expensive to boot and expensive to be wrong about, so this test
walks the whole package and imports each module, catching renamed or deleted
symbols before any GPU time is spent.

Modules that legitimately require a booted Kit are imported after AppLauncher.

    .venv_isaacsim/bin/python tests/test_imports.py
"""

from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    # Kit first: most of genmech.tasks needs isaaclab sub-namespaces resolved.
    app = AppLauncher(args).app

    import genmech

    failures: list[tuple[str, str]] = []
    checked = 0
    for mod in pkgutil.walk_packages(genmech.__path__, prefix="genmech."):
        name = mod.name
        # Entry-point scripts parse argv at import time; skip them.
        if name.rsplit(".", 1)[-1] in ("train", "play_video", "run_eval",
                                       "aggregate", "build_allegro_urdf",
                                       "reachability_viewer"):
            continue
        try:
            importlib.import_module(name)
            checked += 1
        except Exception as exc:  # noqa: BLE001 - report every failure, not the first
            failures.append((name, f"{type(exc).__name__}: {exc}"))

    for name, err in failures:
        print(f"[imports] FAIL {name}\n          {err}")
    print(f"[imports] {checked} modules imported, {len(failures)} failed")

    if failures:
        raise AssertionError(f"{len(failures)} module(s) failed to import")

    print("[imports] module import test OK")
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
