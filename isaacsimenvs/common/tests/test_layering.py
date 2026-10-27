"""The import layering: hand_sampler <- isaacsimenvs.common <- the tasks. Kit-free.

``common`` is shared by every task, so it may import none of them; ``hand_sampler`` imports no simulator code at all;
and the Kit-free half of ``common`` (the robot registry, the observation layout, Sdf helpers, object authoring) must
import without booting Isaac Sim, which is what lets the offline tools and viewers use it.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TASKS = ("isaacsimenvs.pose_reaching_6d", "isaacsimenvs.inhand_reorient", "isaacsimenvs.inhand_olympics")
# Known exceptions, each a lazy import inside one function. The legacy old-genotype viewer grafts a design onto the
# arm with coevolution's viewer helpers (only for its interactive view; its PNG drawings import nothing upward).
ALLOWED = {("hand_sampler/viewer.py", "coevolution.pose_viewer")}


def _imports(path: Path) -> list[tuple[int, str]]:
    """Absolute module names a file imports (relative imports resolved against its package)."""
    pkg = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = pkg.rsplit(".", node.level - 1)[0] if node.level else ""
            mod = ".".join(p for p in (base, node.module) if p)
            out.append((node.lineno, mod))
    return out


def _sources(folder: str) -> list[Path]:
    return [p for p in sorted((ROOT / folder).rglob("*.py")) if "tests" not in p.relative_to(ROOT).parts]


@pytest.mark.parametrize("folder, banned", [
    ("isaacsimenvs/common", TASKS),
    ("hand_sampler", ("isaacsimenvs", "coevolution", "isaaclab", "pxr", "omni")),
])
def test_no_upward_imports(folder, banned):
    bad = [f"{p.relative_to(ROOT)}:{line} imports {mod}"
           for p in _sources(folder) for line, mod in _imports(p)
           if any(mod == b or mod.startswith(b + ".") for b in banned)
           and (str(p.relative_to(ROOT)), mod) not in ALLOWED]
    assert not bad, "\n".join(bad)


def test_kit_free_modules_import_without_kit():
    code = ("import sys\n"
            "import isaacsimenvs.common.robots, isaacsimenvs.common.obs_layout, isaacsimenvs.common.sdf\n"
            "import isaacsimenvs.common.author_objects\n"
            "from isaacsimenvs.common.robots import get_robot_spec\n"
            "get_robot_spec('wuji2_left_uniform_handonly'); get_robot_spec('multi:uniform')\n"
            "print(sorted({m.split('.')[0] for m in sys.modules} & {'pxr', 'isaaclab', 'omni', 'isaacsim'}))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout
