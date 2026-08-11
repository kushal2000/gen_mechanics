"""Repo-root resolution.

Replaces ``isaacgymenvs.utils.utils.get_repo_root_dir``. Asset paths in the
configs are repo-relative, so everything resolves against REPO_ROOT rather than
the process CWD.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT: Path = Path(__file__).resolve().parents[2]

ASSETS_DIR: Path = REPO_ROOT / "assets"


def resolve(relpath: str | Path) -> Path:
    """Resolve a repo-relative path; absolute paths pass through unchanged."""
    p = Path(relpath)
    return p if p.is_absolute() else REPO_ROOT / p


__all__ = ["REPO_ROOT", "ASSETS_DIR", "resolve"]
