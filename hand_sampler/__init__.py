"""hand_sampler: the hand design space, and the robots it is measured against."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]

ASSETS_DIR: Path = REPO_ROOT / "assets"


# Lives here because every layer needs repo-relative asset paths.
def resolve(relpath: str | Path) -> Path:
    """Resolve a repo-relative path; absolute paths pass through unchanged."""
    p = Path(relpath)
    return p if p.is_absolute() else REPO_ROOT / p


__all__ = ["ASSETS_DIR", "REPO_ROOT", "resolve"]
