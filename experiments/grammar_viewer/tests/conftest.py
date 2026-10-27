import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIEWER_DIR = HERE.parent
REPO_ROOT = VIEWER_DIR.parents[1]
for p in (str(VIEWER_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
