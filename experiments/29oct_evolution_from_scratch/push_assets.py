"""Put every evolved hand on master, where the wandb pose viewer fetches hand meshes from (raw.githubusercontent).

    .venv_isaacsim/bin/python experiments/29oct_evolution_from_scratch/push_assets.py [run ...]

Copies ``debug_outputs/evolution/<run>/hands/<id>/*`` (URDF, spec, design, plate meshes) and the runs' generation
manifests to ``assets/urdf/29oct_evolution/<run>/`` in ONE commit on top of origin/master, made with a temporary
index (git read-tree / update-index / write-tree / commit-tree): no checkout, so the working tree and the branch
the running jobs read are never touched. Pushes, retrying from a fresh fetch if another push got there first (the two
runs call this at the end of every generation). A no-op when master already has everything. Default runs: baseline,
map_elites.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = "assets/urdf/29oct_evolution"


def git(*args, env=None, inp=None) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, input=inp).stdout.strip()


def files_for(run: str) -> list[tuple[Path, str]]:
    src = REPO / "debug_outputs/evolution" / run
    out = [(f, f"{DEST}/{run}/{f.relative_to(src / 'hands')}") for f in sorted((src / "hands").rglob("*"))
           if f.is_file()]
    out += [(m, f"{DEST}/{run}/manifests/{m.parent.name}.json") for m in sorted(src.glob("gen_*/manifest.json"))]
    return out


def push(runs: list[str], tries: int = 5) -> None:
    for attempt in range(tries):
        git("fetch", "-q", "origin", "master")
        base = git("rev-parse", "origin/master")
        with tempfile.TemporaryDirectory() as tmp:
            env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
            git("read-tree", base, env=env)
            n = 0
            for run in runs:
                for src, dst in files_for(run):
                    blob = git("hash-object", "-w", str(src))
                    git("update-index", "--add", "--cacheinfo", f"100644,{blob},{dst}", env=env)
                    n += 1
            tree = git("write-tree", env=env)
        if tree == git("rev-parse", f"{base}^{{tree}}"):
            print(f"[push_assets] master already has all {n} files")
            return
        msg = (f"{DEST}: the evolved hands so far ({', '.join(runs)})\n\nURDF, spec, design and plate meshes of every hand "
               "the 29 Oct evolution runs have built, and their generation manifests, so the wandb pose viewer can "
               "fetch them. Written by experiments/29oct_evolution_from_scratch/push_assets.py (branch "
               "embodiment_evolution).\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
               "Claude-Session: https://claude.ai/code/session_01VEjfLD7vkNCwPYPC2HsL71\n")
        commit = git("commit-tree", tree, "-p", base, inp=msg)
        r = subprocess.run(["git", "push", "-q", "origin", f"{commit}:refs/heads/master"], cwd=REPO,
                           capture_output=True, text=True)
        if r.returncode == 0:
            local = git("rev-parse", "master")
            if subprocess.run(["git", "merge-base", "--is-ancestor", local, commit], cwd=REPO).returncode == 0:
                git("update-ref", "refs/heads/master", commit, local)        # local master follows, never rewinds
            changed = git("diff", "--name-only", base, commit).count("\n") + 1
            print(f"[push_assets] pushed {commit[:7]} to master: {changed} files changed ({n} in total)")
            return
        print(f"[push_assets] push rejected (attempt {attempt + 1}): {r.stderr.strip()[:200]}; refetching")
        time.sleep(5 + 5 * attempt)
    raise SystemExit("[push_assets] gave up after repeated rejected pushes")


if __name__ == "__main__":
    push(sys.argv[1:] or ["baseline", "map_elites"])
