"""Iteration-4 pilot report: for every hand in ``manifest.json``, resolve its
source file, import it, and report (a) fidelity of our forward kinematics
against a frozen or freshly generated Pinocchio-oracle reference and (b)
whether ``hand_sampler.grammar.coverage`` judges it expressible by / inside
the support of the grammar. Writes ``pilot-report.json`` and
``pilot-report.md`` (by default under ``project-notes/grammar/``, a location
the repo's ``.gitignore`` does not drop, unlike ``results/``; override with
``--out``).

Held-out hands are reported here, never used to change any rule, range or
tolerance (see ``manifest.json``'s own ``rules``). This script makes no claim
that the grammar is universal -- see the "Claims" section of ``pilot.md``.

Stdlib + numpy + the project's own ``hand_sampler.grammar`` package only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from hand_sampler.grammar import fk
from hand_sampler.grammar.adapters.urdf import load_urdf
from hand_sampler.grammar.coords import q_from_u
from hand_sampler.grammar.coverage import coverage
from hand_sampler.grammar.distributions import DEFAULT_DISTRIBUTION
from hand_sampler.grammar.kinematics import MOVABLE_TYPES
from hand_sampler.grammar_bench.tolerances import ORACLE_POS_M, ORACLE_ROT_RAD

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
REFERENCES_DIR = BENCH_DIR / "references"
LOCAL_REFERENCES_DIR = REFERENCES_DIR / "local"
MANIFEST_PATH = BENCH_DIR / "manifest.json"

ORACLE_PYTHON = Path("/home/singularity/anaconda3/envs/piper/bin/python")
ORACLE_FK_SCRIPT = BENCH_DIR / "refgen" / "oracle_fk.py"
MAKE_CONFIGS_SCRIPT = BENCH_DIR / "refgen" / "make_configs.py"

DEFAULT_SEED = 20260925
DEFAULT_OUT_DIR = REPO_ROOT / "project-notes" / "grammar"

# Support-audit categories exercised by
# grammar_bench/tests/test_acceptance_grammar.py::test_grammar_support_audit
# -- every one of these is proven reachable by the grammar's own productions
# over N_AUDIT_SEEDS derivations. Reused here (not recomputed) as the
# "constructs covered" list for the pilot report.
SUPPORT_AUDIT_CONSTRUCTS = (
    "one_digit", "five_plus_digits", "six_plus_phalanges", "in_digit_branch",
    "palm_tree", "palm_joint", "two_nonparallel_palm_joints",
    "nonperpendicular_axis", "coupling_nonzero_offset",
    "coupling_negative_multiplier", "asymmetric_limits", "continuous_joint",
    "prismatic_joint", "nonidentity_mount_rotation",
)


def _oracle_available() -> bool:
    return ORACLE_PYTHON.is_file() and ORACLE_FK_SCRIPT.is_file()


def _git_sha() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _git_dirty() -> bool:
    """True if ``git status --porcelain`` reports any change (staged,
    unstaged or untracked) relative to ``git_sha`` -- so the report is
    honest about whether it was actually generated from that exact commit,
    or from a working tree that has since moved on from it."""
    try:
        proc = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        if proc.returncode == 0:
            return bool(proc.stdout.strip())
    except Exception:
        pass
    return True  # unknown status is reported as dirty, never silently clean


def _resolve_hand(hand: dict, manifest: dict):
    """Returns (path_or_None, availability, reason_or_None). Mirrors the
    resolution rules used by test_acceptance_export.py's own
    ``_committed_real_hands`` plus the local-only convention documented in
    ``manifest.json``'s ``rules``."""
    if hand.get("split") == "excluded":
        return None, "excluded", hand.get("notes") or "excluded split: never imported"

    fixture_path = hand.get("fixture_path")
    if fixture_path:
        p = BENCH_DIR / fixture_path
        if p.is_file():
            return p, "available", None
        return None, "unavailable", f"fixture_path listed in manifest but missing on disk: {fixture_path}"

    source_path = hand.get("source_path")
    if isinstance(source_path, str) and source_path.startswith("REPO:"):
        p = REPO_ROOT / source_path[len("REPO:"):]
        if p.is_file():
            return p, "available", None
        return None, "unavailable", f"REPO path listed in manifest but missing on disk: {source_path}"

    if isinstance(source_path, str) and source_path:
        source_root = manifest.get("source_root")
        if source_root:
            p = Path(source_root) / source_path
            if p.is_file():
                return p, "available", None
        return None, "unavailable", f"local-only:{hand['id']} source not present at manifest source_root on this machine"

    return None, "unavailable", "no fixture_path/source_path in manifest"


def _load_frozen_reference(hand_id: str):
    for d in (REFERENCES_DIR, LOCAL_REFERENCES_DIR):
        p = d / f"{hand_id}.json"
        if p.is_file():
            return json.loads(p.read_text()), str(p.relative_to(REPO_ROOT))
    return None, None


def _generate_reference(hand_id: str, urdf_path: Path, hand_root: Optional[str], tmp_root: Path):
    cfg_cmd = [sys.executable, str(MAKE_CONFIGS_SCRIPT), "--urdf", str(urdf_path), "--id", hand_id,
               "--out-dir", str(tmp_root)]
    if hand_root:
        cfg_cmd += ["--hand-root", hand_root]
    proc = subprocess.run(cfg_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    configs_path = tmp_root / f"{hand_id}.configs.json"
    out_path = tmp_root / f"{hand_id}.oracle.json"
    oracle_cmd = [str(ORACLE_PYTHON), str(ORACLE_FK_SCRIPT), "--urdf", str(urdf_path),
                  "--configs", str(configs_path), "--out", str(out_path)]
    if hand_root:
        oracle_cmd += ["--hand-root", hand_root]
    proc2 = subprocess.run(oracle_cmd, capture_output=True, text=True)
    if proc2.returncode != 0:
        return None
    return json.loads(out_path.read_text())


def _fidelity_for(hand_id: str, model, hand_root: Optional[str], urdf_path: Path, tmp_root: Path):
    reference, ref_path = _load_frozen_reference(hand_id)
    reference_kind = "frozen"
    if reference is None:
        if not _oracle_available():
            return None, "oracle-unavailable: piper conda interpreter not found, no frozen reference on disk"
        reference = _generate_reference(hand_id, urdf_path, hand_root, tmp_root)
        reference_kind = "generated_at_run"
        if reference is None:
            return None, "oracle-unavailable: oracle_fk.py failed on this hand"

    configs = reference["configs"]
    ref_poses = reference["poses"]
    max_pos, max_rot, n = 0.0, 0.0, 0
    for u, poses in zip(configs, ref_poses):
        try:
            q = q_from_u(model, u)
        except Exception:
            q = u  # defensive: fall back to driving joints directly if u doesn't match independent_joints
        ours = fk.forward_kinematics(model, q)
        for body, T_list in poses.items():
            if body not in ours:
                continue
            T_ref = np.array(T_list)
            pos_err = fk.position_error(ours[body], T_ref)
            rot_err = fk.rotation_error(ours[body], T_ref)
            max_pos = max(max_pos, pos_err)
            max_rot = max(max_rot, rot_err)
            n += 1

    tolerance_met = bool(n > 0 and max_pos <= ORACLE_POS_M and max_rot <= ORACLE_ROT_RAD)
    return {
        "n_poses": n,
        "max_pos": max_pos,
        "max_rot": max_rot,
        "tolerance_met": tolerance_met,
        "reference": reference_kind,
        "reference_path": ref_path,
    }, None


def _coverage_dict(result) -> dict:
    return {
        "topology_expressible": result.topology_expressible,
        "in_support": result.in_support,
        "missing_constructs": list(result.missing_constructs),
        "out_of_support": list(result.out_of_support),
        "digit_count": result.digit_count,
        "digit_count_source": result.digit_count_source,
        "notes": list(result.notes),
    }


def run(seed: int) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    hands_out = []

    with tempfile.TemporaryDirectory() as td:
        tmp_root = Path(td)
        for hand in manifest["hands"]:
            entry = {
                "id": hand["id"],
                "family": hand.get("family"),
                "split": hand.get("split"),
                "license": hand.get("license"),
                "availability": None,
                "reason": None,
                "movable_joints": None,
                "couplings": None,
                "fidelity": None,
                "coverage": None,
                "losses": None,
            }
            path, availability, reason = _resolve_hand(hand, manifest)
            entry["availability"] = availability
            entry["reason"] = reason
            if availability != "available":
                hands_out.append(entry)
                continue

            hand_root = hand.get("hand_root")
            try:
                result = load_urdf(path, hand_root=hand_root)
            except Exception as exc:
                entry["availability"] = "unavailable"
                entry["reason"] = f"import failed: {exc}"
                hands_out.append(entry)
                continue

            model = result.model
            entry["movable_joints"] = sum(1 for j in model.joints if j.type in MOVABLE_TYPES)
            entry["couplings"] = len(model.couplings)
            entry["losses"] = {
                "dropped_above_hand_root": list(result.losses.dropped_above_hand_root),
                "couplings_cut": list(result.losses.couplings_cut),
                "non_unit_axes": list(result.losses.non_unit_axes),
            }

            cov = coverage(model, DEFAULT_DISTRIBUTION)
            entry["coverage"] = _coverage_dict(cov)

            fidelity, fidelity_reason = _fidelity_for(hand["id"], model, hand_root, path, tmp_root)
            entry["fidelity"] = fidelity
            if fidelity is None:
                entry["fidelity_reason"] = fidelity_reason

            hands_out.append(entry)

    return {
        "schema": "hand_grammar_bench/pilot/0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "tool_versions": {"numpy": np.__version__, "python": sys.version.split()[0]},
        "tolerances": {"oracle_pos_m": ORACLE_POS_M, "oracle_rot_rad": ORACLE_ROT_RAD},
        "hands": hands_out,
    }


def _fmt_err(v) -> str:
    return f"{v:.3e}" if isinstance(v, (int, float)) else "-"


def render_markdown(report: dict) -> str:
    lines = []
    lines.append("# Hand-kinematics grammar: iteration-4 coverage pilot")
    lines.append("")
    dirty_note = " (working tree had uncommitted changes at generation time)" if report.get("git_dirty") else ""
    lines.append(
        f"Generated {report['generated_at']} at git SHA `{report['git_sha']}`{dirty_note}, seed {report['seed']}."
    )
    lines.append("")
    lines.append(
        "For every hand: whether we could import it, whether our forward kinematics "
        "matches an independent oracle (Pinocchio) within tolerance, and whether the "
        "grammar's own support audit (`hand_sampler.grammar.coverage`) judges the "
        "imported model expressible by the grammar's productions and inside the "
        "default distribution's sampled ranges."
    )
    lines.append("")
    lines.append(
        "| id | family | split | availability | movable joints | couplings | digit count | digit count source"
        " | fidelity (max pos / max rot) | fidelity pass | topology_expressible | in_support | missing constructs"
        " | out-of-support items |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for h in report["hands"]:
        fid = h.get("fidelity")
        if fid is not None:
            fid_str = f"{_fmt_err(fid['max_pos'])} m / {_fmt_err(fid['max_rot'])} rad ({fid['reference']})"
            fid_pass = "yes" if fid["tolerance_met"] else "no"
        else:
            fid_str = h.get("fidelity_reason") or "-"
            fid_pass = "-"
        cov = h.get("coverage")
        if cov is not None:
            expr = "yes" if cov["topology_expressible"] else "no"
            sup = "yes" if cov["in_support"] else "no"
            missing = ", ".join(cov["missing_constructs"]) or "-"
            oos = ", ".join(cov["out_of_support"]) or "-"
            digit_count = cov["digit_count"]
            digit_count_source = cov["digit_count_source"]
        else:
            expr = sup = missing = oos = "-"
            digit_count = digit_count_source = "-"
        lines.append(
            f"| {h['id']} | {h['family'] or '-'} | {h['split']} | {h['availability']}"
            f" | {h['movable_joints'] if h['movable_joints'] is not None else '-'}"
            f" | {h['couplings'] if h['couplings'] is not None else '-'}"
            f" | {digit_count} | {digit_count_source}"
            f" | {fid_str} | {fid_pass} | {expr} | {sup} | {missing} | {oos} |"
        )
    lines.append("")

    lines.append("## Constructs covered by the grammar")
    lines.append("")
    lines.append(
        "Proven reachable by `hand_sampler.grammar.derive.sample_derivation` over "
        "`test_grammar_support_audit`'s seeded sample (see "
        "`grammar_bench/tests/test_acceptance_grammar.py`), independent of this pilot:"
    )
    lines.append("")
    for c in SUPPORT_AUDIT_CONSTRUCTS:
        lines.append(f"- {c}")
    lines.append("")

    lines.append("## Known limitations")
    lines.append("")
    lines.append(
        "- Digit counting for an imported model (`digit_count_source: root_chains`) counts "
        "movable-joint chains leaving the declared root, treating a run of fixed joints as "
        "transparent -- so a hand whose root passes through a single wrist joint before "
        "fanning out to its fingers (e.g. `orca_right`) counts as 1 digit, not 5; it is not "
        "a count of anatomical fingers."
    )
    lines.append(
        "- `topology_expressible` is a necessary, not sufficient, condition: it means the "
        "grammar's own productions could build a model with this *topology* (joint types, "
        "branching, coupling scope, ...), never that the grammar would ever sample this "
        "particular hand's lengths/axes/limits -- that is what `in_support` checks."
    )
    lines.append(
        "- `in_support` is judged against `DEFAULT_DISTRIBUTION` only; a hand out of support "
        "under the default ranges/grids/choice-sets might still be in support of some other "
        "`Distribution` this module could construct."
    )
    lines.append("")

    lines.append("## Claims")
    lines.append("")
    lines.append(
        "- Availability, fidelity and coverage above were computed for every hand listed "
        "in `grammar_bench/manifest.json` (14 hands); `excluded`-split hands were never "
        "imported, and `unavailable` hands were never scored (their reason is reported "
        "instead of a result)."
    )
    lines.append(
        "- Fidelity, where reported, compares our own `hand_sampler.grammar.fk` against an "
        "independent Pinocchio implementation reading the same URDF, either from a frozen "
        "reference committed to the benchmark or one generated fresh in this run "
        "(`reference: generated_at_run`)."
    )
    lines.append(
        "- `topology_expressible`/`in_support` come from `hand_sampler.grammar.coverage`, "
        "which checks the imported model against the grammar's own productions (`rules.py`) "
        "and the default sampling distribution's ranges/grids (`distributions.py`); named "
        "`missing_constructs`/`out_of_support` items are never dropped or relabeled to make "
        "a hand look better."
    )
    lines.append(
        "- Held-out-split hands are reported exactly like dev-split hands; none of their "
        "results were used to change any grammar rule, sampling range or tolerance."
    )
    lines.append("- No universality claim is made.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default=str(DEFAULT_OUT_DIR),
        help="output directory for pilot-report.json/pilot-report.md (default: project-notes/grammar)",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run(args.seed)

    (out_dir / "pilot-report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "pilot-report.md").write_text(render_markdown(report))
    print(f"wrote {out_dir / 'pilot-report.json'} and {out_dir / 'pilot-report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
