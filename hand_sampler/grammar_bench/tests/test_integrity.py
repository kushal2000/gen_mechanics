"""Benchmark integrity checks: fixture provenance, independence of the reference
implementations, and honest skip reasons."""

import hashlib
import json
import re
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BENCH_DIR / "manifest.json"


def _manifest():
    return json.loads(MANIFEST_PATH.read_text())


def test_committed_fixture_sha256_matches_manifest():
    manifest = _manifest()
    checked = 0
    for hand in manifest["hands"]:
        if not hand.get("commit_allowed") or not hand.get("fixture_path"):
            continue
        fixture_path = BENCH_DIR / hand["fixture_path"]
        assert fixture_path.is_file(), f"{hand['id']}: missing committed fixture {fixture_path}"
        digest = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        assert digest == hand["sha256"], f"{hand['id']}: sha256 mismatch ({digest} != {hand['sha256']})"
        checked += 1
    assert checked >= 3, "expected at least the three dev-set committed hands to be checked"


def _strip_comments_and_strings(source: str) -> str:
    """Best-effort removal of Python comments and string literals so the
    independence lint doesn't get confused by mentions inside docstrings."""
    # Remove triple-quoted strings first (docstrings), then line comments.
    no_triple = re.sub(r'"""(?:.|\n)*?"""|\'\'\'(?:.|\n)*?\'\'\'', "", source)
    no_strings = re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', "", no_triple)
    lines = []
    for line in no_strings.splitlines():
        idx = line.find("#")
        lines.append(line if idx == -1 else line[:idx])
    return "\n".join(lines)


def test_analytic_and_refgen_do_not_import_the_implementation():
    forbidden = "hand_sampler.grammar"
    candidates = [BENCH_DIR / "analytic.py"]
    refgen_dir = BENCH_DIR / "refgen"
    if refgen_dir.is_dir():
        candidates.extend(sorted(refgen_dir.glob("*.py")))
    checked = 0
    for path in candidates:
        if not path.is_file():
            continue
        code_only = _strip_comments_and_strings(path.read_text())
        assert forbidden not in code_only, f"{path} references {forbidden!r} outside comments/strings"
        checked += 1
    assert checked >= 1, "expected at least analytic.py to be checked"


def test_no_unexplained_skips_in_grammar_bench_tests():
    tests_dir = Path(__file__).resolve().parent
    skip_call = re.compile(r"pytest\.skip\(\s*(?:reason\s*=\s*)?([\"'])(.*?)\1", re.DOTALL)
    skipif_reason = re.compile(r"pytest\.mark\.skipif\([^)]*reason\s*=\s*([\"'])(.*?)\1", re.DOTALL)
    allowed_prefixes = ("local-only:", "oracle-unavailable")
    checked = 0
    for path in sorted(tests_dir.glob("test_*.py")):
        source = path.read_text()
        for pattern in (skip_call, skipif_reason):
            for match in pattern.finditer(source):
                reason = match.group(2)
                assert reason.startswith(allowed_prefixes), (
                    f"{path}: skip reason {reason!r} must start with one of {allowed_prefixes}"
                )
        checked += 1
    assert checked >= 1
