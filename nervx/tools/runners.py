"""Test-runner adapters.

Wraps pytest (and, in the future, other runners) so Claude receives a
compact summary instead of thousands of tokens of raw traceback output.
Raw output is cached under ``.nervx/run_cache`` and retrievable via a
short run ID.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def run_pytest(args: list[str], nervx_dir: str) -> str:
    """Run pytest and return a condensed summary.

    The raw combined stdout+stderr is always cached in
    ``<nervx_dir>/run_cache/raw_<run_id>.txt``; the returned string ends
    with a ``(raw output: nervx run pytest --raw <run_id>)`` hint.
    """
    cache_dir = Path(nervx_dir) / "run_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    json_report = cache_dir / "pytest_report.json"
    if json_report.exists():
        try:
            json_report.unlink()
        except OSError:
            pass

    base_cmd = [sys.executable, "-m", "pytest", "--tb=short", "-q"] + list(args)

    # First try with json-report; fall back to text parsing if the plugin
    # isn't installed (pytest will then complain about unknown arguments,
    # and we rerun without them).
    json_cmd = base_cmd + [
        "--json-report",
        f"--json-report-file={json_report}",
    ]
    # Force utf-8 decoding so non-ASCII bytes in pytest output (unicode
    # assertion messages, non-latin file paths) don't crash the reader
    # thread under Windows cp1252.
    try:
        result = subprocess.run(
            json_cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return f"pytest failed to start: {e}"

    combined_json_attempt = (result.stdout or "") + (result.stderr or "")
    if (
        "unrecognized arguments" in combined_json_attempt
        and "--json-report" in combined_json_attempt
    ):
        # Plugin missing — retry in plain text mode.
        try:
            result = subprocess.run(
                base_cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=600,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return f"pytest failed to start: {e}"

    combined = (result.stdout or "") + (result.stderr or "")
    run_id = hashlib.md5(combined.encode("utf-8", errors="replace")).hexdigest()[:8]
    (cache_dir / f"raw_{run_id}.txt").write_text(combined, encoding="utf-8", errors="replace")

    if json_report.exists():
        try:
            report = json.loads(json_report.read_text(encoding="utf-8"))
            return _format_pytest_json(report, run_id)
        except (OSError, json.JSONDecodeError):
            pass

    return _parse_pytest_text_output(combined, run_id)


def read_raw(run_id: str, nervx_dir: str) -> str:
    """Return the cached raw output for a previous run."""
    path = Path(nervx_dir) / "run_cache" / f"raw_{run_id}.txt"
    if not path.exists():
        return f"No cached output for run {run_id}"
    return path.read_text(encoding="utf-8", errors="replace")


# ── formatters ──────────────────────────────────────────────────────


def _format_pytest_json(report: dict, run_id: str) -> str:
    summary = report.get("summary") or {}
    passed = int(summary.get("passed", 0) or 0)
    failed = int(summary.get("failed", 0) or 0)
    errors = int(summary.get("error", 0) or 0)
    skipped = int(summary.get("skipped", 0) or 0)
    duration = float(report.get("duration", 0.0) or 0.0)

    lines: list[str] = [
        f"passed={passed} failed={failed} errors={errors} "
        f"skipped={skipped} duration={duration:.1f}s"
    ]

    for test in report.get("tests") or []:
        outcome = test.get("outcome")
        if outcome not in ("failed", "error"):
            continue
        nodeid = test.get("nodeid", "?")
        longrepr = (
            (test.get("call") or {}).get("longrepr")
            or (test.get("setup") or {}).get("longrepr")
            or ""
        )
        lines.append(f"  {outcome.upper()}: {nodeid}")
        short = _extract_failure_line(longrepr)
        if short:
            lines.append(f"    {short}")

    lines.append(f"  (raw output: nervx run pytest --raw {run_id})")
    return "\n".join(lines)


def _parse_pytest_text_output(output: str, run_id: str) -> str:
    """Fallback parser for when --json-report isn't available."""
    text_lines = output.strip().splitlines()

    # Find the summary line — pytest emits things like
    # "=== 3 passed, 1 failed in 0.42s ===" or "no tests ran in 0.01s".
    summary = ""
    summary_tokens = (
        "passed", "failed", "error", "skipped", "no tests", "deselected",
    )
    for line in reversed(text_lines):
        stripped = line.strip()
        if not stripped:
            continue
        if any(tok in stripped for tok in summary_tokens):
            summary = stripped.strip("= ")
            break

    result: list[str] = [summary or "Could not parse pytest output"]
    for line in text_lines:
        if line.startswith("FAILED") or line.startswith("ERROR"):
            result.append(f"  {line.strip()}")
    result.append(f"  (raw output: nervx run pytest --raw {run_id})")
    return "\n".join(result)


def _extract_failure_line(longrepr) -> str:
    if not longrepr:
        return ""
    if isinstance(longrepr, dict):
        text = longrepr.get("reprcrash", {}).get("message", "") or str(longrepr)
    else:
        text = str(longrepr)

    # Prefer an assertion line if one exists.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("E   ", "AssertionError", "assert ")):
            return stripped[:160]
    # Fallback: first non-empty, non-traceback line.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("File ", ">", "self = ")):
            return stripped[:160]
    return text.splitlines()[0][:160] if text else ""
