"""Gitignore-style pattern matching for `.nervxignore`.

No external dependency — `fnmatch` + `pathlib` only.

Semantics (matching .gitignore where practical):
    - One pattern per line; `#` comments; blank lines ignored.
    - Trailing `/` = directory-only pattern.
    - `*` matches anything except `/`.
    - `**` matches any number of directories.
    - A leading `!` negates a previous match (re-include).
    - Patterns without `/` match any path component.
    - Patterns with `/` are anchored to the repo root.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

DEFAULT_IGNORE: list[str] = [
    "__pycache__/", ".git/", ".venv/", "venv/", "env/",
    "node_modules/", ".tox/", ".mypy_cache/", ".pytest_cache/",
    "*.egg-info/", "dist/", "build/", ".nervx/",
    "*.min.js", "*.min.css", "*.bundle.js", "*.chunk.js",
]


def load_ignore_patterns(repo_root: str) -> list[str]:
    """Load patterns from `.nervxignore` (if present) on top of the defaults."""
    patterns = list(DEFAULT_IGNORE)
    ignore_file = Path(repo_root) / ".nervxignore"
    if ignore_file.exists():
        try:
            text = ignore_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return patterns
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def should_ignore(file_path: str, patterns: list[str]) -> bool:
    """Check if a repo-relative path matches any ignore pattern.

    `file_path` should use forward slashes (e.g. "frontend/assets/x.min.js").
    """
    # Normalize to forward slashes
    file_path = file_path.replace("\\", "/")
    # A trailing slash means "this is a directory" — callers pass `dir/` for
    # the walk prune path. In that case every component is a directory.
    path_is_dir = file_path.endswith("/")
    stripped = file_path.rstrip("/")
    path = Path(stripped)
    parts = path.parts
    # When the path itself is a directory, include the final part as a
    # directory component; otherwise it's a filename.
    dir_parts = parts if path_is_dir else parts[:-1]

    ignored = False
    for raw in patterns:
        pattern = raw
        negate = pattern.startswith("!")
        if negate:
            pattern = pattern[1:]

        is_dir_pattern = pattern.endswith("/")
        if is_dir_pattern:
            pattern = pattern.rstrip("/")

        # `**/foo` is equivalent to bare `foo` (match at any depth).
        while pattern.startswith("**/"):
            pattern = pattern[3:]

        if not pattern:
            continue

        matched = False

        if "**" in pattern:
            # In fnmatch, `*` already matches path separators, so `**` is redundant.
            # Normalize and match against the full path.
            flat = pattern.replace("**", "*")
            matched = fnmatch(stripped, flat)
        elif "/" in pattern:
            # Anchored path-style pattern (relative to repo root).
            if is_dir_pattern:
                # Match any file under the pattern directory.
                matched = (
                    fnmatch(stripped, pattern + "/*")
                    or fnmatch(stripped, pattern)
                )
            else:
                matched = fnmatch(stripped, pattern)
        else:
            # Basename-only pattern: match against any path component.
            if is_dir_pattern:
                matched = any(fnmatch(part, pattern) for part in dir_parts)
            else:
                matched = fnmatch(path.name, pattern) or any(
                    fnmatch(part, pattern) for part in parts
                )

        if matched:
            ignored = not negate

    return ignored
