"""``nervx uses <identifier>`` — runtime scan for bare-token usages.

Why this exists
----------------
``nervx string-refs`` deliberately only matches quoted string literals
(``"voted_for"`` inside JSON keys / dict subscripts), so cross-language
renames that include attribute access, assignment targets, destructuring,
type hints, or match-arm bindings need a second command. ``uses`` is that
command: a regex-based scanner that walks the repo the same way ``nervx
build`` walks it (honoring ``.nervxignore`` and the default excludes) and
reports every non-comment line where the identifier appears as a
whole-word token.

Design notes
-------------
- Purely a query-time operation. No new index — results change if the
  code changes, but that's fine for a rename workflow that runs ``uses``
  then immediately edits.
- Language-agnostic: uses ``\\b<ident>\\b`` on raw lines and skips lines
  whose leading non-whitespace is a comment marker (``#``, ``//``,
  ``/*``, ``*``, ``--``). This handles Python, JS/TS, Java, Go, Rust,
  C/C++, C#, Ruby, SQL, and Lua-style comments.
- File discovery reuses ``nervx.build.walk_files`` so the same
  ``.nervxignore`` rules keep ``uses`` from crawling ``node_modules/``,
  ``.venv/``, vendor trees, etc.
"""

from __future__ import annotations

import fnmatch
import os
import re

from nervx.build import walk_files

# Match 3-64 char identifier; rejects whitespace and punctuation.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

# Leading tokens that indicate a comment line. Cheap prefix check avoids
# trying to build a language-specific comment parser.
_COMMENT_PREFIXES: tuple[str, ...] = ("#", "//", "/*", "*", "--", ";")

# Per-file cap on matches so a single logging file with thousands of
# occurrences doesn't drown out the rest of the repo.
_PER_FILE_CAP = 200


def find_identifier_uses(
    repo_root: str,
    identifier: str,
    path_filter: str | None = None,
    max_results: int = 500,
) -> dict:
    """Scan the repo for lines that contain ``identifier`` as a whole word.

    ``path_filter`` is an optional glob (e.g. ``src/**/*.py``) applied
    against each file's relative path. ``max_results`` caps the total
    number of hits returned across all files.
    """
    if not _IDENT_RE.match(identifier or ""):
        return {
            "op": "uses",
            "identifier": identifier,
            "error": (
                "Identifier must start with a letter or underscore and only "
                "contain letters/digits/underscore (1-64 chars)."
            ),
        }

    # Compile once with \b word boundaries so ``voted`` does not match
    # ``voted_for`` or vice versa.
    pattern = re.compile(r"\b" + re.escape(identifier) + r"\b")

    try:
        files = walk_files(repo_root)
    except Exception as e:
        return {
            "op": "uses",
            "identifier": identifier,
            "error": f"walk failed: {e}",
        }

    hits: list[dict] = []
    files_scanned = 0
    truncated = False

    for full_path in files:
        rel = os.path.relpath(full_path, repo_root).replace("\\", "/")
        if path_filter and not fnmatch.fnmatch(rel, path_filter):
            continue

        files_scanned += 1
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            continue

        file_hits = 0
        for i, line in enumerate(lines, start=1):
            if file_hits >= _PER_FILE_CAP:
                break
            # Cheap comment skip: strip leading whitespace and check a few
            # known comment prefixes.
            stripped = line.lstrip()
            if stripped.startswith(_COMMENT_PREFIXES):
                continue
            if not pattern.search(line):
                continue

            hits.append({
                "file": rel,
                "line": i,
                "text": line.rstrip("\n"),
            })
            file_hits += 1

            if len(hits) >= max_results:
                truncated = True
                break

        if truncated:
            break

    # Group hits by file for the consumer — easier to render and mirrors
    # what ``string-refs`` already returns.
    by_file: dict[str, list[dict]] = {}
    for h in hits:
        by_file.setdefault(h["file"], []).append({
            "line": h["line"],
            "text": h["text"],
        })

    return {
        "op": "uses",
        "identifier": identifier,
        "count": len(hits),
        "files_scanned": files_scanned,
        "truncated": truncated,
        "by_file": by_file,
    }


def format_uses(result: dict) -> str:
    """Render ``find_identifier_uses`` as a compact text report."""
    if "error" in result:
        return result["error"]

    ident = result.get("identifier", "")
    count = result.get("count", 0)
    files_scanned = result.get("files_scanned", 0)
    by_file: dict[str, list[dict]] = result.get("by_file") or {}

    if not by_file:
        return (
            f"No uses of '{ident}' found across {files_scanned} scanned "
            f"files. Check spelling, or try `nervx string-refs {ident}` "
            f"for quoted-string matches only."
        )

    lines: list[str] = []
    header = (
        f"## uses: {ident}  ({count} hit"
        f"{'s' if count != 1 else ''} across {len(by_file)} file"
        f"{'s' if len(by_file) != 1 else ''})"
    )
    lines.append(header)
    if result.get("truncated"):
        lines.append("  (truncated — rerun with --limit <N> to widen)")
    lines.append("")

    for file_path in sorted(by_file):
        hits = by_file[file_path]
        lines.append(f"{file_path} ({len(hits)}):")
        for hit in hits:
            text = hit["text"].strip()
            if len(text) > 120:
                text = text[:117] + "..."
            lines.append(f"  {hit['line']:>5}  {text}")
        lines.append("")

    return "\n".join(lines).rstrip()
