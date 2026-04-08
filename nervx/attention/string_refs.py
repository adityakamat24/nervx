"""`nervx string-refs <identifier>` — cross-language string-literal lookup.

Indexed at build time by ``compute_symbol_hashes_and_strings`` in
``nervx/build.py``. Returns every file:line where an identifier-shaped
**quoted string literal** occurs (e.g. ``"user_id"`` in a JSON file, a
Python dict key, or a JS property access via bracket notation). Catches
cross-language field name drift: Python attribute ↔ JSON key ↔ JS property.

This does NOT index bare code tokens — a class name like ``RadixCache``
referenced as ``RadixCache(...)`` in Python code will not be found here.
For bare-token references use ``nervx callers`` (for call sites) or
fall back to grep.
"""

from __future__ import annotations

from nervx.memory.store import GraphStore


def find_string_refs(store: GraphStore, identifier: str) -> dict:
    rows = store.get_string_refs(identifier)
    return {
        "literal": identifier,
        "count": len(rows),
        "refs": [
            {
                "file_path": r["file_path"],
                "line_number": r["line_number"],
                "context": r.get("context", ""),
            }
            for r in rows
        ],
    }


def format_string_refs(result: dict) -> str:
    if result["count"] == 0:
        return (
            f'No quoted string literals matching "{result["literal"]}" found. '
            f"(string-refs only indexes quoted strings — for bare code-token "
            f"references such as class/function names, use `nervx callers` "
            f"or grep.)"
        )
    lines = [f'## string-refs: "{result["literal"]}"  ({result["count"]} refs)']
    # Group by file for compactness.
    by_file: dict[str, list[int]] = {}
    for ref in result["refs"]:
        by_file.setdefault(ref["file_path"], []).append(ref["line_number"])
    for fp in sorted(by_file):
        line_nums = ", ".join(str(n) for n in sorted(by_file[fp]))
        lines.append(f"  {fp}: {line_nums}")
    return "\n".join(lines)
