"""`nervx peek` — tiny preview of a symbol.

Returns ~50-token metadata: signature, first doc line, callees, caller
count, test-coverage hint. No source code. Used as a pre-read probe.
"""

from __future__ import annotations

import json

from nervx.attention.fuzzy import resolve_symbol
from nervx.memory.store import GraphStore


def peek_symbol(
    store: GraphStore,
    symbol_id: str,
    repo_root: str = ".",
) -> dict | str:
    """Return a structured preview of a symbol.

    On resolution failure returns an error string (the caller can detect
    with ``isinstance(result, str)``).
    """
    node, error = resolve_symbol(store, symbol_id)
    if node is None:
        return error

    nid = node["id"]
    line_start = node.get("line_start") or 0
    line_end = node.get("line_end") or 0
    line_count = max(0, line_end - line_start + 1) if line_start and line_end else 0

    # Callees (via `calls` edges).
    callees: list[str] = []
    for edge in store.get_edges_from(nid):
        if edge["edge_type"] != "calls":
            continue
        target = edge["target_id"]
        callees.append(target.split("::")[-1] if "::" in target else target)

    # Callers (via `called_by` edges; schema keeps them in sync via REVERSE_EDGE_MAP).
    caller_count = sum(
        1 for e in store.get_edges_to(nid) if e["edge_type"] == "calls"
    )

    # Naive test-coverage probe: any node tagged "test" that calls this symbol.
    test_count = 0
    try:
        row = store.conn.execute(
            """
            SELECT COUNT(*) AS cnt FROM edges e
            JOIN nodes n ON e.source_id = n.id
            WHERE e.target_id = ?
              AND e.edge_type = 'calls'
              AND n.tags LIKE '%"test"%'
            """,
            (nid,),
        ).fetchone()
        test_count = row["cnt"] if row else 0
    except Exception:
        test_count = 0

    # Tags (stored as JSON string in the DB row).
    raw_tags = node.get("tags") or "[]"
    try:
        tags = json.loads(raw_tags) if isinstance(raw_tags, str) else list(raw_tags)
    except (TypeError, ValueError):
        tags = []
    notable = [
        t for t in tags
        if not t.startswith("extends:") and not t.startswith("decorator:")
    ]

    doc = (node.get("docstring") or "").strip()
    if doc:
        doc = doc.splitlines()[0][:120]

    return {
        "id": nid,
        "kind": node.get("kind", ""),
        "file_path": node.get("file_path", ""),
        "line_start": line_start,
        "line_end": line_end,
        "line_count": line_count,
        "signature": (node.get("signature") or "").strip(),
        "docstring_first_line": doc,
        "importance": float(node.get("importance") or 0.0),
        "callees": callees[:8],
        "callees_total": len(callees),
        "caller_count": caller_count,
        "test_count": test_count,
        "tags": notable,
    }


def format_peek(peek: dict | str) -> str:
    """Render a peek dict as the ~50-token preview text."""
    if isinstance(peek, str):
        return peek

    loc = f"{peek['file_path']}:{peek['line_start']}-{peek['line_end']}"
    lines: list[str] = [f"## {peek['id']}  ({loc})"]

    if peek["signature"]:
        lines.append(f"  {peek['signature']}")
    if peek["docstring_first_line"]:
        lines.append(f'  "{peek["docstring_first_line"]}"')

    meta_parts = [f"{peek['line_count']} lines"]
    if peek["callees"]:
        more = "" if peek["callees_total"] <= len(peek["callees"]) else "…"
        meta_parts.append(f"calls: {', '.join(peek['callees'])}{more}")
    lines.append(f"  [{', '.join(meta_parts)}]")

    info = [f"importance={peek['importance']:.0f}", f"{peek['caller_count']} callers"]
    if peek["test_count"] == 0:
        info.append("no test coverage")
    else:
        info.append(f"{peek['test_count']} test refs")
    lines.append(f"  [{', '.join(info)}]")

    if peek["tags"]:
        lines.append(f"  tags: {', '.join(peek['tags'])}")

    return "\n".join(lines)
