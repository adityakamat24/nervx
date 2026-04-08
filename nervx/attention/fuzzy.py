"""Fuzzy symbol lookup with 'did you mean?' suggestions.

Used by commands that take a symbol_id argument (read, callers, blast-radius).
When an exact match fails, suggest the closest symbols so Claude doesn't waste
a round-trip guessing the fully-qualified name.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from nervx.memory.store import GraphStore


def fuzzy_find_symbol(store: GraphStore, query: str, limit: int = 5) -> list[str]:
    """Find symbols that closely match a query string.

    Tries, in order:
    1. Suffix match — query matches the end of the full node id.
    2. Name match — query short-name equals the node name.
    3. Qualified-name match — query matches ClassName.method portion.
    4. Name contains query.
    5. Substring of full id.
    6. Fuzzy similarity on the short name.
    """
    query_lower = query.lower()

    # Extract the "name" part: "file.py::ClassName.method" -> "ClassName.method"
    query_name = query.split("::")[-1] if "::" in query else query
    # Just the final segment: "ClassName.method" -> "method"
    query_short = query_name.split(".")[-1] if "." in query_name else query_name

    rows = store.conn.execute(
        "SELECT id, name FROM nodes WHERE kind != 'file'"
    ).fetchall()

    candidates: list[tuple[float, str]] = []

    for node_id, node_name in rows:
        node_id_lower = node_id.lower()
        node_name_lower = node_name.lower()
        score = 0.0

        if node_id_lower.endswith(query_lower):
            score = 0.95
        elif node_name_lower == query_short.lower():
            score = 0.90
        elif node_id_lower.endswith(query_name.lower()):
            score = 0.88
        elif query_short.lower() in node_name_lower:
            score = 0.70
        elif query_lower in node_id_lower:
            score = 0.60
        else:
            ratio = SequenceMatcher(
                None, query_short.lower(), node_name_lower
            ).ratio()
            if ratio > 0.5:
                score = ratio * 0.5  # scale fuzzy matches down

        if score > 0.3:
            candidates.append((score, node_id))

    candidates.sort(key=lambda x: -x[0])
    return [node_id for _, node_id in candidates[:limit]]


def resolve_symbol(
    store: GraphStore, query: str
) -> tuple[dict | None, str]:
    """Resolve a symbol query to a node.

    Returns (node_dict, error_message).
    - Found:                 (node, "")
    - One strong suggestion: (node, "")  — auto-resolved
    - Ambiguous or missing:  (None, "Symbol not found: ... Did you mean: ...")
    """
    # Exact match first
    node = store.get_node(query)
    if node:
        return node, ""

    suggestions = fuzzy_find_symbol(store, query)

    if not suggestions:
        return None, f"Symbol not found: {query}"

    if len(suggestions) == 1:
        # Single strong match — auto-resolve
        node = store.get_node(suggestions[0])
        if node:
            return node, ""

    lines = [f"Symbol not found: {query}", "", "Did you mean:"]
    for s in suggestions:
        n = store.get_node(s)
        if n:
            lines.append(
                f"  {s}  [{n['kind']}] {n['file_path']}:{n['line_start']}"
            )
        else:
            lines.append(f"  {s}")

    return None, "\n".join(lines)
