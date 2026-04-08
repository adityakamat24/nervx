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
    return [nid for nid, _ in fuzzy_find_scored(store, query, limit)]


def fuzzy_find_scored(
    store: GraphStore, query: str, limit: int = 5
) -> list[tuple[str, float]]:
    """Like ``fuzzy_find_symbol`` but returns ``(node_id, score)`` pairs."""
    # B11: normalize Windows-style backslashes so pasted paths resolve.
    query = query.replace("\\", "/")
    query_lower = query.lower()

    # Extract the "name" part: "file.py::ClassName.method" -> "ClassName.method"
    query_name = query.split("::")[-1] if "::" in query else query
    # Just the final segment: "ClassName.method" -> "method"
    query_short = query_name.split(".")[-1] if "." in query_name else query_name

    rows = store.conn.execute(
        "SELECT id, name FROM nodes WHERE kind != 'file'"
    ).fetchall()

    candidates: list[tuple[float, str]] = []

    query_short_l = query_short.lower()
    query_name_l = query_name.lower()

    # If the query carries extra context (file path or qualifier before ::),
    # track it so we can break ties between identically-named symbols.
    query_context = query_lower
    if "::" in query_context:
        query_context = query_context.split("::", 1)[0]

    # "ClassName.method" queries carry a qualifier we can match explicitly;
    # plain "method" queries have no qualifier and should fall back to the
    # short-name branch.
    query_has_qualifier = "." in query_name

    for node_id, node_name in rows:
        node_id_lower = node_id.lower()
        node_name_lower = node_name.lower()
        score = 0.0

        # Qualified-name exact match — "ClassName.method" → "...::ClassName.method".
        # Beats plain short-name matches so users can use Class.method shorthand
        # without the file path, even when the method name is common across
        # classes.
        if query_has_qualifier and (
            node_id_lower.endswith("::" + query_name_l)
            or node_id_lower == query_name_l
        ):
            score = 0.99
        # Exact name match — "I know the name, just not the file path" case.
        elif node_name_lower == query_short_l:
            score = 0.98
        # Full-id suffix match (e.g. query "ClassName.method")
        elif node_id_lower.endswith("::" + query_name_l) or node_id_lower == query_name_l:
            score = 0.95
        # Query is the full id suffix (after the ::)
        elif "::" in node_id_lower and node_id_lower.split("::", 1)[1] == query_lower:
            score = 0.94
        elif node_id_lower.endswith(query_lower):
            score = 0.80
        elif query_short_l in node_name_lower:
            score = 0.70
        elif query_lower in node_id_lower:
            score = 0.60
        else:
            ratio = SequenceMatcher(None, query_short_l, node_name_lower).ratio()
            if ratio > 0.5:
                score = ratio * 0.5  # scale fuzzy matches down

        # Tiny context tie-breaker: if the query mentions part of the
        # file path or qualifier, prefer nodes that include it. Keeps the
        # total score under 1.0 so exact matches still dominate.
        if score >= 0.5 and query_context and query_context != query_short_l:
            if query_context in node_id_lower:
                score += 0.01

        if score > 0.3:
            candidates.append((score, node_id))

    candidates.sort(key=lambda x: -x[0])
    return [(nid, sc) for sc, nid in candidates[:limit]]


def resolve_symbol(
    store: GraphStore, query: str, pick: int | None = None,
) -> tuple[dict | None, str]:
    """Resolve a symbol query to a node.

    Returns (node_dict, error_message).
    - Found:                 (node, "")
    - One strong suggestion: (node, "")  — auto-resolved
    - Clear winner:          (node, "")  — top score strictly beats the rest
    - Ambiguous or missing:  (None, "Symbol not found: ... Did you mean: ...")

    ``pick`` (1-indexed, B10) — after fuzzy scoring, bypass the ambiguity
    gate and return the Nth candidate. Stateless shortcut for rerunning a
    failed query with a specific choice from the did-you-mean list.
    """
    # B11: normalize pasted Windows paths up front. Applies to exact lookup
    # too so `store.get_node("nervx\\cli\\main.py")` hits the stored id.
    query = query.replace("\\", "/")

    # If the user explicitly asked for the Nth pick, skip the exact-match
    # shortcut and go straight to scoring so the order is deterministic.
    if pick is None:
        node = store.get_node(query)
        if node:
            return node, ""

    scored = fuzzy_find_scored(store, query)

    if not scored:
        return None, f"Symbol not found: {query}"

    if pick is not None:
        idx = pick - 1
        if idx < 0 or idx >= len(scored):
            return None, (
                f"--pick {pick} out of range (have {len(scored)} candidates for '{query}')"
            )
        node = store.get_node(scored[idx][0])
        if node:
            return node, ""
        return None, f"Symbol not found: {scored[idx][0]}"

    # One match, or a clear winner (top score strictly greater than the next).
    top_id, top_score = scored[0]
    if len(scored) == 1 or top_score > scored[1][1]:
        node = store.get_node(top_id)
        if node:
            return node, ""

    lines = [f"Symbol not found: {query}", "", "Did you mean:"]
    for i, (s, _) in enumerate(scored, 1):
        n = store.get_node(s)
        if n:
            lines.append(
                f"  [{i}] {s}  [{n['kind']}] {n['file_path']}:{n['line_start']}"
            )
        else:
            lines.append(f"  [{i}] {s}")
    lines.append("")
    lines.append("Rerun with --pick N to select one of the candidates above.")

    return None, "\n".join(lines)
