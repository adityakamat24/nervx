"""`nervx verify "A calls B"` — graph path existence check.

Parses a natural-language statement like "foo calls bar" and answers
yes/no with the path. ~50 tokens instead of 1500 tokens of source reads.
"""

from __future__ import annotations

from nervx.attention.fuzzy import resolve_symbol
from nervx.attention.graph_paths import bfs_path
from nervx.memory.store import GraphStore


_EDGE_KEYWORDS: list[tuple[str, str]] = [
    # Phrase → edge type. Order matters: multi-word phrases first.
    ("is called by", "called_by"),
    ("called by", "called_by"),
    ("inherits from", "inherits"),
    ("inherits", "inherits"),
    ("extends", "inherits"),
    ("imports", "imports"),
    ("imported by", "imported_by"),
    ("calls", "calls"),
]


def verify_statement(store: GraphStore, statement: str) -> dict | str:
    """Return a dict with the verification result, or an error string."""
    text = statement.strip().strip('"').strip("'")
    lowered = text.lower()

    keyword = ""
    edge_type = ""
    for phrase, etype in _EDGE_KEYWORDS:
        if phrase in lowered:
            keyword = phrase
            edge_type = etype
            break
    if not keyword:
        return (
            "Could not parse statement. Use: 'A calls B', 'A imports B', "
            "'A extends B', or 'A is called by B'."
        )

    idx = lowered.index(keyword)
    left = text[:idx].strip().strip('"').strip("'")
    right = text[idx + len(keyword):].strip().strip('"').strip("'")

    if not left or not right:
        return f"Could not parse statement: missing symbol around '{keyword}'."

    source_node, err_s = resolve_symbol(store, left)
    if source_node is None:
        return f"verify: could not resolve left-hand symbol '{left}'.\n{err_s}"
    target_node, err_t = resolve_symbol(store, right)
    if target_node is None:
        return f"verify: could not resolve right-hand symbol '{right}'.\n{err_t}"

    path = bfs_path(
        store,
        source_node["id"],
        target_node["id"],
        edge_type=edge_type,
        max_depth=6,
    )

    if path:
        names: list[str] = []
        for nid in path:
            n = store.get_node(nid)
            names.append(n["name"] if n else nid.split("::")[-1])
        return {
            "confirmed": True,
            "edge_type": edge_type,
            "source": source_node["id"],
            "target": target_node["id"],
            "path": path,
            "names": names,
            "hops": len(path) - 1,
        }

    return {
        "confirmed": False,
        "edge_type": edge_type,
        "source": source_node["id"],
        "target": target_node["id"],
        "path": [],
        "names": [],
        "hops": 0,
    }


def format_verify(result: dict | str) -> str:
    if isinstance(result, str):
        return result
    if result["confirmed"]:
        arrow = " → ".join(result["names"])
        return (
            f"Confirmed: {arrow}  "
            f"({result['hops']} hop{'s' if result['hops'] != 1 else ''}, {result['edge_type']})"
        )
    return (
        f"No {result['edge_type']} path from "
        f"{result['source']} to {result['target']} (searched 6 hops)."
    )
