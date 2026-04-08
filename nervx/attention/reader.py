"""`nervx read` — print source of a symbol (and optionally its callees).

Returns the actual source code of a specific symbol, plus the source of
symbols it calls up to depth N. Gives Claude exactly the code it needs
without loading entire files.
"""

from __future__ import annotations

from pathlib import Path

from nervx.attention.fuzzy import resolve_symbol
from nervx.memory.store import GraphStore


def read_symbol(
    store: GraphStore,
    symbol_id: str,
    context_depth: int = 0,
    repo_root: str = ".",
) -> str:
    """Read the source code of a symbol and optionally its callees."""
    node, error = resolve_symbol(store, symbol_id)
    if node is None:
        return error

    collected: list[tuple[int, dict]] = []
    _collect_symbols(
        store, node["id"], context_depth, collected, visited=set(), current_depth=0
    )

    output_parts: list[str] = []
    last_depth = 0
    for depth, sym_node in collected:
        source = _read_source_lines(sym_node, repo_root)
        if source is None:
            continue

        header = (
            f"## {sym_node['id']}  "
            f"(lines {sym_node['line_start']}-{sym_node['line_end']})"
        )
        if depth > last_depth:
            header = f"\n--- called at depth {depth} ---\n\n{header}"
        last_depth = max(last_depth, depth)

        output_parts.append(f"{header}\n\n{source}")

    if not output_parts:
        return f"Could not read source for: {node['id']}"

    return "\n\n".join(output_parts)


def _collect_symbols(
    store: GraphStore,
    node_id: str,
    max_depth: int,
    result: list[tuple[int, dict]],
    visited: set[str],
    current_depth: int = 0,
) -> None:
    """BFS-ish walk: emit the symbol, then recurse into `calls` edges."""
    if node_id in visited:
        return
    visited.add(node_id)

    node = store.get_node(node_id)
    if node is None or node["kind"] == "file":
        return

    result.append((current_depth, node))

    if current_depth >= max_depth:
        return

    edges = store.get_edges_from(node_id)
    for edge in edges:
        if edge["edge_type"] != "calls":
            continue
        _collect_symbols(
            store,
            edge["target_id"],
            max_depth,
            result,
            visited,
            current_depth + 1,
        )


def _read_source_lines(node: dict, repo_root: str) -> str | None:
    """Read the actual source lines for a node from disk."""
    fp = node.get("file_path")
    if not fp:
        return None

    file_path = Path(repo_root) / fp
    if not file_path.exists():
        return None

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    lines = text.splitlines()
    start = node.get("line_start")
    end = node.get("line_end")
    if not start or not end:
        return None

    start_idx = max(0, start - 1)  # DB is 1-indexed; slice is 0-indexed
    end_idx = end  # end is inclusive in the DB
    if start_idx >= len(lines):
        return None

    return "\n".join(lines[start_idx:end_idx])
