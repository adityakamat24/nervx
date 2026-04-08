"""`nervx callers` — show what calls a symbol.

A focused version of blast-radius for the single most common question:
"what calls this function?"
"""

from __future__ import annotations

from nervx.attention.fuzzy import resolve_symbol
from nervx.memory.store import GraphStore


def find_callers(
    store: GraphStore, symbol_id: str, max_depth: int = 1
) -> str:
    """Find all callers of a symbol up to `max_depth` (BFS layers)."""
    node, error = resolve_symbol(store, symbol_id)
    if node is None:
        return error
    symbol_id = node["id"]

    lines: list[str] = []
    header = (
        f"## Callers of: {node['name']} "
        f"({node['file_path']}:{node['line_start']})"
    )
    lines.append(header)
    lines.append("")

    visited: set[str] = {symbol_id}
    targets: set[str] = {symbol_id}
    any_caller_found = False

    for depth in range(1, max(1, max_depth) + 1):
        label = "Direct callers" if depth == 1 else f"Indirect callers (depth {depth})"
        layer_callers: list[str] = []
        next_targets: set[str] = set()

        for target in targets:
            # `calls` edges are caller -> callee, so callers of X are the
            # source_ids of rows where target_id = X.
            rows = store.conn.execute(
                """
                SELECT DISTINCT e.source_id, n.name, n.file_path,
                       n.line_start, n.signature, n.kind
                FROM edges e
                JOIN nodes n ON e.source_id = n.id
                WHERE e.target_id = ? AND e.edge_type = 'calls'
                """,
                (target,),
            ).fetchall()

            for caller_id, name, fp, line, sig, kind in rows:
                if caller_id in visited:
                    continue
                visited.add(caller_id)
                next_targets.add(caller_id)

                via = ""
                if depth > 1:
                    target_node = store.get_node(target)
                    if target_node:
                        via = f"  [calls {target_node['name']}]"

                label_text = sig or name or caller_id
                loc = f"{fp}:{line}"
                layer_callers.append(f"  {loc:<40} {label_text}{via}")

        if layer_callers:
            any_caller_found = True
            lines.append(f"{label} ({len(layer_callers)}):")
            lines.extend(layer_callers)
            lines.append("")

        targets = next_targets
        if not targets:
            break

    if not any_caller_found:
        lines.append(
            "  No callers found. This symbol may be a framework entry point "
            "or dead code."
        )

    return "\n".join(lines)
