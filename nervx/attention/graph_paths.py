"""Shared graph-path primitives (BFS) used by verify and trace."""

from __future__ import annotations

from collections import deque

from nervx.memory.store import GraphStore


def bfs_path(
    store: GraphStore,
    source_id: str,
    target_id: str,
    edge_type: str = "calls",
    max_depth: int = 6,
) -> list[str]:
    """Return the shortest path of node IDs from source to target, or [] if none.

    ``edge_type`` controls which edge to traverse (``calls``, ``imports``,
    ``inherits``, ...). BFS stops at ``max_depth`` hops.
    """
    if source_id == target_id:
        return [source_id]

    visited: set[str] = {source_id}
    parents: dict[str, str] = {}
    queue: deque[tuple[str, int]] = deque([(source_id, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for edge in store.get_edges_from(current):
            if edge["edge_type"] != edge_type:
                continue
            nxt = edge["target_id"]
            if nxt in visited:
                continue
            visited.add(nxt)
            parents[nxt] = current
            if nxt == target_id:
                return _reconstruct(parents, source_id, target_id)
            queue.append((nxt, depth + 1))

    return []


def _reconstruct(parents: dict[str, str], source_id: str, target_id: str) -> list[str]:
    path = [target_id]
    while path[-1] != source_id:
        path.append(parents[path[-1]])
    return list(reversed(path))
