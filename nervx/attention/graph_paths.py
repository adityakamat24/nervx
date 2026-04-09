"""Shared graph-path primitives (BFS) used by verify and trace."""

from __future__ import annotations

import json
from collections import deque

from nervx.memory.store import GraphStore


def _is_low_confidence(edge) -> bool:
    """Return True if an edge row was emitted by fan-out disambiguation.

    0.2.6: the linker tags fan-out ``calls`` edges with
    ``metadata.confidence = "low"`` so strict consumers (verify, trace,
    ``ask calls``) can ignore them. Any other edge is treated as
    high-confidence.
    """
    meta_raw = edge.get("metadata") if hasattr(edge, "get") else edge["metadata"]
    if not meta_raw:
        return False
    try:
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    except (TypeError, ValueError):
        return False
    return meta.get("confidence") == "low"


def bfs_path(
    store: GraphStore,
    source_id: str,
    target_id: str,
    edge_type: str | tuple[str, ...] = "calls",
    max_depth: int = 6,
    strict: bool = False,
) -> list[str]:
    """Return the shortest path of node IDs from source to target, or [] if none.

    ``edge_type`` controls which edge(s) to traverse. Accepts a single edge
    type string (``"calls"``, ``"imports"``, ``"inherits"``, ...) or a tuple
    of types for multi-edge traversal (used by trace's inheritance fallback).
    BFS stops at ``max_depth`` hops.

    ``strict`` (0.2.6): when True, skip edges whose metadata carries
    ``confidence="low"`` (fan-out candidates the linker could not
    disambiguate). Callers making a truth claim — "does A definitely
    reach B?" — should set this so we never confirm a path that only
    exists because of speculative method-name matching.
    """
    if source_id == target_id:
        return [source_id]

    allowed: tuple[str, ...] = (edge_type,) if isinstance(edge_type, str) else tuple(edge_type)

    visited: set[str] = {source_id}
    parents: dict[str, str] = {}
    queue: deque[tuple[str, int]] = deque([(source_id, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for edge in store.get_edges_from(current):
            if edge["edge_type"] not in allowed:
                continue
            if strict and _is_low_confidence(edge):
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
