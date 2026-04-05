"""Concept path detection: end-to-end data flows through the codebase."""

from __future__ import annotations

import json
import re
from collections import defaultdict

from nervx.memory.store import GraphStore


def detect_concept_paths(store: GraphStore):
    """Run all concept path detection strategies.

    Pre-loads all data into memory for fast BFS/traversal.
    """
    store.conn.execute("DELETE FROM concept_paths")
    store.conn.commit()

    # Pre-load into memory
    all_nodes = store.get_all_nodes()
    all_edges = store.get_all_edges()

    nodes_by_id = {n["id"]: n for n in all_nodes}

    # Build in-memory adjacency lists for "calls" edges only
    calls_adj: dict[str, list[str]] = defaultdict(list)
    for e in all_edges:
        if e["edge_type"] == "calls":
            calls_adj[e["source_id"]].append(e["target_id"])

    # Load keywords for domain clustering
    kw_rows = store.conn.execute(
        "SELECT keyword, node_id FROM keywords WHERE source = 'name'"
    ).fetchall()

    with store.batch():
        _entry_to_terminal_chains(store, all_nodes, calls_adj)
        _domain_clusters(store, kw_rows, calls_adj, nodes_by_id)
        _long_call_chains(store, all_nodes, calls_adj, nodes_by_id)


def _entry_to_terminal_chains(store, all_nodes, calls_adj):
    """BFS from entrypoint/route_handler nodes along calls edges."""
    entry_nodes = []
    for node in all_nodes:
        tags = json.loads(node["tags"]) if isinstance(node["tags"], str) else node["tags"]
        if "entrypoint" in tags or "route_handler" in tags:
            entry_nodes.append(node)

    for entry in entry_nodes:
        paths = _bfs_paths(calls_adj, entry["id"], max_depth=8, max_paths=2, max_branch=3)
        for path in paths:
            if len(path) < 2:
                continue
            start_name = _short_name(path[0])
            end_name = _short_name(path[-1])
            name = f"{start_name}_to_{end_name}"
            slug = f"flow_{name}".replace(".", "_").replace("::", "_")

            store.add_concept_path(
                id=slug, name=name,
                node_ids=path, path_type="call_chain",
            )


def _bfs_paths(calls_adj, start_id, max_depth=8, max_paths=2, max_branch=3):
    """In-memory BFS along calls edges, collecting complete paths."""
    paths: list[list[str]] = []
    stack = [(start_id, [start_id])]
    visited_global: set[str] = {start_id}

    while stack and len(paths) < max_paths:
        current_id, path = stack.pop(0)

        if len(path) >= max_depth:
            paths.append(path)
            continue

        targets = calls_adj.get(current_id, [])

        if not targets:
            if len(path) >= 2:
                paths.append(path)
            continue

        branches_added = 0
        for target in targets:
            if branches_added >= max_branch:
                break
            if target in visited_global:
                continue
            visited_global.add(target)
            stack.append((target, path + [target]))
            branches_added += 1

        if branches_added == 0 and len(path) >= 2:
            paths.append(path)

    return paths


def _domain_clusters(store, kw_rows, calls_adj, nodes_by_id):
    """Find keywords in 3-15 symbols, group and order by call relationships."""
    keyword_nodes: dict[str, list[str]] = defaultdict(list)
    for row in kw_rows:
        kw = row["keyword"]
        if len(kw) >= 4:
            keyword_nodes[kw].append(row["node_id"])

    for kw, node_ids in keyword_nodes.items():
        if len(node_ids) < 3 or len(node_ids) > 15:
            continue

        ordered = _topological_order(node_ids, calls_adj)
        if len(ordered) < 3:
            continue

        store.add_concept_path(
            id=f"domain_{kw}", name=f"domain_{kw}",
            node_ids=ordered, path_type="domain_cluster",
        )


def _topological_order(node_ids, calls_adj):
    """In-memory topological sort by call relationships."""
    id_set = set(node_ids)
    in_deg: dict[str, int] = {nid: 0 for nid in node_ids}
    adj: dict[str, list[str]] = defaultdict(list)

    for nid in node_ids:
        for target in calls_adj.get(nid, []):
            if target in id_set:
                adj[nid].append(target)
                in_deg[target] += 1

    queue = [nid for nid in node_ids if in_deg[nid] == 0]
    result = []
    while queue:
        nid = queue.pop(0)
        result.append(nid)
        for neighbor in adj[nid]:
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)

    remaining = [nid for nid in node_ids if nid not in set(result)]
    result.extend(remaining)
    return result


def _long_call_chains(store, all_nodes, calls_adj, nodes_by_id):
    """Find linear sequences where each node has exactly one outgoing calls edge."""
    used: set[str] = set()

    for node in all_nodes:
        nid = node["id"]
        if nid in used or node["kind"] == "file":
            continue

        chain = _trace_linear_chain(nid, calls_adj, nodes_by_id, used)
        if len(chain) >= 3:
            used.update(chain)
            start_name = _short_name(chain[0])
            end_name = _short_name(chain[-1])
            name = f"{start_name}_to_{end_name}"
            slug = f"chain_{name}".replace(".", "_").replace("::", "_")

            store.add_concept_path(
                id=slug, name=name,
                node_ids=chain, path_type="call_chain",
            )


def _trace_linear_chain(start_id, calls_adj, nodes_by_id, used):
    """Trace a linear chain of single-outgoing-calls from start_id."""
    chain = [start_id]
    visited = {start_id}
    current = start_id

    while True:
        targets = calls_adj.get(current, [])
        if len(targets) != 1:
            break

        next_id = targets[0]
        if next_id in visited or next_id in used:
            break

        node = nodes_by_id.get(next_id)
        if not node or node["kind"] == "file":
            break

        chain.append(next_id)
        visited.add(next_id)
        current = next_id

    return chain


def _short_name(node_id: str) -> str:
    if "::" in node_id:
        return node_id.split("::")[-1].replace(".", "_")
    return node_id.replace("/", "_").replace(".", "_")
