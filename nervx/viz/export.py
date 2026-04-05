"""Export graph data to JSON for visualization."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from nervx.memory.store import GraphStore

# Forward edge types to keep (drop reverse edges to cut array in half)
_FORWARD_EDGE_TYPES = frozenset({"calls", "imports", "inherits", "instantiates"})


def export_viz_data(store: GraphStore, max_nodes: int = 1000) -> dict:
    """Export full graph data for visualization.

    Returns a dict matching the nervx-viz.json schema.
    All data is bulk-loaded for performance (<1 second target).

    For large repos (more nodes than max_nodes), only the top N nodes
    by importance are exported. Edges, patterns, concept paths, and
    fragile zones are filtered to reference only kept nodes.
    """
    # ── Bulk load everything ────────────────────────────────────
    all_nodes = store.get_all_nodes()
    all_edges = store.get_all_edges()
    all_file_stats = store.get_all_file_stats()
    all_patterns = store.get_all_patterns()
    all_concept_paths = store.get_concept_paths()
    all_cochanges = store.get_all_cochanges()
    conflict_ids = set(store.get_contract_conflicts())

    # ── Truncate large graphs ──────────────────────────────────
    total_node_count = len(all_nodes)
    truncated = False
    if max_nodes > 0 and total_node_count > max_nodes:
        truncated = True
        # Sort by importance descending, keep top max_nodes (skip file nodes from the cut)
        non_file = [n for n in all_nodes if n["kind"] != "file"]
        file_nodes = [n for n in all_nodes if n["kind"] == "file"]
        non_file.sort(key=lambda n: -(n["importance"] or 0))
        kept = non_file[:max_nodes]
        # Include file nodes only if they're parents of kept nodes
        kept_ids = {n["id"] for n in kept}
        kept_file_ids = {n["parent_id"] for n in kept if n.get("parent_id")} & {n["id"] for n in file_nodes}
        for fn in file_nodes:
            if fn["id"] in kept_file_ids:
                kept.append(fn)
        all_nodes = kept

    # ── Build in-memory indexes ─────────────────────────────────
    nodes_by_id: dict[str, dict] = {n["id"]: n for n in all_nodes}

    edges_to: dict[str, list[dict]] = defaultdict(list)  # target -> edges
    edges_from: dict[str, list[dict]] = defaultdict(list)  # source -> edges
    # If truncated, only keep edges between kept nodes
    if truncated:
        kept_node_ids = set(nodes_by_id.keys())
        all_edges = [e for e in all_edges if e["source_id"] in kept_node_ids and e["target_id"] in kept_node_ids]
    for e in all_edges:
        edges_from[e["source_id"]].append(e)
        edges_to[e["target_id"]].append(e)

    file_stats_by_path: dict[str, dict] = {s["file_path"]: s for s in all_file_stats}
    patterns_by_node: dict[str, list[dict]] = defaultdict(list)
    for p in all_patterns:
        patterns_by_node[p["node_id"]].append(p)

    # ── Compute modules ─────────────────────────────────────────
    module_nodes: dict[str, list[dict]] = defaultdict(list)
    node_module: dict[str, str] = {}
    for n in all_nodes:
        fp = n["file_path"]
        mod = fp.split("/")[0] if "/" in fp else "(root)"
        module_nodes[mod].append(n)
        node_module[n["id"]] = mod

    # ── Pre-compute warnings per node ───────────────────────────
    warnings_by_node: dict[str, list[str]] = defaultdict(list)
    _compute_all_warnings(
        all_nodes, nodes_by_id, edges_from, edges_to,
        file_stats_by_path, patterns_by_node, conflict_ids,
        warnings_by_node,
    )

    # ── Tech stack ──────────────────────────────────────────────
    try:
        from nervx.attention.briefing import _detect_tech_stack
        tech_str = _detect_tech_stack(store)
        tech_stack = [t.strip() for t in tech_str.split(",") if t.strip()]
    except Exception:
        tech_stack = []

    # ── Assemble JSON structure ─────────────────────────────────
    repo_name = store.get_meta("repo_root") or ""
    repo_name = Path(repo_name).name if repo_name else "unknown"

    meta = {
        "repo_name": repo_name,
        "node_count": int(store.get_meta("node_count") or 0),
        "edge_count": int(store.get_meta("edge_count") or 0),
        "build_time": store.get_meta("last_build") or "",
        "stack": tech_stack,
        "truncated": truncated,
        "total_node_count": total_node_count,
        "viz_node_count": len(all_nodes),
    }

    modules = _build_modules(module_nodes, file_stats_by_path)

    nodes = []
    for n in all_nodes:
        tags = json.loads(n["tags"]) if isinstance(n["tags"], str) else (n["tags"] or [])
        nodes.append({
            "id": n["id"],
            "name": n["name"],
            "kind": n["kind"],
            "file": n["file_path"],
            "line_start": n["line_start"],
            "line_end": n["line_end"],
            "signature": n["signature"] or "",
            "docstring": n["docstring"] or "",
            "tags": tags,
            "importance": n["importance"] or 0.0,
            "module": node_module.get(n["id"], "(root)"),
            "parent": n["parent_id"] or "",
            "warnings": warnings_by_node.get(n["id"], []),
        })

    edges = []
    for e in all_edges:
        if e["edge_type"] in _FORWARD_EDGE_TYPES:
            edges.append({
                "source": e["source_id"],
                "target": e["target_id"],
                "type": e["edge_type"],
                "weight": e["weight"] or 1.0,
            })

    cochanges = []
    for c in all_cochanges:
        cochanges.append({
            "file_a": c["file_a"],
            "file_b": c["file_b"],
            "coupling": c["coupling_score"],
            "co_commits": c["co_commit_count"],
        })

    concept_paths = []
    for p in all_concept_paths:
        node_ids = json.loads(p["node_ids"]) if isinstance(p["node_ids"], str) else (p["node_ids"] or [])
        # If truncated, filter to paths where all nodes are present
        if truncated:
            node_ids = [nid for nid in node_ids if nid in nodes_by_id]
            if len(node_ids) < 2:
                continue
        concept_paths.append({
            "id": p["id"],
            "name": p["name"],
            "type": p["path_type"],
            "node_ids": node_ids,
        })

    patterns = []
    for p in all_patterns:
        if truncated and p["node_id"] not in nodes_by_id:
            continue
        detail = json.loads(p["detail"]) if isinstance(p["detail"], str) else (p["detail"] or {})
        patterns.append({
            "node_id": p["node_id"],
            "pattern": p["pattern"],
            "implication": p["implication"] or "",
            "detail": detail,
        })

    hotspots = []
    for s in all_file_stats:
        if s["commits_30d"] and s["commits_30d"] > 0:
            hotspots.append({
                "file": s["file_path"],
                "commits_30d": s["commits_30d"],
                "commits_7d": s["commits_7d"] or 0,
                "primary_author": s["primary_author"] or "",
                "author_count": s["author_count"] or 0,
            })
    hotspots.sort(key=lambda h: -h["commits_30d"])

    fragile_zones = _build_fragile_zones(all_nodes, nodes_by_id, edges_from, edges_to, conflict_ids, warnings_by_node)

    contract_conflicts = _build_contract_conflicts(store, conflict_ids)

    return {
        "meta": meta,
        "modules": modules,
        "nodes": nodes,
        "edges": edges,
        "cochanges": cochanges,
        "concept_paths": concept_paths,
        "patterns": patterns,
        "hotspots": hotspots,
        "fragile_zones": fragile_zones,
        "contract_conflicts": contract_conflicts,
    }


def write_viz_json(data: dict, output_path: str) -> None:
    """Serialize viz data dict to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Module building ─────────────────────────────────────────────


def _build_modules(
    module_nodes: dict[str, list[dict]],
    file_stats_by_path: dict[str, dict],
) -> list[dict]:
    """Build module records with aggregated stats."""
    modules = []
    for mod_id in sorted(module_nodes.keys()):
        nodes = module_nodes[mod_id]
        files = set()
        test_count = 0
        testable_count = 0
        hotspot_score = 0

        for n in nodes:
            files.add(n["file_path"])
            tags = json.loads(n["tags"]) if isinstance(n["tags"], str) else (n["tags"] or [])
            if n["kind"] != "file":
                testable_count += 1
                if "test" in tags:
                    test_count += 1

        # Hotspot: sum commits_30d for files in this module
        for fp in files:
            stats = file_stats_by_path.get(fp)
            if stats and stats["commits_30d"]:
                hotspot_score += stats["commits_30d"]

        # Describe module
        desc = _describe_module(nodes)
        test_coverage = test_count / testable_count if testable_count > 0 else 0.0

        modules.append({
            "id": mod_id,
            "label": f"{mod_id}/",
            "description": desc,
            "file_count": len(files),
            "node_count": len(nodes),
            "hotspot_score": hotspot_score,
            "test_coverage": round(test_coverage, 2),
        })

    return modules


def _describe_module(nodes: list[dict]) -> str:
    """Auto-describe a module based on dominant tags and kinds."""
    from collections import Counter
    tag_counts: Counter = Counter()
    kind_counts: Counter = Counter()

    for n in nodes:
        kind_counts[n["kind"]] += 1
        tags = json.loads(n["tags"]) if isinstance(n["tags"], str) else (n["tags"] or [])
        for t in tags:
            if not t.startswith("extends:"):
                tag_counts[t] += 1

    total = len(nodes)
    if total == 0:
        return "module"
    if tag_counts.get("test", 0) > total * 0.3:
        return "tests"
    if tag_counts.get("route_handler", 0) > total * 0.3:
        return "API routes/handlers"
    if tag_counts.get("data_model", 0) > total * 0.3:
        return "data models"

    n_classes = kind_counts.get("class", 0)
    n_funcs = kind_counts.get("function", 0) + kind_counts.get("method", 0)
    parts = []
    if n_classes:
        parts.append(f"{n_classes} classes")
    if n_funcs:
        parts.append(f"{n_funcs} functions")
    return ", ".join(parts) if parts else "module"


# ── Warning pre-computation ─────────────────────────────────────


def _compute_all_warnings(
    all_nodes: list[dict],
    nodes_by_id: dict[str, dict],
    edges_from: dict[str, list[dict]],
    edges_to: dict[str, list[dict]],
    file_stats_by_path: dict[str, dict],
    patterns_by_node: dict[str, list[dict]],
    conflict_ids: set[str],
    warnings_by_node: dict[str, list[str]],
):
    """Compute warnings for all nodes using in-memory data.

    Warning categories (matching reflexes/warnings.py):
    - HIGH_IMPACT: 5+ callers
    - CONTRACT_CONFLICT: callers disagree on error handling
    - NO_TESTS: importance > 5, no test coverage
    - ACTIVE_CHURN: 3+ commits in last 7 days
    - MULTI_AUTHOR: 3+ contributors
    - PATTERN: detected architectural pattern
    """
    for n in all_nodes:
        if n["kind"] == "file":
            continue

        nid = n["id"]
        tags = json.loads(n["tags"]) if isinstance(n["tags"], str) else (n["tags"] or [])
        importance = n["importance"] or 0.0

        # HIGH_IMPACT: 5+ callers (called_by edges)
        called_by = [e for e in edges_from.get(nid, []) if e["edge_type"] == "called_by"]
        if len(called_by) >= 5:
            warnings_by_node[nid].append(
                f"HIGH_IMPACT: {n['name']} has {len(called_by)} callers."
            )

        # CONTRACT_CONFLICT
        if nid in conflict_ids:
            warnings_by_node[nid].append(
                f"CONTRACT_CONFLICT: Callers disagree on how to handle {n['name']}."
            )

        # NO_TESTS: importance > 5, no test node in edges
        if importance > 5 and "test" not in tags:
            has_test = False
            all_connected = edges_from.get(nid, []) + edges_to.get(nid, [])
            for e in all_connected:
                other_id = e["target_id"] if e["source_id"] == nid else e["source_id"]
                other = nodes_by_id.get(other_id)
                if other:
                    otags = json.loads(other["tags"]) if isinstance(other["tags"], str) else (other["tags"] or [])
                    if "test" in otags:
                        has_test = True
                        break
            if not has_test:
                warnings_by_node[nid].append(
                    f"NO_TESTS: {n['name']} (importance={importance:.0f}) has no test coverage."
                )

        # ACTIVE_CHURN: 3+ commits in 7 days
        file_stats = file_stats_by_path.get(n["file_path"])
        if file_stats and file_stats.get("commits_7d", 0) >= 3:
            warnings_by_node[nid].append(
                f"ACTIVE_CHURN: {n['file_path']} has {file_stats['commits_7d']} commits in 7 days."
            )

        # MULTI_AUTHOR: 3+ contributors
        if file_stats and file_stats.get("author_count", 0) >= 3:
            warnings_by_node[nid].append(
                f"MULTI_AUTHOR: {n['file_path']} has {file_stats['author_count']} contributors."
            )

        # PATTERN: architectural pattern detected
        for pat in patterns_by_node.get(nid, []):
            warnings_by_node[nid].append(
                f"PATTERN: {pat['pattern'].upper()} - {pat['implication']}"
            )


# ── Fragile zones ──────────────────────────────────────────────


def _build_fragile_zones(
    all_nodes: list[dict],
    nodes_by_id: dict[str, dict],
    edges_from: dict[str, list[dict]],
    edges_to: dict[str, list[dict]],
    conflict_ids: set[str],
    warnings_by_node: dict[str, list[str]],
) -> list[dict]:
    """Identify fragile zones: high-importance nodes with serious warnings.

    A node qualifies as fragile if:
    - importance >= 25 with any warning, OR
    - importance >= 15 with a serious warning (HIGH_IMPACT or CONTRACT_CONFLICT), OR
    - importance >= 15 with multiple warning types

    Capped at 30 to avoid alert fatigue.
    """
    zones = []
    for n in all_nodes:
        if n["kind"] == "file":
            continue
        importance = n["importance"] or 0.0
        if importance < 15:
            continue

        node_warnings = warnings_by_node.get(n["id"], [])
        reasons = []
        has_serious = False
        for w in node_warnings:
            if w.startswith("NO_TESTS:"):
                reasons.append("no test coverage")
            elif w.startswith("CONTRACT_CONFLICT:"):
                reasons.append("contract conflict")
                has_serious = True
            elif w.startswith("HIGH_IMPACT:"):
                reasons.append("high caller count")
                has_serious = True

        if not reasons:
            continue

        # Gate: importance >= 25 always qualifies; 15-25 needs serious or multiple
        if importance < 25 and not has_serious and len(reasons) < 2:
            continue

        zones.append({
            "node_id": n["id"],
            "importance": importance,
            "reasons": reasons,
        })

    zones.sort(key=lambda z: -z["importance"])
    return zones[:30]


# ── Contract conflicts ──────────────────────────────────────────


def _build_contract_conflicts(store: GraphStore, conflict_ids: set[str]) -> list[dict]:
    """Build contract conflict records with caller strategies."""
    conflicts = []
    for fid in conflict_ids:
        contracts = store.get_contracts_for_function(fid)
        if not contracts:
            continue

        error_strategies = set()
        return_strategies = set()
        for c in contracts:
            if c["error_handling"] and c["error_handling"] != "none":
                error_strategies.add(c["error_handling"])
            if c["return_usage"]:
                return_strategies.add(c["return_usage"])

        conflicts.append({
            "function_id": fid,
            "caller_count": len(contracts),
            "error_strategies": sorted(error_strategies),
            "return_strategies": sorted(return_strategies),
        })

    return conflicts
