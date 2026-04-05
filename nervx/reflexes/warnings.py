"""Warning system, contract analysis, and blast radius computation."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from nervx.memory.store import GraphStore


@dataclass
class Warning:
    category: str  # HIGH_IMPACT, CONTRACT_CONFLICT, NO_TESTS, etc.
    node_id: str
    message: str


@dataclass
class BlastRadius:
    target_id: str
    direct: list[str] = field(default_factory=list)    # depth 1
    indirect: list[str] = field(default_factory=list)   # depth 2
    distant: list[str] = field(default_factory=list)    # depth 3
    temporal: list[str] = field(default_factory=list)    # co-change files
    tests: list[str] = field(default_factory=list)       # test nodes
    total_affected: int = 0


def collect_warnings(store: GraphStore, node_ids: list[str]) -> list[Warning]:
    """Collect all warnings for a set of nodes."""
    warnings: list[Warning] = []

    for nid in node_ids:
        node = store.get_node(nid)
        if not node:
            continue

        # HIGH_IMPACT: 5+ callers
        called_by = [e for e in store.get_edges_to(nid) if e["edge_type"] == "called_by"]
        # Wait — called_by edges have source=nid, target=caller (reverse convention)
        # Actually let's check: if A calls B, we store (A,B,calls) and (B,A,called_by)
        # So for B, called_by edges FROM B have targets = callers
        called_by_edges = [e for e in store.get_edges_from(nid) if e["edge_type"] == "called_by"]
        if len(called_by_edges) >= 5:
            callers = []
            for e in called_by_edges[:8]:
                cn = store.get_node(e["target_id"])
                if cn:
                    callers.append(cn["name"])
            warnings.append(Warning(
                "HIGH_IMPACT", nid,
                f"{node['name']} has {len(called_by_edges)} callers. "
                f"Changes here affect: {', '.join(callers)}",
            ))

        # CONTRACT_CONFLICT
        if nid in store.get_contract_conflicts():
            contracts = store.get_contracts_for_function(nid)
            strategies = set()
            for c in contracts:
                strategies.add(c["error_handling"])
            warnings.append(Warning(
                "CONTRACT_CONFLICT", nid,
                f"Callers of {node['name']} disagree on error handling: "
                f"{', '.join(strategies)}",
            ))

        # NO_TESTS: important node with no test coverage
        tags = json.loads(node["tags"]) if isinstance(node["tags"], str) else node["tags"]
        if node["importance"] > 5 and "test" not in tags:
            has_test = False
            # Check if any test node has an edge to this node
            all_edges = store.get_edges_to(nid) + store.get_edges_from(nid)
            for e in all_edges:
                other_id = e["target_id"] if e["source_id"] == nid else e["source_id"]
                other = store.get_node(other_id)
                if other:
                    other_tags = json.loads(other["tags"]) if isinstance(other["tags"], str) else other["tags"]
                    if "test" in other_tags:
                        has_test = True
                        break
            if not has_test:
                warnings.append(Warning(
                    "NO_TESTS", nid,
                    f"{node['name']} (importance={node['importance']:.1f}) has no test coverage.",
                ))

        # ACTIVE_CHURN
        file_stats = store.get_file_stats(node["file_path"])
        if file_stats and file_stats["commits_7d"] >= 3:
            warnings.append(Warning(
                "ACTIVE_CHURN", nid,
                f"{node['file_path']} changed {file_stats['commits_7d']} times in the last 7 days. "
                f"Check for in-progress work.",
            ))

        # MULTI_AUTHOR
        if file_stats and file_stats["author_count"] >= 3:
            warnings.append(Warning(
                "MULTI_AUTHOR", nid,
                f"{node['file_path']} has {file_stats['author_count']} contributors.",
            ))

        # TEMPORAL_COUPLING
        cochanges = store.get_cochanges_for_file(node["file_path"])
        coupled_files = []
        for cc in cochanges:
            if cc["coupling_score"] >= 0.5:
                other = cc["file_b"] if cc["file_a"] == node["file_path"] else cc["file_a"]
                coupled_files.append(f"{other} ({int(cc['coupling_score'] * 100)}%)")
        if coupled_files:
            warnings.append(Warning(
                "TEMPORAL_COUPLING", nid,
                f"Changes to {node['file_path']} usually also require changes to: "
                f"{', '.join(coupled_files[:3])}",
            ))

        # PATTERN warnings
        patterns = store.get_patterns_for_node(nid)
        for p in patterns:
            warnings.append(Warning(
                "PATTERN", nid,
                p["implication"],
            ))

    return warnings


def analyze_contracts(store: GraphStore, parse_results=None):
    """Analyze caller error handling and return usage patterns.

    Uses raw_calls from parse_results to populate the contracts table.
    """
    if parse_results is None:
        return

    from nervx.perception.linker import build_symbol_index, _resolve_single_call, _build_import_map

    symbol_index = build_symbol_index(parse_results)
    import_map = _build_import_map(parse_results)

    for pr in parse_results:
        for rc in pr.raw_calls:
            target = _resolve_single_call(rc, pr.file_path, symbol_index, import_map)
            if target is None:
                continue

            error_handling = "none"
            if rc.error_handling:
                pattern = rc.error_handling.get("pattern", "try_except")
                exc = rc.error_handling.get("exception", "")
                error_handling = f"{pattern}:{exc}" if exc else pattern

            store.add_contract(
                function_id=target.id,
                caller_id=rc.caller_id,
                error_handling=error_handling,
                return_usage=rc.return_usage,
            )


def compute_blast_radius(
    store: GraphStore,
    node_id: str,
    max_depth: int = 3,
) -> BlastRadius:
    """Compute downstream impact of changing a symbol via BFS."""
    result = BlastRadius(target_id=node_id)

    visited: set[str] = {node_id}
    current_level = [node_id]

    for depth in range(1, max_depth + 1):
        next_level: list[str] = []

        for nid in current_level:
            # Follow called_by and inherited_by edges
            for e in store.get_edges_from(nid):
                if e["edge_type"] in ("called_by", "inherited_by"):
                    target = e["target_id"]
                    if target not in visited:
                        visited.add(target)
                        next_level.append(target)

        if depth == 1:
            result.direct = next_level[:]
        elif depth == 2:
            result.indirect = next_level[:]
        elif depth == 3:
            result.distant = next_level[:]

        current_level = next_level

    # Temporal: co-change files
    node = store.get_node(node_id)
    if node:
        cochanges = store.get_cochanges_for_file(node["file_path"])
        for cc in cochanges:
            if cc["coupling_score"] >= 0.3:
                other = cc["file_b"] if cc["file_a"] == node["file_path"] else cc["file_a"]
                result.temporal.append(other)

    # Tests: test nodes with edges to any symbol in the blast radius
    all_affected = set(result.direct + result.indirect + result.distant + [node_id])
    for nid in list(all_affected):
        all_edges = store.get_edges_from(nid) + store.get_edges_to(nid)
        for e in all_edges:
            other_id = e["target_id"] if e["source_id"] == nid else e["source_id"]
            other = store.get_node(other_id)
            if other:
                other_tags = json.loads(other["tags"]) if isinstance(other["tags"], str) else other["tags"]
                if "test" in other_tags and other_id not in result.tests:
                    result.tests.append(other_id)

    result.total_affected = len(set(result.direct + result.indirect + result.distant))
    return result
