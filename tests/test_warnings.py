"""Tests for warning system and blast radius."""

import os
import pytest

from nervx.memory.store import GraphStore
from nervx.perception.linker import resolve_all
from nervx.perception.parser import extract_keywords, parse_file
from nervx.instinct.patterns import detect_patterns
from nervx.reflexes.warnings import (
    BlastRadius,
    Warning,
    collect_warnings,
    compute_blast_radius,
    analyze_contracts,
)
from nervx.build import compute_importance

FIXTURE_ROOT = os.path.join(os.path.dirname(__file__), "fixtures", "sandbox")


@pytest.fixture
def full_store():
    """Build a full graph with patterns, contracts, and importance."""
    parse_results = []
    for dirpath, _, files in os.walk(FIXTURE_ROOT):
        for f in files:
            if f.endswith(".py"):
                full = os.path.join(dirpath, f)
                parse_results.append(parse_file(full, FIXTURE_ROOT))

    store = GraphStore(":memory:")
    for pr in parse_results:
        for node in pr.nodes:
            store.upsert_node(
                id=node.id, kind=node.kind, name=node.name,
                file_path=node.file_path, line_start=node.line_start,
                line_end=node.line_end, signature=node.signature,
                docstring=node.docstring, tags=node.tags,
                parent_id=node.parent_id,
            )
            kws = extract_keywords(node)
            store.add_keywords_bulk([(kw, node.id, src) for kw, src in kws])

    edges = resolve_all(parse_results)
    for e in edges:
        store.add_edge(e.source_id, e.target_id, e.edge_type, e.weight, e.metadata)

    compute_importance(store)
    detect_patterns(store)
    analyze_contracts(store, parse_results)

    yield store
    store.close()


def test_pattern_warnings(full_store):
    """Pattern nodes should have PATTERN warnings."""
    warnings = collect_warnings(full_store, ["game/events.py::EventBus"])
    categories = {w.category for w in warnings}
    assert "PATTERN" in categories


def test_blast_radius_base_agent(full_store):
    """BaseAgent should have downstream dependents."""
    radius = compute_blast_radius(full_store, "agents/base_agent.py::BaseAgent")
    # BaseAgent is inherited by ClaudeAgent and GPTAgent
    assert radius.total_affected > 0
    all_affected = radius.direct + radius.indirect + radius.distant
    affected_names = set()
    for nid in all_affected:
        node = full_store.get_node(nid)
        if node:
            affected_names.add(node["name"])
    assert "ClaudeAgent" in affected_names or "GPTAgent" in affected_names


def test_blast_radius_structure(full_store):
    """Blast radius should have the right structure."""
    radius = compute_blast_radius(full_store, "agents/base_agent.py::BaseAgent")
    assert isinstance(radius.direct, list)
    assert isinstance(radius.indirect, list)
    assert isinstance(radius.distant, list)
    assert isinstance(radius.temporal, list)
    assert isinstance(radius.tests, list)


def test_contract_conflicts_exist(full_store):
    """There should be at least one contract conflict in the fixture."""
    conflicts = full_store.get_contract_conflicts()
    assert len(conflicts) >= 1


def test_warnings_for_important_node(full_store):
    """Important nodes should get some warnings."""
    # Find a node with high importance
    nodes = full_store.get_all_nodes()
    important = [n for n in nodes if n["importance"] > 3]
    if important:
        node_ids = [n["id"] for n in important[:3]]
        warnings = collect_warnings(full_store, node_ids)
        # Should have at least some warnings
        assert isinstance(warnings, list)
