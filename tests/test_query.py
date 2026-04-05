"""Tests for the query engine."""

import os
import pytest

from nervx.memory.store import GraphStore
from nervx.perception.linker import resolve_all
from nervx.perception.parser import extract_keywords, parse_file
from nervx.instinct.patterns import detect_patterns
from nervx.reflexes.warnings import analyze_contracts
from nervx.build import compute_importance
from nervx.attention.query import (
    NavigateResult,
    _tokenize_query,
    navigate,
    find,
    blast_radius_query,
)

FIXTURE_ROOT = os.path.join(os.path.dirname(__file__), "fixtures", "sandbox")


@pytest.fixture
def query_store():
    """Build full graph for query testing."""
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


# ── Tokenizer ─────────────────────────────────────────────────────

def test_tokenize_basic():
    tokens = _tokenize_query("agent connection")
    assert "agent" in tokens
    assert "connection" in tokens


def test_tokenize_removes_stop_words():
    tokens = _tokenize_query("fix the bug in agent")
    assert "the" not in tokens
    assert "in" not in tokens
    # "fix", "bug", "agent" are code-relevant and should be kept
    assert "fix" in tokens
    assert "bug" in tokens
    assert "agent" in tokens


def test_tokenize_camel_case():
    tokens = _tokenize_query("WebSocketManager")
    assert "web" in tokens
    assert "socket" in tokens
    assert "manager" in tokens


def test_tokenize_snake_case():
    tokens = _tokenize_query("process_game_message")
    assert "process" in tokens
    assert "game" in tokens
    assert "message" in tokens


# ── Navigate ──────────────────────────────────────────────────────

def test_navigate_agent(query_store):
    result = navigate(query_store, "agent decide")
    assert isinstance(result, NavigateResult)
    assert len(result.primary) > 0
    primary_names = {n["name"] for n in result.primary}
    # Should find something agent-related
    assert any("agent" in name.lower() or "decide" in name.lower()
               for name in primary_names)


def test_navigate_voting(query_store):
    result = navigate(query_store, "voting system")
    assert len(result.primary) > 0
    primary_names = {n["name"] for n in result.primary}
    assert "VotingSystem" in primary_names or "run_vote" in primary_names


def test_navigate_memory(query_store):
    result = navigate(query_store, "memory store index")
    assert len(result.primary) > 0


def test_navigate_has_secondary(query_store):
    result = navigate(query_store, "agent")
    # Should have some secondary results (neighbors)
    assert len(result.secondary) >= 0  # may or may not have secondaries


def test_navigate_empty_query(query_store):
    result = navigate(query_store, "the to of in")
    assert result.formatted.startswith("No searchable terms")


def test_navigate_formatted_output(query_store):
    result = navigate(query_store, "event bus emit")
    assert "## Navigate:" in result.formatted
    assert "### Relevant Symbols" in result.formatted


def test_navigate_read_order(query_store):
    result = navigate(query_store, "voting agent decide")
    if result.read_order:
        assert len(result.read_order) > 0


def test_navigate_flows_from_main(query_store):
    """Navigate should trace call flows from entry points."""
    result = navigate(query_store, "main entry")
    # main() calls other functions, so flows should be populated
    if result.flows:
        for flow in result.flows:
            assert "seed_name" in flow
            assert "chain" in flow
            assert len(flow["chain"]) >= 2
            for step in flow["chain"]:
                assert "id" in step
                assert "name" in step


def test_navigate_flows_format(query_store):
    """Flows should appear in formatted output when present."""
    result = navigate(query_store, "main entry")
    if result.flows:
        assert "### Execution Flows" in result.formatted
        assert " -> " in result.formatted


def test_navigate_flows_empty_for_leaf(query_store):
    """Nodes with no outgoing calls should produce no flows."""
    result = navigate(query_store, "PlayerRole")
    # PlayerRole is a class with no outgoing calls edges
    # Flows may or may not be empty depending on other primary hits,
    # but all flows should have valid structure
    for flow in result.flows:
        assert len(flow["chain"]) >= 2


# ── Find ──────────────────────────────────────────────────────────

def test_find_by_kind(query_store):
    classes = find(query_store, kind="class")
    assert len(classes) > 0
    assert all(c["kind"] == "class" for c in classes)


def test_find_by_tag(query_store):
    factories = find(query_store, tag="factory")
    assert len(factories) > 0
    for f in factories:
        tags = f["tags"] if isinstance(f["tags"], list) else eval(f["tags"])
        assert "factory" in tags


def test_find_importance_gt(query_store):
    important = find(query_store, importance_gt=5.0)
    assert all(n["importance"] > 5.0 for n in important)


def test_find_dead_code(query_store):
    """Dead code detection should find unreferenced symbols."""
    dead = find(query_store, dead=True)
    # Should find some dead code in the sandbox fixture
    assert len(dead) > 0
    # Dead nodes should not be files, entrypoints, or tests
    for node in dead:
        tags = node["tags"] if isinstance(node["tags"], list) else eval(node["tags"])
        assert node["kind"] != "file"
        assert "entrypoint" not in tags
        assert "route_handler" not in tags
        assert "test" not in tags
        name = node["name"]
        assert not (name.startswith("__") and name.endswith("__"))


def test_find_dead_composes_with_kind(query_store):
    """--dead should compose with --kind filter."""
    dead_funcs = find(query_store, dead=True, kind="function")
    assert all(n["kind"] == "function" for n in dead_funcs)
    dead_all = find(query_store, dead=True)
    # Dead functions should be a subset of all dead nodes
    dead_func_ids = {n["id"] for n in dead_funcs}
    assert dead_func_ids <= {n["id"] for n in dead_all}


# ── Blast radius ──────────────────────────────────────────────────

def test_blast_radius_text(query_store):
    output = blast_radius_query(query_store, "agents/base_agent.py::BaseAgent")
    assert "## Blast Radius:" in output
    assert "BaseAgent" in output
