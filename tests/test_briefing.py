"""Tests for briefing generator and concept paths."""

import os
import pytest

from nervx.memory.store import GraphStore
from nervx.perception.linker import resolve_all
from nervx.perception.parser import extract_keywords, parse_file
from nervx.instinct.patterns import detect_patterns
from nervx.reflexes.warnings import analyze_contracts
from nervx.build import compute_importance
from nervx.attention.concepts import detect_concept_paths
from nervx.attention.briefing import generate_briefing

FIXTURE_ROOT = os.path.join(os.path.dirname(__file__), "fixtures", "sandbox")


@pytest.fixture
def briefing_store():
    """Build full graph with all features for briefing testing."""
    parse_results = []
    for dirpath, _, files in os.walk(FIXTURE_ROOT):
        for f in files:
            if f.endswith(".py"):
                full = os.path.join(dirpath, f)
                parse_results.append(parse_file(full, FIXTURE_ROOT))

    store = GraphStore(":memory:")
    store.set_meta("node_count", "0")
    store.set_meta("edge_count", "0")

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
    detect_concept_paths(store)

    nc = len(store.get_all_nodes())
    ec = len(store.conn.execute("SELECT * FROM edges").fetchall())
    store.set_meta("node_count", str(nc))
    store.set_meta("edge_count", str(ec))

    yield store
    store.close()


# ── Concept paths ─────────────────────────────────────────────────

def test_concept_paths_detected(briefing_store):
    paths = briefing_store.get_concept_paths()
    assert len(paths) >= 1


def test_concept_path_has_entry_chain(briefing_store):
    paths = briefing_store.get_concept_paths()
    call_chains = [p for p in paths if p["path_type"] == "call_chain"]
    assert len(call_chains) >= 1


# ── Briefing ──────────────────────────────────────────────────────

def test_briefing_contains_module_map(briefing_store):
    briefing = generate_briefing(briefing_store, FIXTURE_ROOT)
    assert "## Module Map" in briefing
    assert "agents/" in briefing
    assert "game/" in briefing
    assert "memory/" in briefing


def test_briefing_contains_entry_points(briefing_store):
    briefing = generate_briefing(briefing_store, FIXTURE_ROOT)
    assert "## Entry Points" in briefing
    assert "main" in briefing


def test_briefing_contains_patterns(briefing_store):
    briefing = generate_briefing(briefing_store, FIXTURE_ROOT)
    assert "## Detected Patterns" in briefing
    assert "EVENT_BUS" in briefing
    assert "FACTORY" in briefing
    assert "STRATEGY" in briefing


def test_briefing_contains_key_flows(briefing_store):
    briefing = generate_briefing(briefing_store, FIXTURE_ROOT)
    assert "## Key Flows" in briefing


def test_briefing_contains_fragile_zones(briefing_store):
    briefing = generate_briefing(briefing_store, FIXTURE_ROOT)
    assert "## Fragile Zones" in briefing


def test_briefing_is_reasonable_length(briefing_store):
    briefing = generate_briefing(briefing_store, FIXTURE_ROOT)
    # Rough token estimate: words / 0.75
    words = len(briefing.split())
    # Should be in reasonable range (allow some margin)
    assert words < 1000
