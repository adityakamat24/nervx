"""Tests for architectural pattern detection."""

import os
import pytest

from nervx.memory.store import GraphStore
from nervx.perception.linker import resolve_all
from nervx.perception.parser import extract_keywords, parse_file
from nervx.instinct.patterns import detect_patterns
from nervx.build import compute_importance

FIXTURE_ROOT = os.path.join(os.path.dirname(__file__), "fixtures", "sandbox")


@pytest.fixture
def store_with_patterns():
    """Build a full graph from fixture and detect patterns."""
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

    yield store
    store.close()


def test_event_bus_detected(store_with_patterns):
    patterns = store_with_patterns.get_patterns_for_node("game/events.py::EventBus")
    pattern_names = {p["pattern"] for p in patterns}
    assert "event_bus" in pattern_names


def test_factory_detected(store_with_patterns):
    patterns = store_with_patterns.get_patterns_for_node("agents/factory.py::create_agent")
    pattern_names = {p["pattern"] for p in patterns}
    assert "factory" in pattern_names


def test_strategy_detected(store_with_patterns):
    patterns = store_with_patterns.get_patterns_for_node("agents/base_agent.py::BaseAgent")
    pattern_names = {p["pattern"] for p in patterns}
    assert "strategy" in pattern_names


def test_repository_detected(store_with_patterns):
    patterns = store_with_patterns.get_patterns_for_node("game/state.py::GameState")
    pattern_names = {p["pattern"] for p in patterns}
    assert "repository" in pattern_names


def test_singleton_detected(store_with_patterns):
    patterns = store_with_patterns.get_patterns_for_node("game/events.py::EventBus")
    pattern_names = {p["pattern"] for p in patterns}
    assert "singleton" in pattern_names


def test_pattern_count(store_with_patterns):
    all_patterns = store_with_patterns.get_all_patterns()
    assert len(all_patterns) >= 5  # event_bus, factory, strategy, repository, singleton
