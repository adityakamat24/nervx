"""Integration tests: parse fixture project, link, store, verify."""

import os

import pytest

from nervx.memory.store import GraphStore
from nervx.perception.linker import resolve_all
from nervx.perception.parser import extract_keywords, parse_file


FIXTURE_ROOT = os.path.join(os.path.dirname(__file__), "fixtures", "sandbox")


def _parse_fixture():
    """Parse all .py files in the fixture project."""
    results = []
    for dirpath, _, files in os.walk(FIXTURE_ROOT):
        for f in files:
            if f.endswith(".py"):
                full = os.path.join(dirpath, f)
                results.append(parse_file(full, FIXTURE_ROOT))
    return results


@pytest.fixture
def fixture_graph():
    """Parse fixture, resolve edges, store everything in a GraphStore."""
    parse_results = _parse_fixture()
    edges = resolve_all(parse_results)

    store = GraphStore(":memory:")

    # Store nodes and keywords
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

    # Store edges
    for e in edges:
        store.add_edge(e.source_id, e.target_id, e.edge_type,
                       e.weight, e.metadata)

    yield store
    store.close()


# ── Basic counts ──────────────────────────────────────────────────

def test_fixture_has_files(fixture_graph):
    file_nodes = fixture_graph.get_nodes_by_kind("file")
    assert len(file_nodes) >= 15  # at least 15 .py files


def test_fixture_has_classes(fixture_graph):
    classes = fixture_graph.get_nodes_by_kind("class")
    names = {c["name"] for c in classes}
    assert "BaseAgent" in names
    assert "ClaudeAgent" in names
    assert "GPTAgent" in names
    assert "EventBus" in names
    assert "GameState" in names
    assert "VotingSystem" in names


def test_fixture_has_functions(fixture_graph):
    funcs = fixture_graph.get_nodes_by_kind("function")
    names = {f["name"] for f in funcs}
    assert "create_agent" in names
    assert "main" in names


def test_fixture_has_methods(fixture_graph):
    methods = fixture_graph.get_nodes_by_kind("method")
    names = {m["name"] for m in methods}
    assert "decide" in names
    assert "connect" in names
    assert "emit" in names
    assert "subscribe" in names


# ── Inheritance edges ─────────────────────────────────────────────

def test_inheritance_edges(fixture_graph):
    inherits_edges = fixture_graph.get_edges_by_type("inherits")
    sources = {e["source_id"] for e in inherits_edges}
    targets = {e["target_id"] for e in inherits_edges}

    # ClaudeAgent and GPTAgent inherit from BaseAgent
    assert any("ClaudeAgent" in s for s in sources)
    assert any("GPTAgent" in s for s in sources)
    assert any("BaseAgent" in t for t in targets)

    # Reverse edges should exist
    inherited_by = fixture_graph.get_edges_by_type("inherited_by")
    assert len(inherited_by) >= 2


# ── Import edges ──────────────────────────────────────────────────

def test_import_edges_exist(fixture_graph):
    import_edges = fixture_graph.get_edges_by_type("imports")
    assert len(import_edges) > 0


# ── Call edges ────────────────────────────────────────────────────

def test_call_edges_exist(fixture_graph):
    call_edges = fixture_graph.get_edges_by_type("calls")
    assert len(call_edges) > 0


def test_factory_has_callers(fixture_graph):
    """create_agent should have called_by edges."""
    factory_nodes = fixture_graph.get_nodes_by_name("create_agent")
    assert len(factory_nodes) >= 1
    factory_id = factory_nodes[0]["id"]
    called_by = [e for e in fixture_graph.get_edges_by_type("called_by")
                 if e["source_id"] == factory_id]
    assert len(called_by) >= 1


# ── Keyword search ────────────────────────────────────────────────

def test_keyword_search_agent(fixture_graph):
    results = fixture_graph.search_keywords(["agent"])
    assert len(results) > 0
    ids = {r[0] for r in results}
    assert any("BaseAgent" in id for id in ids)


def test_keyword_search_voting(fixture_graph):
    results = fixture_graph.search_keywords(["voting"])
    assert len(results) > 0
