"""Tests for GraphStore."""

import pytest

from nervx.memory.store import GraphStore


@pytest.fixture
def store():
    with GraphStore(":memory:") as s:
        yield s


def test_schema_creates_cleanly(store):
    tables = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {r["name"] for r in tables}
    expected = {
        "nodes", "edges", "cochanges", "keywords", "file_stats",
        "file_hashes", "concept_paths", "patterns", "contracts", "meta",
    }
    assert expected.issubset(table_names)


def test_upsert_and_get_node(store):
    store.upsert_node(
        id="app.py::main",
        kind="function",
        name="main",
        file_path="app.py",
        line_start=1,
        line_end=10,
        signature="main() -> None",
        tags=["entrypoint"],
    )
    node = store.get_node("app.py::main")
    assert node is not None
    assert node["kind"] == "function"
    assert node["name"] == "main"
    assert node["signature"] == "main() -> None"
    assert '"entrypoint"' in node["tags"]


def test_get_node_not_found(store):
    assert store.get_node("nonexistent") is None


def test_get_nodes_by_file(store):
    store.upsert_node(id="a.py", kind="file", name="a.py", file_path="a.py")
    store.upsert_node(id="a.py::foo", kind="function", name="foo", file_path="a.py")
    store.upsert_node(id="b.py::bar", kind="function", name="bar", file_path="b.py")
    nodes = store.get_nodes_by_file("a.py")
    assert len(nodes) == 2
    ids = {n["id"] for n in nodes}
    assert ids == {"a.py", "a.py::foo"}


def test_get_nodes_by_kind(store):
    store.upsert_node(id="a.py::Foo", kind="class", name="Foo", file_path="a.py")
    store.upsert_node(id="a.py::bar", kind="function", name="bar", file_path="a.py")
    classes = store.get_nodes_by_kind("class")
    assert len(classes) == 1
    assert classes[0]["name"] == "Foo"


def test_add_edge_creates_reverse_pair(store):
    store.upsert_node(id="a.py::foo", kind="function", name="foo", file_path="a.py")
    store.upsert_node(id="b.py::bar", kind="function", name="bar", file_path="b.py")
    store.add_edge("a.py::foo", "b.py::bar", "calls")

    forward = store.get_edges_from("a.py::foo")
    assert any(e["edge_type"] == "calls" and e["target_id"] == "b.py::bar" for e in forward)

    reverse = store.get_edges_from("b.py::bar")
    assert any(e["edge_type"] == "called_by" and e["target_id"] == "a.py::foo" for e in reverse)


def test_instantiates_no_reverse(store):
    store.add_edge("a.py::foo", "b.py::Bar", "instantiates")
    forward = store.get_edges_from("a.py::foo")
    assert len(forward) == 1
    reverse = store.get_edges_from("b.py::Bar")
    assert len(reverse) == 0


def test_in_out_degree(store):
    store.upsert_node(id="a", kind="function", name="a", file_path="a.py")
    store.upsert_node(id="b", kind="function", name="b", file_path="b.py")
    store.upsert_node(id="c", kind="function", name="c", file_path="c.py")
    store.add_edge("a", "b", "calls")
    store.add_edge("c", "b", "calls")
    assert store.get_in_degree("b") >= 2
    assert store.get_out_degree("a") >= 1


def test_keyword_search(store):
    store.add_keywords_bulk([
        ("agent", "a.py::Agent", "name"),
        ("base", "a.py::Agent", "name"),
        ("agent", "b.py::create_agent", "name"),
        ("create", "b.py::create_agent", "name"),
    ])
    results = store.search_keywords(["agent"])
    assert len(results) == 2

    results = store.search_keywords(["agent", "base"])
    assert results[0][0] == "a.py::Agent"
    assert results[0][1] == 2


def test_keyword_search_empty(store):
    assert store.search_keywords([]) == []


def test_cochange_ordering(store):
    store.upsert_cochange("a.py", "b.py", 5, 10, 8, "2024-01-01", 0.5)
    store.upsert_cochange("a.py", "c.py", 2, 10, 5, "2024-01-02", 0.2)
    results = store.get_cochanges_for_file("a.py")
    assert len(results) == 2
    assert results[0]["coupling_score"] >= results[1]["coupling_score"]


def test_file_stats(store):
    store.upsert_file_stats("a.py", total_commits=10, commits_30d=3, commits_7d=1)
    stats = store.get_file_stats("a.py")
    assert stats is not None
    assert stats["total_commits"] == 10
    assert store.get_file_stats("nonexistent") is None


def test_file_hashes(store):
    store.upsert_file_hash("a.py", "abc123", "2024-01-01T00:00:00")
    h = store.get_file_hash("a.py")
    assert h["content_hash"] == "abc123"
    all_h = store.get_all_file_hashes()
    assert all_h == {"a.py": "abc123"}


def test_concept_paths(store):
    store.add_concept_path("flow_main_to_end", "main_to_end", ["a", "b", "c"], "call_chain")
    paths = store.get_concept_paths()
    assert len(paths) == 1
    assert paths[0]["name"] == "main_to_end"


def test_patterns(store):
    store.add_pattern("a.py::EventBus", "event_bus", {"listeners": 5}, "Static call graph incomplete")
    patterns = store.get_patterns_for_node("a.py::EventBus")
    assert len(patterns) == 1
    assert patterns[0]["pattern"] == "event_bus"
    all_p = store.get_all_patterns()
    assert len(all_p) == 1


def test_contracts_and_conflicts(store):
    store.add_contract("a.py::foo", "b.py::bar", "try_except:ValueError", "assigned")
    store.add_contract("a.py::foo", "c.py::baz", "none", "ignored")
    contracts = store.get_contracts_for_function("a.py::foo")
    assert len(contracts) == 2
    conflicts = store.get_contract_conflicts()
    assert "a.py::foo" in conflicts


def test_meta(store):
    store.set_meta("repo_root", "/tmp/test")
    assert store.get_meta("repo_root") == "/tmp/test"
    assert store.get_meta("nonexistent") is None


def test_clear_all(store):
    store.upsert_node(id="a.py::foo", kind="function", name="foo", file_path="a.py")
    store.add_keyword("foo", "a.py::foo", "name")
    store.set_meta("key", "val")
    store.clear_all()
    assert store.get_node("a.py::foo") is None
    assert store.get_meta("key") is None
    assert store.search_keywords(["foo"]) == []


def test_clear_file_data(store):
    store.upsert_node(id="a.py", kind="file", name="a.py", file_path="a.py")
    store.upsert_node(id="a.py::foo", kind="function", name="foo", file_path="a.py")
    store.upsert_node(id="b.py::bar", kind="function", name="bar", file_path="b.py")
    store.add_edge("a.py::foo", "b.py::bar", "calls")
    store.add_keyword("foo", "a.py::foo", "name")
    store.add_keyword("bar", "b.py::bar", "name")

    store.clear_file_data("a.py")

    assert store.get_node("a.py::foo") is None
    assert store.get_node("b.py::bar") is not None
    assert store.search_keywords(["foo"]) == []
    assert len(store.search_keywords(["bar"])) == 1
    assert store.get_edges_from("a.py::foo") == []
    assert store.get_edges_from("b.py::bar") == []
