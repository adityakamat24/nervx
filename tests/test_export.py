"""Tests for viz export module."""

import json
import os

import pytest

from nervx.memory.store import GraphStore
from nervx.perception.linker import resolve_all
from nervx.perception.parser import extract_keywords, parse_file
from nervx.instinct.patterns import detect_patterns
from nervx.reflexes.warnings import analyze_contracts
from nervx.build import compute_importance
from nervx.attention.concepts import detect_concept_paths
from nervx.viz.export import export_viz_data, write_viz_json

FIXTURE_ROOT = os.path.join(os.path.dirname(__file__), "fixtures", "sandbox")


@pytest.fixture
def viz_store():
    """Build full graph for viz export testing."""
    parse_results = []
    for dirpath, _, files in os.walk(FIXTURE_ROOT):
        for f in files:
            if f.endswith(".py"):
                full = os.path.join(dirpath, f)
                parse_results.append(parse_file(full, FIXTURE_ROOT))

    store = GraphStore(":memory:")
    store.set_meta("repo_root", FIXTURE_ROOT)
    store.set_meta("node_count", "0")
    store.set_meta("edge_count", "0")
    store.set_meta("last_build", "2026-04-04T00:00:00Z")

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


def test_export_returns_valid_structure(viz_store):
    """Export should return dict with all required top-level keys."""
    data = export_viz_data(viz_store)
    required_keys = {
        "meta", "modules", "nodes", "edges", "cochanges",
        "concept_paths", "patterns", "hotspots", "fragile_zones",
        "contract_conflicts",
    }
    assert required_keys == set(data.keys())


def test_export_meta_has_required_fields(viz_store):
    """Meta section should have repo_name, node_count, edge_count, build_time, stack."""
    data = export_viz_data(viz_store)
    meta = data["meta"]
    assert "repo_name" in meta
    assert meta["repo_name"] == "sandbox"
    assert isinstance(meta["node_count"], int)
    assert isinstance(meta["edge_count"], int)
    assert meta["node_count"] > 0
    assert "build_time" in meta
    assert isinstance(meta["stack"], list)


def test_export_nodes_have_required_fields(viz_store):
    """Each node should have all required fields."""
    data = export_viz_data(viz_store)
    assert len(data["nodes"]) > 0
    for node in data["nodes"]:
        assert "id" in node
        assert "name" in node
        assert "kind" in node
        assert "file" in node
        assert "line_start" in node
        assert "tags" in node
        assert isinstance(node["tags"], list)
        assert "importance" in node
        assert "module" in node
        assert "warnings" in node
        assert isinstance(node["warnings"], list)


def test_export_nodes_have_warnings(viz_store):
    """At least some nodes should have warnings pre-computed."""
    data = export_viz_data(viz_store)
    nodes_with_warnings = [n for n in data["nodes"] if n["warnings"]]
    # The sandbox fixture should produce some warnings
    assert len(nodes_with_warnings) > 0


def test_export_edges_are_forward_only(viz_store):
    """Edges should only contain forward types (calls, imports, inherits)."""
    data = export_viz_data(viz_store)
    reverse_types = {"called_by", "imported_by", "inherited_by"}
    for edge in data["edges"]:
        assert edge["type"] not in reverse_types
        assert "source" in edge
        assert "target" in edge
        assert "type" in edge
        assert "weight" in edge


def test_export_modules_group_by_directory(viz_store):
    """Modules should group nodes by top-level directory."""
    data = export_viz_data(viz_store)
    assert len(data["modules"]) > 0
    for mod in data["modules"]:
        assert "id" in mod
        assert "label" in mod
        assert "node_count" in mod
        assert mod["node_count"] > 0
        assert "file_count" in mod
        assert "hotspot_score" in mod
        assert "test_coverage" in mod


def test_export_concept_paths_parsed(viz_store):
    """Concept paths should have node_ids as a list (not JSON string)."""
    data = export_viz_data(viz_store)
    for path in data["concept_paths"]:
        assert "id" in path
        assert "name" in path
        assert "node_ids" in path
        assert isinstance(path["node_ids"], list)
        assert "type" in path


def test_export_patterns_have_detail(viz_store):
    """Patterns should have detail as a dict (not JSON string)."""
    data = export_viz_data(viz_store)
    for pat in data["patterns"]:
        assert "node_id" in pat
        assert "pattern" in pat
        assert "detail" in pat
        assert isinstance(pat["detail"], dict)
        assert "implication" in pat


def test_export_fragile_zones_detected(viz_store):
    """Fragile zones should identify high-importance nodes with reasons."""
    data = export_viz_data(viz_store)
    for zone in data["fragile_zones"]:
        assert "node_id" in zone
        assert "importance" in zone
        assert zone["importance"] > 5
        assert "reasons" in zone
        assert isinstance(zone["reasons"], list)
        assert len(zone["reasons"]) > 0


def test_export_json_serializable(viz_store):
    """The entire export dict should be JSON serializable."""
    data = export_viz_data(viz_store)
    result = json.dumps(data, ensure_ascii=False)
    assert len(result) > 100  # non-trivial output


def test_write_viz_json(viz_store, tmp_path):
    """write_viz_json should create a valid JSON file."""
    data = export_viz_data(viz_store)
    out = str(tmp_path / "test-viz.json")
    write_viz_json(data, out)
    assert os.path.exists(out)
    with open(out, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["meta"]["repo_name"] == "sandbox"
    assert len(loaded["nodes"]) > 0


def test_export_truncation(viz_store):
    """export_viz_data with small max_nodes should truncate."""
    full = export_viz_data(viz_store, max_nodes=0)
    total = len(full["nodes"])

    # Truncate to 5 nodes
    small = export_viz_data(viz_store, max_nodes=5)
    assert small["meta"]["truncated"] is True
    assert small["meta"]["total_node_count"] == total
    assert small["meta"]["viz_node_count"] <= 5 + total  # kept + some file parents
    # Non-file nodes should be at most 5
    non_file = [n for n in small["nodes"] if n["kind"] != "file"]
    assert len(non_file) <= 5
    # All edges should reference only kept nodes
    kept_ids = {n["id"] for n in small["nodes"]}
    for e in small["edges"]:
        assert e["source"] in kept_ids
        assert e["target"] in kept_ids
    # Patterns should reference only kept nodes
    for p in small["patterns"]:
        assert p["node_id"] in kept_ids


def test_export_no_truncation_when_under_limit(viz_store):
    """No truncation when node count is under max_nodes."""
    data = export_viz_data(viz_store, max_nodes=10000)
    assert data["meta"]["truncated"] is False
