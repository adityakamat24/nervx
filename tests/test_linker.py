"""Tests for the linker (edge resolution)."""

import pytest

from nervx.perception.linker import (
    Edge,
    build_symbol_index,
    resolve_all,
    resolve_calls,
    resolve_imports,
    resolve_inheritance,
)
from nervx.perception.parser import Node, ParseResult, RawCall, RawImport


def _make_node(id, kind="function", name=None, file_path=None, **kwargs):
    if name is None:
        name = id.split("::")[-1] if "::" in id else id
    if file_path is None:
        file_path = id.split("::")[0] if "::" in id else id
    return Node(
        id=id, kind=kind, name=name, file_path=file_path,
        line_start=1, line_end=10, signature=f"{name}()",
        docstring=None, tags=kwargs.get("tags", []),
        parent_id=kwargs.get("parent_id", ""),
    )


def _make_pr(file_path, nodes, raw_calls=None, raw_imports=None):
    return ParseResult(
        file_path=file_path,
        nodes=nodes,
        raw_calls=raw_calls or [],
        raw_imports=raw_imports or [],
    )


# ── Symbol index ──────────────────────────────────────────────────

def test_build_symbol_index():
    n1 = _make_node("a.py::foo")
    n2 = _make_node("b.py::bar")
    n3 = _make_node("c.py::foo")  # duplicate name
    file_node = _make_node("a.py", kind="file")

    prs = [
        _make_pr("a.py", [file_node, n1]),
        _make_pr("b.py", [n2]),
        _make_pr("c.py", [n3]),
    ]
    idx = build_symbol_index(prs)
    assert len(idx["foo"]) == 2
    assert len(idx["bar"]) == 1
    assert "a.py" not in idx  # file nodes excluded


# ── Import resolution ─────────────────────────────────────────────

def test_resolve_imports_basic():
    n1 = _make_node("a.py", kind="file")
    n2 = _make_node("b.py", kind="file")
    n3 = _make_node("b.py::Bar", kind="class", name="Bar")

    imp = RawImport(
        importer_file="a.py",
        module_path="b",
        imported_names=["Bar"],
        is_from_import=True,
    )
    prs = [
        _make_pr("a.py", [n1], raw_imports=[imp]),
        _make_pr("b.py", [n2, n3]),
    ]
    idx = build_symbol_index(prs)
    edges = resolve_imports(prs, idx)

    # Should have edges: a.py->b.py (file import) and a.py->b.py::Bar (symbol import)
    targets = {e.target_id for e in edges}
    assert "b.py" in targets
    assert "b.py::Bar" in targets


def test_resolve_imports_no_match():
    """Import of external module should produce no edges."""
    n1 = _make_node("a.py", kind="file")
    imp = RawImport(importer_file="a.py", module_path="os.path", imported_names=["join"])
    prs = [_make_pr("a.py", [n1], raw_imports=[imp])]
    idx = build_symbol_index(prs)
    edges = resolve_imports(prs, idx)
    assert len(edges) == 0


# ── Call resolution ───────────────────────────────────────────────

def test_resolve_calls_simple():
    """A calls B, both in same file."""
    n1 = _make_node("a.py::foo")
    n2 = _make_node("a.py::bar")
    call = RawCall(caller_id="a.py::foo", callee_text="bar", line=5)
    prs = [_make_pr("a.py", [n1, n2], raw_calls=[call])]
    idx = build_symbol_index(prs)
    edges = resolve_calls(prs, idx)
    assert len(edges) == 1
    assert edges[0].source_id == "a.py::foo"
    assert edges[0].target_id == "a.py::bar"
    assert edges[0].edge_type == "calls"


def test_resolve_calls_self_dot():
    """self.method() should strip self. and resolve."""
    cls = _make_node("a.py::Foo", kind="class", name="Foo")
    m1 = _make_node("a.py::Foo.run", kind="method", name="run")
    m2 = _make_node("a.py::Foo.helper", kind="method", name="helper")
    call = RawCall(caller_id="a.py::Foo.run", callee_text="self.helper", line=5)
    prs = [_make_pr("a.py", [cls, m1, m2], raw_calls=[call])]
    idx = build_symbol_index(prs)
    edges = resolve_calls(prs, idx)
    assert len(edges) == 1
    assert edges[0].target_id == "a.py::Foo.helper"


def test_resolve_calls_cross_file():
    """Cross-file call via import."""
    n1 = _make_node("a.py::foo")
    n2 = _make_node("b.py::bar")
    call = RawCall(caller_id="a.py::foo", callee_text="bar", line=5)
    imp = RawImport(importer_file="a.py", module_path="b", imported_names=["bar"], is_from_import=True)
    prs = [
        _make_pr("a.py", [n1], raw_calls=[call], raw_imports=[imp]),
        _make_pr("b.py", [n2]),
    ]
    idx = build_symbol_index(prs)
    edges = resolve_calls(prs, idx)
    assert len(edges) == 1
    assert edges[0].target_id == "b.py::bar"


def test_resolve_calls_unresolved():
    """Call to stdlib function produces no edge."""
    n1 = _make_node("a.py::foo")
    call = RawCall(caller_id="a.py::foo", callee_text="print", line=5)
    prs = [_make_pr("a.py", [n1], raw_calls=[call])]
    idx = build_symbol_index(prs)
    edges = resolve_calls(prs, idx)
    assert len(edges) == 0


def test_resolve_calls_disambiguate_same_file():
    """When two files have 'process', prefer same-file."""
    n1 = _make_node("a.py::run")
    n2 = _make_node("a.py::process")
    n3 = _make_node("b.py::process")
    call = RawCall(caller_id="a.py::run", callee_text="process", line=5)
    prs = [
        _make_pr("a.py", [n1, n2], raw_calls=[call]),
        _make_pr("b.py", [n3]),
    ]
    idx = build_symbol_index(prs)
    edges = resolve_calls(prs, idx)
    assert len(edges) == 1
    assert edges[0].target_id == "a.py::process"


def test_resolve_calls_with_error_handling():
    """Error handling metadata flows through to edges."""
    n1 = _make_node("a.py::foo")
    n2 = _make_node("a.py::bar")
    call = RawCall(
        caller_id="a.py::foo", callee_text="bar", line=5,
        error_handling={"pattern": "try_except", "exception": "ValueError"},
    )
    prs = [_make_pr("a.py", [n1, n2], raw_calls=[call])]
    idx = build_symbol_index(prs)
    edges = resolve_calls(prs, idx)
    assert len(edges) == 1
    assert edges[0].metadata.get("error_handling", {}).get("exception") == "ValueError"


# ── Inheritance resolution ────────────────────────────────────────

def test_resolve_inheritance():
    base = _make_node("a.py::Base", kind="class", name="Base")
    child = _make_node("b.py::Child", kind="class", name="Child", tags=["extends:Base"])
    prs = [
        _make_pr("a.py", [base]),
        _make_pr("b.py", [child]),
    ]
    idx = build_symbol_index(prs)
    edges = resolve_inheritance(prs, idx)
    assert len(edges) == 1
    assert edges[0].source_id == "b.py::Child"
    assert edges[0].target_id == "a.py::Base"
    assert edges[0].edge_type == "inherits"


# ── Full resolution ───────────────────────────────────────────────

def test_resolve_all():
    base = _make_node("a.py::Base", kind="class", name="Base")
    child = _make_node("b.py::Child", kind="class", name="Child", tags=["extends:Base"])
    func = _make_node("b.py::run")
    call = RawCall(caller_id="b.py::run", callee_text="Base", line=5)
    imp = RawImport(importer_file="b.py", module_path="a", imported_names=["Base"], is_from_import=True)

    prs = [
        _make_pr("a.py", [base]),
        _make_pr("b.py", [child, func], raw_calls=[call], raw_imports=[imp]),
    ]
    edges = resolve_all(prs)
    types = {e.edge_type for e in edges}
    assert "imports" in types
    assert "inherits" in types
    assert "calls" in types
