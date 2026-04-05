"""Tests for tree-sitter AST parser."""

import os
import tempfile

import pytest

from nervx.perception.parser import (
    Node,
    ParseResult,
    extract_keywords,
    parse_file,
    parse_python,
    _split_identifier,
)


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temp repo root and return its path."""
    return str(tmp_path)


def _write_file(repo_root, rel_path, content):
    """Write a file in the repo and return its absolute path."""
    full = os.path.join(repo_root, rel_path.replace("/", os.sep))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return full


# ── Identifier splitting ──────────────────────────────────────────

def test_split_snake_case():
    assert _split_identifier("process_game_message") == ["process", "game", "message"]


def test_split_camel_case():
    assert _split_identifier("WebSocketManager") == ["web", "socket", "manager"]


def test_split_single_word():
    assert _split_identifier("connect") == ["connect"]


# ── Basic file parsing ────────────────────────────────────────────

def test_parse_empty_file(tmp_repo):
    path = _write_file(tmp_repo, "empty.py", "")
    result = parse_python(path, tmp_repo)
    assert result.file_path == "empty.py"
    assert len(result.nodes) == 1
    assert result.nodes[0].kind == "file"


def test_parse_file_with_module_docstring(tmp_repo):
    path = _write_file(tmp_repo, "mod.py", '"""This is a module."""\n\nx = 1\n')
    result = parse_python(path, tmp_repo)
    file_node = result.nodes[0]
    assert file_node.kind == "file"
    assert file_node.docstring == "This is a module."


def test_parse_function(tmp_repo):
    code = '''
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"
'''
    path = _write_file(tmp_repo, "hello.py", code)
    result = parse_python(path, tmp_repo)

    funcs = [n for n in result.nodes if n.kind == "function"]
    assert len(funcs) == 1
    f = funcs[0]
    assert f.name == "greet"
    assert f.id == "hello.py::greet"
    assert "name: str" in f.signature
    assert "str" in f.signature
    assert f.docstring == "Say hello."
    assert f.parent_id == "hello.py"


def test_parse_class_with_methods(tmp_repo):
    code = '''
class Animal:
    """An animal."""
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return "..."

    @staticmethod
    def kingdom():
        return "Animalia"
'''
    path = _write_file(tmp_repo, "zoo.py", code)
    result = parse_python(path, tmp_repo)

    classes = [n for n in result.nodes if n.kind == "class"]
    assert len(classes) == 1
    c = classes[0]
    assert c.name == "Animal"
    assert c.id == "zoo.py::Animal"
    assert c.docstring == "An animal."

    methods = [n for n in result.nodes if n.kind == "method"]
    assert len(methods) == 3
    names = {m.name for m in methods}
    assert names == {"__init__", "speak", "kingdom"}

    kingdom = next(m for m in methods if m.name == "kingdom")
    assert "static" in kingdom.tags


def test_parse_inheritance(tmp_repo):
    code = '''
class Base:
    pass

class Child(Base):
    pass
'''
    path = _write_file(tmp_repo, "inh.py", code)
    result = parse_python(path, tmp_repo)

    child = next(n for n in result.nodes if n.name == "Child")
    assert any(t.startswith("extends:") for t in child.tags)
    assert "extends:Base" in child.tags


def test_parse_async_function(tmp_repo):
    code = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    pass
'''
    path = _write_file(tmp_repo, "async_mod.py", code)
    result = parse_python(path, tmp_repo)

    funcs = [n for n in result.nodes if n.kind == "function"]
    assert len(funcs) == 1
    assert "async" in funcs[0].tags


# ── Tag detection ────────���────────────────────────────────────────

def test_tags_entrypoint(tmp_repo):
    code = 'def main():\n    pass\n'
    path = _write_file(tmp_repo, "app.py", code)
    result = parse_python(path, tmp_repo)
    func = next(n for n in result.nodes if n.kind == "function")
    assert "entrypoint" in func.tags


def test_tags_test(tmp_repo):
    code = 'def test_something():\n    pass\n'
    path = _write_file(tmp_repo, "t.py", code)
    result = parse_python(path, tmp_repo)
    func = next(n for n in result.nodes if n.kind == "function")
    assert "test" in func.tags


def test_tags_callback(tmp_repo):
    code = 'def on_message(msg):\n    pass\ndef handle_error(e):\n    pass\n'
    path = _write_file(tmp_repo, "cb.py", code)
    result = parse_python(path, tmp_repo)
    funcs = [n for n in result.nodes if n.kind == "function"]
    for f in funcs:
        assert "callback" in f.tags


def test_tags_factory(tmp_repo):
    code = 'def create_agent(t):\n    pass\n'
    path = _write_file(tmp_repo, "f.py", code)
    result = parse_python(path, tmp_repo)
    func = next(n for n in result.nodes if n.kind == "function")
    assert "factory" in func.tags


def test_tags_private_and_dunder(tmp_repo):
    code = '''
def _internal():
    pass

def __special__():
    pass
'''
    path = _write_file(tmp_repo, "priv.py", code)
    result = parse_python(path, tmp_repo)
    internal = next(n for n in result.nodes if n.name == "_internal")
    special = next(n for n in result.nodes if n.name == "__special__")
    assert "private" in internal.tags
    assert "dunder" in special.tags


def test_tags_abstract(tmp_repo):
    code = '''
from abc import abstractmethod

class Base:
    @abstractmethod
    def run(self):
        pass
'''
    path = _write_file(tmp_repo, "abs.py", code)
    result = parse_python(path, tmp_repo)
    run = next(n for n in result.nodes if n.name == "run")
    assert "abstract" in run.tags


def test_tags_data_model(tmp_repo):
    code = '''
from dataclasses import dataclass

@dataclass
class Config:
    host: str
    port: int
'''
    path = _write_file(tmp_repo, "cfg.py", code)
    result = parse_python(path, tmp_repo)
    cls = next(n for n in result.nodes if n.kind == "class")
    assert "data_model" in cls.tags


# ── Import extraction ─��───────────────────────────────────────────

def test_import_statement(tmp_repo):
    code = 'import os\nimport os.path\n'
    path = _write_file(tmp_repo, "imp.py", code)
    result = parse_python(path, tmp_repo)
    assert len(result.raw_imports) == 2
    assert result.raw_imports[0].module_path == "os"
    assert not result.raw_imports[0].is_from_import


def test_from_import(tmp_repo):
    code = 'from agents.factory import create_agent, AgentType\n'
    path = _write_file(tmp_repo, "imp2.py", code)
    result = parse_python(path, tmp_repo)
    assert len(result.raw_imports) == 1
    imp = result.raw_imports[0]
    assert imp.module_path == "agents.factory"
    assert imp.is_from_import
    assert "create_agent" in imp.imported_names
    assert "AgentType" in imp.imported_names


# ── Call extraction ───────���───────────────────────────────────────

def test_raw_calls(tmp_repo):
    code = '''
def run():
    x = foo()
    bar()
    self.baz.qux(1, 2)
'''
    path = _write_file(tmp_repo, "calls.py", code)
    result = parse_python(path, tmp_repo)
    calls = result.raw_calls
    callee_texts = [c.callee_text for c in calls]
    assert "foo" in callee_texts
    assert "bar" in callee_texts
    assert any("baz" in t or "qux" in t for t in callee_texts)


def test_return_usage_assigned(tmp_repo):
    code = '''
def run():
    x = foo()
'''
    path = _write_file(tmp_repo, "ru.py", code)
    result = parse_python(path, tmp_repo)
    call = next(c for c in result.raw_calls if c.callee_text == "foo")
    assert call.return_usage == "assigned"


def test_return_usage_ignored(tmp_repo):
    code = '''
def run():
    foo()
'''
    path = _write_file(tmp_repo, "ru2.py", code)
    result = parse_python(path, tmp_repo)
    call = next(c for c in result.raw_calls if c.callee_text == "foo")
    assert call.return_usage == "ignored"


def test_return_usage_returned(tmp_repo):
    code = '''
def run():
    return foo()
'''
    path = _write_file(tmp_repo, "ru3.py", code)
    result = parse_python(path, tmp_repo)
    call = next(c for c in result.raw_calls if c.callee_text == "foo")
    assert call.return_usage == "returned"


# ── Error handling detection ──────────────────────────────────────

def test_error_handling_try_except(tmp_repo):
    code = '''
def run():
    try:
        foo()
    except ValueError:
        pass
'''
    path = _write_file(tmp_repo, "eh.py", code)
    result = parse_python(path, tmp_repo)
    call = next(c for c in result.raw_calls if c.callee_text == "foo")
    assert call.error_handling is not None
    assert call.error_handling["pattern"] == "try_except"
    assert "ValueError" in call.error_handling["exception"]


def test_no_error_handling_outside_try(tmp_repo):
    code = '''
def run():
    foo()
'''
    path = _write_file(tmp_repo, "eh2.py", code)
    result = parse_python(path, tmp_repo)
    call = next(c for c in result.raw_calls if c.callee_text == "foo")
    assert call.error_handling is None


# ── Keyword extraction ──────��─────────────────────────────────────

def test_extract_keywords_from_name():
    node = Node(
        id="agents/factory.py::create_agent",
        kind="function",
        name="create_agent",
        file_path="agents/factory.py",
        line_start=1, line_end=5,
        signature="create_agent(t)",
        docstring="Create an agent instance.",
        tags=["factory"],
    )
    kws = extract_keywords(node)
    keywords = [k for k, _ in kws]
    assert "create_agent" in keywords
    assert "create" in keywords
    assert "agent" in keywords
    assert "factory" in keywords
    assert "agents" in keywords
    assert "instance" in keywords


def test_extract_keywords_camel_case():
    node = Node(
        id="ws.py::WebSocketManager",
        kind="class",
        name="WebSocketManager",
        file_path="ws.py",
        line_start=1, line_end=5,
        signature="class WebSocketManager",
        docstring=None,
        tags=[],
    )
    kws = extract_keywords(node)
    keywords = [k for k, _ in kws]
    assert "web" in keywords
    assert "socket" in keywords
    assert "manager" in keywords


# ── Nonexistent file ──────────────────────────────────────────────

def test_parse_nonexistent_file(tmp_repo):
    result = parse_python(os.path.join(tmp_repo, "nope.py"), tmp_repo)
    assert len(result.nodes) == 1
    assert result.nodes[0].kind == "file"


# ── parse_file dispatch ──────────────────────────────────────────

def test_parse_file_unsupported_extension(tmp_repo):
    path = _write_file(tmp_repo, "data.json", '{"key": "value"}')
    result = parse_file(path, tmp_repo)
    assert len(result.nodes) == 1
    assert result.nodes[0].kind == "file"


def test_parse_file_dispatches_python(tmp_repo):
    code = 'def hello():\n    pass\n'
    path = _write_file(tmp_repo, "hello.py", code)
    result = parse_file(path, tmp_repo)
    assert any(n.kind == "function" for n in result.nodes)
