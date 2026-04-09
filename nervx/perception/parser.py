"""Tree-sitter AST parser for extracting structural information from source files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())

# ── Dataclasses ─────────────────────────────────────────────────────

@dataclass
class Node:
    id: str           # Fully qualified: "path/file.py::ClassName.method_name"
    kind: str         # "file", "class", "function", "method"
    name: str         # Short name: "connect", "GameState", "app.py"
    file_path: str    # Relative path: "agents/base_agent.py"
    line_start: int
    line_end: int
    signature: str    # "connect(self, ws_manager) -> None", "class GameState", or "" for files
    docstring: str | None
    tags: list[str] = field(default_factory=list)
    parent_id: str = ""


@dataclass
class RawCall:
    caller_id: str    # Node ID of the calling function
    callee_text: str  # Raw text like "self.memory.store", "asyncio.run", "GameState()"
    line: int
    error_handling: dict | None = None  # {"pattern": "try_except", "exception": "ValueError"}
    return_usage: str = "ignored"       # "assigned", "checked", "ignored", "awaited", "awaited_ignored", "returned"
    # 0.2.6: inferred class name of the call receiver (e.g. ``SamplingParams``
    # for ``sp.verify()`` when ``sp`` was previously assigned
    # ``sp = SamplingParams(...)``). Populated by the Python parser only today;
    # used by the linker to disambiguate method calls that collide across
    # multiple classes. Empty string when unknown — the linker then falls back
    # to its existing fan-out strategy.
    receiver_type: str = ""


@dataclass
class RawImport:
    importer_file: str  # File path of the importing file
    module_path: str    # "memory.pipeline" or "os.path"
    imported_names: list[str] = field(default_factory=list)
    is_from_import: bool = False


@dataclass
class ParseResult:
    file_path: str
    nodes: list[Node] = field(default_factory=list)
    raw_calls: list[RawCall] = field(default_factory=list)
    raw_imports: list[RawImport] = field(default_factory=list)
    error_handling: dict = field(default_factory=dict)  # {caller_id: {callee_name: {pattern, exception}}}


# ── Stop words for keyword extraction ───────────────────────────────

STOP_WORDS = frozenset({
    "self", "cls", "none", "true", "false", "return", "def", "class",
    "import", "from", "if", "else", "elif", "for", "while", "try",
    "except", "with", "as", "in", "is", "not", "and", "or", "the",
    "a", "an", "to", "of", "str", "int", "float", "bool", "list",
    "dict", "set", "tuple", "type", "get", "set", "init", "new",
})

# ── Decorator keywords for route handlers ──────────────────────────

ROUTE_DECORATOR_KEYWORDS = frozenset({
    "route", "get", "post", "put", "delete", "patch", "api_view",
})

# ── Validator decorators (Pydantic, attrs, etc.) ─────────────────

_VALIDATOR_DECORATORS = frozenset({
    "validator", "field_validator", "root_validator", "model_validator",
    "validate", "validates", "field_serializer", "model_serializer",
    "computed_field",
})

# ── Hook/lifecycle decorators ────────────────────────────────────

_HOOK_DECORATORS = frozenset({
    "on_event", "listener", "receiver", "hookimpl", "hookspec",
    "before_request", "after_request", "middleware",
    "startup", "shutdown", "lifespan",
    "event_handler", "signal",
})

# ── Data model base classes ────────────────────────────────────────

DATA_MODEL_BASES = frozenset({
    "BaseModel", "Model", "Schema", "TypedDict", "NamedTuple",
})

# ── Regex helpers ──────────────────────────────────────────────────

_CAMEL_RE1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_RE2 = re.compile(r"([a-z0-9])([A-Z])")


def _split_identifier(name: str) -> list[str]:
    """Split snake_case and camelCase/PascalCase into lowercase words."""
    # Handle snake_case
    if "_" in name:
        parts = name.split("_")
    else:
        # Handle camelCase / PascalCase
        s = _CAMEL_RE1.sub(r"\1_\2", name)
        s = _CAMEL_RE2.sub(r"\1_\2", s)
        parts = s.split("_")
    return [p.lower() for p in parts if p]


# ── Node text helper ───────────────────────────────────────────────

def _text(node) -> str:
    """Get decoded text from a tree-sitter node, or empty string if None."""
    if node is None:
        return ""
    return node.text.decode("utf8")


def _line(node) -> int:
    """Get 1-indexed line number from a tree-sitter node."""
    return node.start_point.row + 1


def _end_line(node) -> int:
    """Get 1-indexed end line number from a tree-sitter node."""
    return node.end_point.row + 1


# ── Relative path helper ──────────────────────────────────────────

def _relative_path(file_path: str, repo_root: str) -> str:
    """Compute relative path using forward slashes."""
    try:
        rel = Path(file_path).resolve().relative_to(Path(repo_root).resolve())
        return str(PurePosixPath(rel))
    except ValueError:
        # If can't compute relative, use the file_path as-is with forward slashes
        return file_path.replace("\\", "/")


# ── Main entry points ─────────────────────────────────────────────

def parse_file(file_path: str, repo_root: str) -> ParseResult:
    """Dispatch to language-specific parser based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".py":
        return parse_python(file_path, repo_root)
    if ext in (".js", ".jsx"):
        from nervx.perception.lang_javascript import parse_javascript
        return parse_javascript(file_path, repo_root)
    if ext == ".ts":
        from nervx.perception.lang_javascript import parse_typescript
        return parse_typescript(file_path, repo_root)
    if ext == ".tsx":
        from nervx.perception.lang_javascript import parse_tsx
        return parse_tsx(file_path, repo_root)
    if ext == ".java":
        from nervx.perception.lang_java import parse_java
        return parse_java(file_path, repo_root)
    if ext == ".go":
        from nervx.perception.lang_go import parse_go
        return parse_go(file_path, repo_root)
    if ext == ".rs":
        from nervx.perception.lang_rust import parse_rust
        return parse_rust(file_path, repo_root)
    if ext in (".c", ".h"):
        from nervx.perception.lang_c import parse_c
        return parse_c(file_path, repo_root)
    if ext in (".cpp", ".hpp", ".cc", ".cxx", ".hh"):
        from nervx.perception.lang_c import parse_cpp
        return parse_cpp(file_path, repo_root)
    if ext == ".cs":
        from nervx.perception.lang_csharp import parse_csharp
        return parse_csharp(file_path, repo_root)
    if ext == ".rb":
        from nervx.perception.lang_ruby import parse_ruby
        return parse_ruby(file_path, repo_root)
    # For unsupported extensions, return minimal result with file node
    rel_path = _relative_path(file_path, repo_root)
    file_node = Node(
        id=rel_path,
        kind="file",
        name=Path(file_path).name,
        file_path=rel_path,
        line_start=1,
        line_end=1,
        signature="",
        docstring=None,
        tags=[],
        parent_id="",
    )
    return ParseResult(file_path=rel_path, nodes=[file_node])


def parse_python(file_path: str, repo_root: str) -> ParseResult:
    """Parse a Python source file using tree-sitter and extract structural info."""
    rel_path = _relative_path(file_path, repo_root)

    # Empty fallback result with just a file node
    def _empty_result(doc: str | None = None, line_end: int = 1) -> ParseResult:
        file_node = Node(
            id=rel_path,
            kind="file",
            name=Path(file_path).name,
            file_path=rel_path,
            line_start=1,
            line_end=line_end,
            signature="",
            docstring=doc,
            tags=[],
            parent_id="",
        )
        return ParseResult(file_path=rel_path, nodes=[file_node])

    # Read the file
    try:
        source_bytes = Path(file_path).read_bytes()
    except (OSError, IOError):
        return _empty_result()

    # Parse with tree-sitter
    try:
        parser = Parser(PY_LANGUAGE)
        tree = parser.parse(source_bytes)
    except Exception:
        return _empty_result()

    root = tree.root_node
    if root is None:
        return _empty_result()

    # Determine total line count
    source_text = source_bytes.decode("utf8", errors="replace")
    total_lines = source_text.count("\n") + (1 if source_text and not source_text.endswith("\n") else 0)
    if total_lines == 0:
        total_lines = 1

    # Extract module docstring
    module_doc = _extract_module_docstring(root)

    # Build the file node
    file_node = Node(
        id=rel_path,
        kind="file",
        name=Path(file_path).name,
        file_path=rel_path,
        line_start=1,
        line_end=total_lines,
        signature="",
        docstring=module_doc,
        tags=[],
        parent_id="",
    )

    result = ParseResult(file_path=rel_path, nodes=[file_node])

    # Detect __all__ exports
    exported_names = _extract_dunder_all(root)

    # Walk top-level children
    for child in root.children:
        _process_top_level(child, rel_path, file_node.id, result)

    # Tag nodes that appear in __all__ as "exported"
    if exported_names:
        for node in result.nodes:
            if node.name in exported_names:
                node.tags.append("exported")

    # Build the error_handling summary dict
    for rc in result.raw_calls:
        if rc.error_handling is not None:
            eh_dict = result.error_handling.setdefault(rc.caller_id, {})
            eh_dict[rc.callee_text] = rc.error_handling

    return result


# ── Module docstring extraction ─────────────────────────────────────

def _extract_module_docstring(root) -> str | None:
    """Extract the module-level docstring (first expression if it's a string)."""
    for child in root.children:
        if child.type == "comment":
            continue
        if child.type == "expression_statement":
            expr = child.children[0] if child.children else None
            if expr is not None and expr.type == "string":
                return _parse_string_content(expr)
        # First non-comment, non-docstring statement means no module docstring
        return None
    return None


def _extract_dunder_all(root) -> set[str]:
    """Extract names from __all__ = [...] at module level."""
    names: set[str] = set()
    for child in root.children:
        if child.type != "expression_statement":
            continue
        expr = child.children[0] if child.children else None
        if expr is None or expr.type != "assignment":
            continue
        # Check left side is __all__
        left = expr.child_by_field_name("left")
        if left is None or _text(left) != "__all__":
            continue
        # Extract string elements from the list on the right
        right = expr.child_by_field_name("right")
        if right is None or right.type != "list":
            continue
        for elem in right.children:
            if elem.type == "string":
                val = _parse_string_content(elem)
                if val:
                    names.add(val)
    return names


# ── Top-level processing ───────────────────────────────────────────

def _process_top_level(ts_node, rel_path: str, file_id: str, result: ParseResult):
    """Process a top-level tree-sitter node."""
    node_type = ts_node.type

    if node_type == "class_definition":
        _process_class(ts_node, rel_path, file_id, result, decorators=[])
    elif node_type == "function_definition":
        _process_function(ts_node, rel_path, file_id, result,
                          is_method=False, class_name=None, decorators=[])
    elif node_type == "decorated_definition":
        _process_decorated(ts_node, rel_path, file_id, result,
                           is_method=False, class_name=None)
    elif node_type in ("import_statement", "import_from_statement"):
        _process_import(ts_node, rel_path, result)


# ── Decorated definition unwrapping ────────────────────────────────

def _process_decorated(ts_node, rel_path: str, parent_id: str, result: ParseResult,
                       is_method: bool, class_name: str | None,
                       class_self_types: dict[str, str] | None = None):
    """Unwrap a decorated_definition to get decorators and the inner def/class."""
    decorators = []
    inner = None

    for child in ts_node.children:
        if child.type == "decorator":
            decorators.append(child)
        elif child.type == "class_definition":
            inner = child
        elif child.type == "function_definition":
            inner = child

    if inner is None:
        return

    if inner.type == "class_definition":
        _process_class(inner, rel_path, parent_id, result, decorators=decorators)
    elif inner.type == "function_definition":
        _process_function(inner, rel_path, parent_id, result,
                          is_method=is_method, class_name=class_name,
                          decorators=decorators,
                          class_self_types=class_self_types)


# ── Class processing ───────────────────────────────────────────────

def _process_class(ts_node, rel_path: str, parent_id: str, result: ParseResult,
                   decorators: list):
    """Extract a class definition and its methods."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    class_name = _text(name_node)

    # Build node ID
    node_id = f"{rel_path}::{class_name}"

    # Base classes
    base_classes = _extract_base_classes(ts_node)

    # Signature: "class ClassName(Base1, Base2)" or "class ClassName"
    if base_classes:
        signature = f"class {class_name}({', '.join(base_classes)})"
    else:
        signature = f"class {class_name}"

    # Decorators text
    decorator_texts = [_get_decorator_text(d) for d in decorators]

    # Tags
    tags = _compute_class_tags(class_name, base_classes, decorator_texts)

    # Docstring
    body = ts_node.child_by_field_name("body")
    docstring = _extract_body_docstring(body)

    # Use the decorated_definition's line range if decorators exist
    if decorators:
        line_start = _line(decorators[0])
    else:
        line_start = _line(ts_node)
    line_end = _end_line(ts_node)

    class_node = Node(
        id=node_id,
        kind="class",
        name=class_name,
        file_path=rel_path,
        line_start=line_start,
        line_end=line_end,
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(class_node)

    # 0.2.6: aggregate ``self.foo = Class(...)`` assignments across every
    # method in the class so each method's call extractor can resolve
    # ``self.store.get()`` to the type assigned in ``__init__`` (or any
    # other method that sets ``self.store``).
    class_self_types = _collect_class_self_types(body) if body is not None else {}

    # Process class body for methods
    if body is not None:
        for child in body.children:
            if child.type == "function_definition":
                _process_function(child, rel_path, node_id, result,
                                  is_method=True, class_name=class_name,
                                  decorators=[],
                                  class_self_types=class_self_types)
            elif child.type == "decorated_definition":
                _process_decorated(child, rel_path, node_id, result,
                                   is_method=True, class_name=class_name,
                                   class_self_types=class_self_types)
            elif child.type == "class_definition":
                # Nested class
                _process_class(child, rel_path, node_id, result, decorators=[])


# ── Function/method processing ─────────────────────────────────────

def _process_function(ts_node, rel_path: str, parent_id: str, result: ParseResult,
                      is_method: bool, class_name: str | None, decorators: list,
                      class_self_types: dict[str, str] | None = None):
    """Extract a function or method definition."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    func_name = _text(name_node)

    # Determine kind
    kind = "method" if is_method else "function"

    # Build node ID
    if class_name:
        node_id = f"{rel_path}::{class_name}.{func_name}"
    else:
        node_id = f"{rel_path}::{func_name}"

    # Check if async
    is_async = any(c.type == "async" for c in ts_node.children)

    # Signature
    signature = _build_function_signature(ts_node, func_name)

    # Decorators text
    decorator_texts = [_get_decorator_text(d) for d in decorators]

    # Tags
    tags = _compute_function_tags(func_name, is_async, decorator_texts)

    # Docstring
    body = ts_node.child_by_field_name("body")
    docstring = _extract_body_docstring(body)

    # Line range (include decorators)
    if decorators:
        line_start = _line(decorators[0])
    else:
        line_start = _line(ts_node)
    line_end = _end_line(ts_node)

    func_node = Node(
        id=node_id,
        kind=kind,
        name=func_name,
        file_path=rel_path,
        line_start=line_start,
        line_end=line_end,
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(func_node)

    # 0.2.6: build receiver-type map for this function scope. Merge the
    # class-level ``self.foo`` map (if this is a method) as a baseline so
    # ``self.store.get()`` inside any method can see the type assigned to
    # ``self.store`` in ``__init__``.
    params_node = ts_node.child_by_field_name("parameters")
    method_locals = _scan_function_local_types(body, params_node)
    if class_self_types:
        merged_locals: dict[str, str] = {**class_self_types, **method_locals}
    else:
        merged_locals = method_locals

    # Extract calls from the function body
    if body is not None:
        _extract_calls_from_body(body, node_id, result,
                                 local_types=merged_locals)


# ── Signature building ─────────────────────────────────────────────

def _build_function_signature(ts_node, func_name: str) -> str:
    """Build a function signature string like 'func(a, b: int) -> str'."""
    params_node = ts_node.child_by_field_name("parameters")
    params_text = _text(params_node) if params_node else "()"

    # Strip outer parens, rebuild cleanly
    inner = params_text.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()

    # Return type
    return_type = ts_node.child_by_field_name("return_type")
    ret_text = ""
    if return_type is not None:
        ret_text = f" -> {_text(return_type)}"

    return f"{func_name}({inner}){ret_text}"


# ── Base class extraction ──────────────────────────────────────────

def _extract_base_classes(class_node) -> list[str]:
    """Extract base class names from a class definition's argument_list or superclasses."""
    bases = []
    # The superclasses are in an argument_list child (between class name and colon)
    for child in class_node.children:
        if child.type == "argument_list":
            for arg_child in child.children:
                if arg_child.type in ("identifier", "attribute", "keyword_argument"):
                    if arg_child.type == "keyword_argument":
                        # e.g., metaclass=ABCMeta — skip for base class list
                        continue
                    bases.append(_text(arg_child))
                elif arg_child.type == "subscript":
                    # e.g., Generic[T] — use full text
                    bases.append(_text(arg_child))
            break
    return bases


# ── Decorator helpers ──────────────────────────────────────────────

def _get_decorator_text(decorator_node) -> str:
    """Get the decorator text without the @ symbol."""
    full = _text(decorator_node)
    if full.startswith("@"):
        return full[1:].strip()
    return full.strip()


def _decorator_tag_name(dec: str) -> str:
    """Strip arguments and whitespace from a decorator, e.g. `app.route("/x")` -> `app.route`."""
    # Cut at first `(` or whitespace
    name = dec.split("(", 1)[0]
    name = name.split()[0] if name.split() else name
    return name.strip()


# ── Tag computation ────────────────────────────────────────────────

def _compute_class_tags(class_name: str, base_classes: list[str], decorators: list[str]) -> list[str]:
    """Compute semantic tags for a class."""
    tags = []

    # Raw decorator names for framework-awareness
    for dec in decorators:
        name = _decorator_tag_name(dec)
        if name:
            tags.append(f"decorator:{name}")

    # extends tag
    if base_classes:
        tags.append(f"extends:{','.join(base_classes)}")

    # data_model
    is_data_model = False
    for base in base_classes:
        # Strip generics: "Generic[T]" -> "Generic"
        base_simple = base.split("[")[0].split(".")[-1]
        if base_simple in DATA_MODEL_BASES:
            is_data_model = True
            break
    for dec in decorators:
        if "dataclass" in dec:
            is_data_model = True
            break
    if is_data_model:
        tags.append("data_model")

    # test
    if class_name.startswith("Test"):
        tags.append("test")

    # private / dunder
    if class_name.startswith("__") and class_name.endswith("__"):
        tags.append("dunder")
    elif class_name.startswith("_"):
        tags.append("private")

    # abstract - class with ABCMeta or ABC base, or @abstractmethod usage
    for base in base_classes:
        base_simple = base.split("[")[0].split(".")[-1]
        if base_simple in ("ABC", "ABCMeta"):
            tags.append("abstract")
            break

    return tags


def _compute_function_tags(func_name: str, is_async: bool, decorators: list[str]) -> list[str]:
    """Compute semantic tags for a function or method."""
    tags = []

    # Raw decorator names for framework-awareness
    for dec in decorators:
        name = _decorator_tag_name(dec)
        if name:
            tags.append(f"decorator:{name}")

    # async
    if is_async:
        tags.append("async")

    # entrypoint
    if func_name in ("main", "__main__"):
        tags.append("entrypoint")

    # test
    if func_name.startswith("test_") or func_name.startswith("Test"):
        tags.append("test")

    # callback (on_, _on_, handle_, _handle_ patterns)
    if func_name.lstrip("_").startswith("on_") or func_name.lstrip("_").startswith("handle_"):
        tags.append("callback")

    # factory
    for prefix in ("create_", "build_", "make_", "get_or_create_"):
        if func_name.startswith(prefix):
            tags.append("factory")
            break

    # serializer
    for prefix in ("to_", "from_", "serialize", "deserialize", "parse_", "format_"):
        if func_name.startswith(prefix):
            tags.append("serializer")
            break

    # private / dunder
    if func_name.startswith("__") and func_name.endswith("__"):
        tags.append("dunder")
    elif func_name.startswith("_"):
        tags.append("private")

    # Decorator-based tags
    for dec in decorators:
        dec_lower = dec.lower()
        # route_handler
        for kw in ROUTE_DECORATOR_KEYWORDS:
            if kw in dec_lower:
                tags.append("route_handler")
                break

        # property / cached_property
        if dec == "property" or dec.endswith(".setter") or dec.endswith(".deleter"):
            if "property" not in tags:
                tags.append("property")
        if dec in ("cached_property", "functools.cached_property"):
            if "property" not in tags:
                tags.append("property")

        # static
        if dec == "staticmethod":
            tags.append("static")

        # classmethod
        if dec == "classmethod":
            tags.append("classmethod")

        # abstract
        if dec == "abstractmethod" or dec.endswith(".abstractmethod"):
            tags.append("abstract")

        # validator (Pydantic, attrs, etc.)
        for vkw in _VALIDATOR_DECORATORS:
            if vkw in dec_lower:
                if "validator" not in tags:
                    tags.append("validator")
                break

        # override
        if dec in ("override", "typing.override", "typing_extensions.override"):
            tags.append("override")

        # overload
        if dec in ("overload", "typing.overload", "typing_extensions.overload"):
            tags.append("overload")

        # event/signal/hook (framework lifecycle)
        for hkw in _HOOK_DECORATORS:
            if hkw in dec_lower:
                if "hook" not in tags:
                    tags.append("hook")
                break

    return tags


# ── Docstring extraction ──────────────────────────────────────────

def _extract_body_docstring(body_node) -> str | None:
    """Extract docstring from the first statement of a body/block."""
    if body_node is None:
        return None
    for child in body_node.children:
        if child.type == "comment":
            continue
        if child.type == "expression_statement":
            expr = child.children[0] if child.children else None
            if expr is not None and expr.type == "string":
                return _parse_string_content(expr)
        return None
    return None


def _parse_string_content(string_node) -> str:
    """Parse a tree-sitter string node to get its content without quotes."""
    raw = _text(string_node)
    # Handle triple-quoted strings
    for prefix in ('"""', "'''", 'r"""', "r'''", 'b"""', "b'''",
                   'f"""', "f'''", 'u"""', "u'''"):
        quote = prefix[-3:]
        if raw.startswith(prefix) and raw.endswith(quote):
            content = raw[len(prefix):-3]
            return content.strip()
    # Handle single-quoted strings
    for prefix in ('"', "'", 'r"', "r'", 'b"', "b'", 'f"', "f'", 'u"', "u'"):
        quote = prefix[-1]
        if raw.startswith(prefix) and raw.endswith(quote) and len(raw) > len(prefix):
            content = raw[len(prefix):-1]
            return content.strip()
    return raw


# ── Import processing ──────────────────────────────────────────────

def _process_import(ts_node, rel_path: str, result: ParseResult):
    """Process import_statement or import_from_statement."""
    if ts_node.type == "import_statement":
        # import os / import os.path / import a, b
        for child in ts_node.children:
            if child.type == "dotted_name":
                module_path = _text(child)
                result.raw_imports.append(RawImport(
                    importer_file=rel_path,
                    module_path=module_path,
                    imported_names=[],
                    is_from_import=False,
                ))
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                if name_node:
                    module_path = _text(name_node)
                    result.raw_imports.append(RawImport(
                        importer_file=rel_path,
                        module_path=module_path,
                        imported_names=[],
                        is_from_import=False,
                    ))
    elif ts_node.type == "import_from_statement":
        # from X import a, b
        module_path = ""
        imported_names = []

        # Find the module path (dotted_name after 'from')
        # And imported names (dotted_name or aliased_import after 'import')
        past_from = False
        past_import = False

        for child in ts_node.children:
            if child.type == "from":
                past_from = True
                continue
            if child.type == "import":
                past_import = True
                continue
            if child.type == "relative_import":
                # from . import something or from .foo import bar
                module_path = _text(child)
                continue

            if past_from and not past_import:
                if child.type == "dotted_name":
                    module_path = _text(child)
            elif past_import:
                if child.type == "dotted_name":
                    imported_names.append(_text(child))
                elif child.type == "aliased_import":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        imported_names.append(_text(name_node))
                elif child.type == "wildcard_import":
                    imported_names.append("*")

        result.raw_imports.append(RawImport(
            importer_file=rel_path,
            module_path=module_path,
            imported_names=imported_names,
            is_from_import=True,
        ))


# ── Local-type inference (0.2.6, Gap 1 Layer A) ────────────────────
#
# Python-specific receiver-type scanner. For each function/method we build a
# `local_types: dict[str, str]` map that answers the question
# "if I see `x.foo()`, what class is `x` most likely an instance of?".
# Later, the linker uses this to pick the right `Class.foo` candidate from
# symbol_index instead of guessing.
#
# Sources of type information (best-effort — never crashes on unknown forms):
#   1. Explicit type annotations:      ``x: SamplingParams``
#   2. Annotated assignments:           ``x: SamplingParams = f()``
#   3. Constructor-call inference:      ``x = ClassName(...)``  → "ClassName"
#   4. Parameter type hints:            ``def f(self, sp: SamplingParams):``
#   5. Per-class ``self.foo = Class(...)`` aggregated across all methods
#      of the enclosing class, so ``self.store.get()`` in method A sees
#      the type assigned to ``self.store`` in ``__init__``.

def _simplify_type_annotation(type_text: str) -> str:
    """Reduce a type annotation to the primary class name if possible.

    ``SamplingParams``                 → ``SamplingParams``
    ``Optional[SamplingParams]``       → ``SamplingParams``
    ``SamplingParams | None``          → ``SamplingParams``
    ``list[Foo]``, ``dict[str, Foo]``  → ``""`` (collection container wins; skip)
    """
    t = (type_text or "").strip()
    if not t:
        return ""
    if t.startswith('"') or t.startswith("'"):
        t = t.strip("\"'")
    if t.startswith("Optional[") and t.endswith("]"):
        t = t[len("Optional["):-1].strip()
    if "|" in t:
        parts = [p.strip() for p in t.split("|")]
        non_none = [p for p in parts if p and p != "None"]
        if len(non_none) == 1:
            t = non_none[0]
        else:
            return ""
    # Bare identifier? (letters/digits/underscore, not starting with digit)
    if t and not t[0].isdigit() and all(c.isalnum() or c == "_" for c in t):
        return t
    return ""


def _record_type_from_assignment(assign_node, var_name: str,
                                 out: dict[str, str]) -> None:
    """Update ``out`` with a type for ``var_name`` from an assignment node.

    Prefers explicit type annotations; falls back to constructor inference
    (RHS is ``ClassName(...)`` and ``ClassName`` starts with uppercase).
    """
    type_node = assign_node.child_by_field_name("type")
    right = assign_node.child_by_field_name("right")

    if type_node is not None:
        simple = _simplify_type_annotation(_text(type_node))
        if simple:
            out[var_name] = simple
            return

    if right is not None and right.type == "call":
        func_node = right.child_by_field_name("function")
        if func_node is not None and func_node.type == "identifier":
            class_name = _text(func_node)
            if class_name and class_name[0].isupper():
                out[var_name] = class_name


def _extract_param_type(param_node, local_types: dict[str, str]) -> None:
    """Pull ``name: Type`` from a function parameters list entry."""
    if param_node.type not in ("typed_parameter", "typed_default_parameter"):
        return
    name = None
    type_text = None
    for child in param_node.children:
        if name is None and child.type == "identifier":
            name = _text(child)
        elif child.type == "type":
            type_text = _text(child)
    if name and type_text:
        simple = _simplify_type_annotation(type_text)
        if simple:
            local_types[name] = simple


def _scan_method_body_for_types(node, local_types: dict[str, str]) -> None:
    """Walk a method/function body collecting local-variable types.

    Doesn't descend into nested function or class definitions — those have
    their own scope and will be scanned when they get processed.
    """
    nt = node.type
    if nt in ("function_definition", "class_definition", "decorated_definition"):
        return

    if nt == "assignment":
        left = node.child_by_field_name("left")
        if left is not None:
            if left.type == "identifier":
                _record_type_from_assignment(node, _text(left), local_types)
            elif left.type == "attribute":
                name = _text(left)
                if name.startswith("self."):
                    _record_type_from_assignment(node, name, local_types)

    for child in node.children:
        _scan_method_body_for_types(child, local_types)


def _scan_function_local_types(body_node, params_node) -> dict[str, str]:
    """Build the receiver-type map for a single function/method scope."""
    local_types: dict[str, str] = {}

    if params_node is not None:
        for child in params_node.children:
            _extract_param_type(child, local_types)

    if body_node is not None:
        _scan_method_body_for_types(body_node, local_types)

    return local_types


def _collect_class_self_types(class_body) -> dict[str, str]:
    """Collect ``self.foo = ClassName(...)`` across every method of a class.

    Captures ``self.X`` assignments from ``__init__`` *and* from any other
    method — users sometimes lazy-initialize attributes outside ``__init__``.
    Stops at nested class definitions, whose ``self`` means something else.
    """
    self_types: dict[str, str] = {}
    if class_body is None:
        return self_types

    def walk(node):
        if node.type == "class_definition":
            return
        if node.type == "assignment":
            left = node.child_by_field_name("left")
            if left is not None and left.type == "attribute":
                name = _text(left)
                if name.startswith("self."):
                    _record_type_from_assignment(node, name, self_types)
        for child in node.children:
            walk(child)

    walk(class_body)
    return self_types


# ── Call extraction ────────────────────────────────────────────────

def _extract_calls_from_body(body_node, caller_id: str, result: ParseResult,
                             local_types: dict[str, str] | None = None):
    """Walk a function body to find all call expressions and record them."""
    # We need to track which try/except context each call is in
    for child in body_node.children:
        _walk_for_calls(child, caller_id, result, error_context=None,
                        local_types=local_types)


def _walk_for_calls(ts_node, caller_id: str, result: ParseResult,
                    error_context: dict | None,
                    local_types: dict[str, str] | None = None):
    """Recursively walk to find call expressions, tracking error context."""
    node_type = ts_node.type

    # Don't descend into nested function or class definitions
    if node_type in ("function_definition", "class_definition", "decorated_definition"):
        return

    # Handle try_statement: update error context for calls inside
    if node_type == "try_statement":
        _process_try_statement(ts_node, caller_id, result, local_types)
        return

    # Handle call expressions
    if node_type == "call":
        _record_call(ts_node, caller_id, result, error_context, local_types)
        # Still need to walk arguments for nested calls
        args = ts_node.child_by_field_name("arguments")
        if args is not None:
            for child in args.children:
                _walk_for_calls(child, caller_id, result, error_context,
                                local_types)
        return

    # Recurse into children
    for child in ts_node.children:
        _walk_for_calls(child, caller_id, result, error_context, local_types)


def _process_try_statement(ts_node, caller_id: str, result: ParseResult,
                           local_types: dict[str, str] | None = None):
    """Process a try/except statement, tracking exception types for calls in the try block."""
    # Collect exception types from except clauses
    exception_types = []
    for child in ts_node.children:
        if child.type == "except_clause":
            exc_type = _extract_except_type(child)
            if exc_type:
                exception_types.append(exc_type)

    error_context = None
    if exception_types:
        error_context = {
            "pattern": "try_except",
            "exception": ",".join(exception_types),
        }
    else:
        error_context = {
            "pattern": "try_except",
            "exception": "BaseException",
        }

    # Walk the try block with error context
    for child in ts_node.children:
        if child.type == "block":
            # First block child of try_statement is the try body
            for block_child in child.children:
                _walk_for_calls(block_child, caller_id, result, error_context,
                                local_types)
            break

    # Walk except/else/finally blocks without the error context (or with their own)
    for child in ts_node.children:
        if child.type in ("except_clause", "else_clause", "finally_clause"):
            for sub in child.children:
                if sub.type == "block":
                    _walk_for_calls(sub, caller_id, result, error_context=None,
                                    local_types=local_types)


def _extract_except_type(except_node) -> str | None:
    """Extract the exception type from an except clause."""
    for child in except_node.children:
        if child.type == "identifier":
            return _text(child)
        if child.type == "as_pattern":
            # except ValueError as e
            for sub in child.children:
                if sub.type == "identifier":
                    return _text(sub)
        if child.type == "tuple":
            # except (ValueError, TypeError)
            names = []
            for sub in child.children:
                if sub.type == "identifier":
                    names.append(_text(sub))
            return ",".join(names) if names else None
    return None


def _record_call(call_node, caller_id: str, result: ParseResult,
                 error_context: dict | None,
                 local_types: dict[str, str] | None = None):
    """Record a single call expression as a RawCall."""
    # Get callee text (the function being called)
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        # Fallback: first child
        if call_node.children:
            func_node = call_node.children[0]
        else:
            return

    callee_text = _text(func_node)
    line = _line(call_node)

    # Determine return_usage
    return_usage = _determine_return_usage(call_node)

    # 0.2.6: infer receiver type for ``x.method()`` calls if we have a
    # local-type map. Tries the longest receiver prefix first so
    # ``self.store.get`` is resolved before falling back to ``self.store``.
    receiver_type = ""
    if local_types and "." in callee_text:
        receiver = callee_text.rsplit(".", 1)[0]
        while receiver:
            if receiver in local_types:
                receiver_type = local_types[receiver]
                break
            if "." not in receiver:
                break
            receiver = receiver.rsplit(".", 1)[0]

    raw_call = RawCall(
        caller_id=caller_id,
        callee_text=callee_text,
        line=line,
        error_handling=error_context,
        return_usage=return_usage,
        receiver_type=receiver_type,
    )
    result.raw_calls.append(raw_call)


def _determine_return_usage(call_node) -> str:
    """Determine how the return value of a call is used based on its AST context."""
    parent = call_node.parent
    if parent is None:
        return "ignored"

    parent_type = parent.type

    # await wrapping: check the await's parent for the actual usage
    if parent_type == "await":
        grandparent = parent.parent
        if grandparent is None:
            return "awaited_ignored"

        gp_type = grandparent.type

        if gp_type == "assignment" or gp_type == "augmented_assignment":
            return "awaited"
        if gp_type == "return_statement":
            return "awaited"
        if gp_type == "expression_statement":
            return "awaited_ignored"
        if gp_type in ("conditional_expression", "boolean_operator", "comparison_operator",
                        "not_operator"):
            return "awaited"
        # Default for awaited
        return "awaited"

    # Direct assignment: x = foo()
    if parent_type == "assignment" or parent_type == "augmented_assignment":
        return "assigned"

    # Return statement: return foo()
    if parent_type == "return_statement":
        return "returned"

    # Condition of if/while/elif: if foo():
    if parent_type in ("if_statement", "while_statement", "elif_clause"):
        return "checked"

    # Boolean expressions: foo() and bar(), not foo()
    if parent_type in ("boolean_operator", "not_operator", "comparison_operator",
                        "conditional_expression"):
        return "checked"

    # assert foo()
    if parent_type == "assert_statement":
        return "checked"

    # Bare expression statement: foo()
    if parent_type == "expression_statement":
        return "ignored"

    # As argument to another call: bar(foo()) - the inner call's value is used
    if parent_type == "argument_list":
        return "assigned"

    # Yield: yield foo()
    if parent_type == "yield":
        return "returned"

    # Part of a list/tuple/dict comprehension or literal
    if parent_type in ("list", "tuple", "dictionary", "set",
                        "list_comprehension", "set_comprehension",
                        "dictionary_comprehension", "generator_expression"):
        return "assigned"

    # Subscript: foo()[0]
    if parent_type == "subscript":
        return "assigned"

    # Attribute access: foo().bar
    if parent_type == "attribute":
        return "assigned"

    # Default
    return "ignored"


# ── Keyword extraction ─────────────────────────────────────────────

def extract_keywords(node: Node) -> list[tuple[str, str]]:
    """Extract (keyword, source) pairs from a Node for indexing.

    Sources: "name", "docstring", "tag", "file_path".
    """
    keywords: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(word: str, source: str):
        w = word.lower().strip()
        if len(w) < 3:
            return
        if w in STOP_WORDS:
            return
        pair = (w, source)
        if pair not in seen:
            seen.add(pair)
            keywords.append(pair)

    # 1. Name decomposition
    # Full name
    _add(node.name, "name")
    # Split into parts
    for part in _split_identifier(node.name):
        _add(part, "name")

    # 2. Docstring words
    if node.docstring:
        for word in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", node.docstring):
            _add(word, "docstring")

    # 3. Tags
    for tag in node.tags:
        # For "extends:Foo,Bar" split into the tag prefix and the values
        if tag.startswith("extends:"):
            _add("extends", "tag")
            for base in tag[len("extends:"):].split(","):
                _add(base.strip(), "tag")
        else:
            _add(tag, "tag")

    # 4. File path segments
    path_parts = node.file_path.replace("\\", "/").split("/")
    for part in path_parts:
        # Remove extension from filename
        stem = part.rsplit(".", 1)[0] if "." in part else part
        _add(stem, "file_path")
        # Also split the stem
        for sub in _split_identifier(stem):
            _add(sub, "file_path")

    return keywords
