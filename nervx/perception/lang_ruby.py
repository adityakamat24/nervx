"""Tree-sitter AST parser for Ruby files (.rb)."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_ruby as tsruby
from tree_sitter import Language, Parser

from nervx.perception.parser import (
    Node, RawCall, RawImport, ParseResult,
    _relative_path, _text, _line, _end_line,
)

# ── Language object ──────────────────────────────────────────────────

RUBY_LANGUAGE = Language(tsruby.language())

# ── Ruby data-model base classes ─────────────────────────────────────

_RUBY_DATA_MODEL_BASES = frozenset({
    "ActiveRecord::Base", "ApplicationRecord",
    "ActiveModel::Model", "ActiveModel::Serializer",
    "Sequel::Model", "Mongoid::Document",
})

# ── Rails callback prefixes ─────────────────────────────────────────

_CALLBACK_PREFIXES = ("on_", "handle_", "before_", "after_")

# ── Factory prefixes ────────────────────────────────────────────────

_FACTORY_PREFIXES = ("create_", "build_", "make_", "new")

# ── Entrypoint file basenames ────────────────────────────────────────

_ENTRYPOINT_FILES = frozenset({
    "config.ru", "rakefile", "rakefile.rb",
    "gemfile", "guardfile", "capfile",
})

# ── attr_* method names ─────────────────────────────────────────────

_ATTR_METHODS = frozenset({"attr_reader", "attr_writer", "attr_accessor"})

# ── Public entry point ──────────────────────────────────────────────


def parse_ruby(file_path: str, repo_root: str) -> ParseResult:
    """Parse a Ruby source file using tree-sitter and extract structural info."""
    rel_path = _relative_path(file_path, repo_root)
    file_name = Path(file_path).name

    def _empty_result(doc: str | None = None, line_end: int = 1) -> ParseResult:
        file_node = Node(
            id=rel_path,
            kind="file",
            name=file_name,
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
        parser = Parser(RUBY_LANGUAGE)
        tree = parser.parse(source_bytes)
    except Exception:
        return _empty_result()

    root = tree.root_node
    if root is None:
        return _empty_result()

    # Determine total line count
    source_text = source_bytes.decode("utf8", errors="replace")
    total_lines = source_text.count("\n") + (
        1 if source_text and not source_text.endswith("\n") else 0
    )
    if total_lines == 0:
        total_lines = 1

    # Check file-level properties
    lower_path = rel_path.lower().replace("\\", "/")
    is_test_file = (
        "test" in lower_path
        or "spec" in lower_path
        or "_test.rb" in lower_path
    )
    is_entrypoint_file = file_name.lower() in _ENTRYPOINT_FILES

    # File tags
    file_tags: list[str] = []
    if is_test_file:
        file_tags.append("test")
    if is_entrypoint_file:
        file_tags.append("entrypoint")

    # Check for file-level code (entrypoint heuristic: code outside class/module)
    has_top_level_code = _has_top_level_code(root)
    if has_top_level_code and "entrypoint" not in file_tags:
        file_tags.append("entrypoint")

    # Check for `if __FILE__ == $0` pattern
    if _has_file_main_guard(root) and "entrypoint" not in file_tags:
        file_tags.append("entrypoint")

    # Build the file node
    file_node = Node(
        id=rel_path,
        kind="file",
        name=file_name,
        file_path=rel_path,
        line_start=1,
        line_end=total_lines,
        signature="",
        docstring=None,
        tags=file_tags,
        parent_id="",
    )

    result = ParseResult(file_path=rel_path, nodes=[file_node])

    # Walk top-level children
    for child in root.children:
        _process_top_level(child, rel_path, file_node.id, result,
                           is_test_file=is_test_file,
                           class_stack=[],
                           private_seen=False)

    # Build the error_handling summary dict
    for rc in result.raw_calls:
        if rc.error_handling is not None:
            eh_dict = result.error_handling.setdefault(rc.caller_id, {})
            eh_dict[rc.callee_text] = rc.error_handling

    return result


# ── Top-level code detection ─────────────────────────────────────────

def _has_top_level_code(root) -> bool:
    """Check if the file has executable code outside class/module definitions."""
    _STRUCTURAL = frozenset({
        "class", "module", "comment", "program",
    })
    _IMPORT_CALLS = frozenset({
        "require", "require_relative", "include", "extend",
        "gem", "load", "autoload",
    })
    for child in root.children:
        if child.type in _STRUCTURAL:
            continue
        # require/require_relative are not really "code"
        if child.type == "call" and _get_call_method_name(child) in _IMPORT_CALLS:
            continue
        if child.type == "method":
            # Top-level method is still structural
            continue
        if child.type == "singleton_method":
            continue
        # Any other expression/statement is top-level code
        if child.type not in ("", "end"):
            return True
    return False


# ── __FILE__ == $0 detection ─────────────────────────────────────────

def _has_file_main_guard(root) -> bool:
    """Check if the file contains `if __FILE__ == $0` or `if $0 == __FILE__` pattern."""
    for child in root.children:
        if child.type == "if":
            cond_text = ""
            for sub in child.children:
                if sub.type == "binary":
                    cond_text = _text(sub)
                    break
            if "__FILE__" in cond_text and "$0" in cond_text:
                return True
    return False


# ── Top-level processing ─────────────────────────────────────────────

def _process_top_level(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_test_file: bool,
    class_stack: list[str],
    private_seen: bool,
):
    """Process a top-level (or body-level) tree-sitter node."""
    node_type = ts_node.type

    if node_type == "class":
        _process_class(ts_node, rel_path, parent_id, result,
                        is_test_file=is_test_file,
                        class_stack=class_stack)
    elif node_type == "module":
        _process_module(ts_node, rel_path, parent_id, result,
                         is_test_file=is_test_file,
                         class_stack=class_stack)
    elif node_type == "method":
        _process_method(ts_node, rel_path, parent_id, result,
                         is_test_file=is_test_file,
                         class_stack=class_stack,
                         is_private=private_seen)
    elif node_type == "singleton_method":
        _process_singleton_method(ts_node, rel_path, parent_id, result,
                                   is_test_file=is_test_file,
                                   class_stack=class_stack)
    elif node_type == "call":
        _process_call_statement(ts_node, rel_path, parent_id, result)


# ── Class processing ─────────────────────────────────────────────────

def _process_class(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_test_file: bool,
    class_stack: list[str],
):
    """Extract a class definition and its body."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    class_name = _text(name_node)

    # Handle scoped names like Outer::Inner
    # The name might be a scope_resolution node
    display_name = class_name

    # Build the class stack for nested IDs
    new_stack = class_stack + [display_name.replace("::", ".")]

    # Build node ID: path::Outer.Inner.ClassName
    if class_stack:
        qual_name = ".".join(class_stack) + "." + display_name.replace("::", ".")
    else:
        qual_name = display_name.replace("::", ".")
    node_id = f"{rel_path}::{qual_name}"

    # Superclass
    superclass_node = ts_node.child_by_field_name("superclass")
    superclass = _text(superclass_node) if superclass_node else ""

    # Signature
    if superclass:
        signature = f"class {display_name} < {superclass}"
    else:
        signature = f"class {display_name}"

    # Docstring: preceding comments
    docstring = _extract_preceding_comments(ts_node)

    # Tags
    tags = _compute_class_tags(display_name, superclass, is_test_file)

    class_node = Node(
        id=node_id,
        kind="class",
        name=display_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(class_node)

    # Process class body
    body = ts_node.child_by_field_name("body")
    if body is not None:
        _process_body(body, rel_path, node_id, result,
                       is_test_file=is_test_file,
                       class_stack=new_stack)


# ── Module processing ────────────────────────────────────────────────

def _process_module(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_test_file: bool,
    class_stack: list[str],
):
    """Extract a module definition and its body."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    module_name = _text(name_node)
    display_name = module_name

    new_stack = class_stack + [display_name.replace("::", ".")]

    if class_stack:
        qual_name = ".".join(class_stack) + "." + display_name.replace("::", ".")
    else:
        qual_name = display_name.replace("::", ".")
    node_id = f"{rel_path}::{qual_name}"

    signature = f"module {display_name}"

    docstring = _extract_preceding_comments(ts_node)

    tags: list[str] = ["module"]
    if is_test_file:
        tags.append("test")

    module_node = Node(
        id=node_id,
        kind="class",
        name=display_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(module_node)

    # Process module body
    body = ts_node.child_by_field_name("body")
    if body is not None:
        _process_body(body, rel_path, node_id, result,
                       is_test_file=is_test_file,
                       class_stack=new_stack)


# ── Body processing (shared by class/module) ─────────────────────────

def _process_body(
    body_node, rel_path: str, parent_id: str, result: ParseResult,
    is_test_file: bool,
    class_stack: list[str],
):
    """Process the body of a class or module, tracking `private` keyword."""
    private_seen = False

    for child in body_node.children:
        node_type = child.type

        # Track access control: `private`, `protected`, `public`
        if node_type == "identifier" and _text(child) == "private":
            private_seen = True
            continue
        if node_type == "identifier" and _text(child) in ("protected",):
            private_seen = True  # treat protected same as private for tagging
            continue
        if node_type == "identifier" and _text(child) == "public":
            private_seen = False
            continue

        # Also handle `private` as a call node (private :method_name)
        if node_type == "call" and _get_call_method_name(child) == "private":
            private_seen = True
            continue

        if node_type == "class":
            _process_class(child, rel_path, parent_id, result,
                            is_test_file=is_test_file,
                            class_stack=class_stack)
        elif node_type == "module":
            _process_module(child, rel_path, parent_id, result,
                             is_test_file=is_test_file,
                             class_stack=class_stack)
        elif node_type == "method":
            _process_method(child, rel_path, parent_id, result,
                             is_test_file=is_test_file,
                             class_stack=class_stack,
                             is_private=private_seen)
        elif node_type == "singleton_method":
            _process_singleton_method(child, rel_path, parent_id, result,
                                       is_test_file=is_test_file,
                                       class_stack=class_stack)
        elif node_type == "call":
            _process_call_statement(child, rel_path, parent_id, result)


# ── Method processing ────────────────────────────────────────────────

def _process_method(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_test_file: bool,
    class_stack: list[str],
    is_private: bool,
):
    """Extract a method definition (def foo)."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    method_name = _text(name_node)

    # Determine kind: "method" if inside a class/module, "function" if top-level
    is_method = len(class_stack) > 0
    kind = "method" if is_method else "function"

    # Build node ID
    if class_stack:
        qual_name = ".".join(class_stack) + "." + method_name
    else:
        qual_name = method_name
    node_id = f"{rel_path}::{qual_name}"

    # Parameters
    params_node = ts_node.child_by_field_name("parameters")
    params_text = _build_params_text(params_node)

    # Signature
    signature = f"def {method_name}({params_text})"

    # Docstring: preceding comments
    docstring = _extract_preceding_comments(ts_node)

    # Check if method body raises NotImplementedError (abstract detection)
    body = ts_node.child_by_field_name("body")
    is_abstract = _body_raises_not_implemented(body)

    # Tags
    tags = _compute_method_tags(method_name, is_test_file, is_private,
                                 is_singleton=False, is_abstract=is_abstract)

    method_node = Node(
        id=node_id,
        kind=kind,
        name=method_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(method_node)

    # Extract calls from the method body
    if body is not None:
        _extract_calls_from_body(body, node_id, result)


# ── Singleton method processing (def self.foo) ──────────────────────

def _process_singleton_method(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_test_file: bool,
    class_stack: list[str],
):
    """Extract a singleton method definition (def self.foo)."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    method_name = _text(name_node)

    # Build node ID
    if class_stack:
        qual_name = ".".join(class_stack) + "." + method_name
    else:
        qual_name = method_name
    node_id = f"{rel_path}::{qual_name}"

    # Parameters
    params_node = ts_node.child_by_field_name("parameters")
    params_text = _build_params_text(params_node)

    # Signature
    signature = f"def self.{method_name}({params_text})"

    # Docstring: preceding comments
    docstring = _extract_preceding_comments(ts_node)

    # Check if method body raises NotImplementedError (abstract detection)
    body = ts_node.child_by_field_name("body")
    is_abstract = _body_raises_not_implemented(body)

    # Tags
    tags = _compute_method_tags(method_name, is_test_file, is_private=False,
                                 is_singleton=True, is_abstract=is_abstract)

    method_node = Node(
        id=node_id,
        kind="method",
        name=method_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(method_node)

    # Extract calls from the method body
    if body is not None:
        _extract_calls_from_body(body, node_id, result)


# ── Call statement processing (require, include, attr_*, etc.) ───────

def _process_call_statement(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
):
    """Process a top-level or body-level call node for imports and attr_* declarations."""
    method_name = _get_call_method_name(ts_node)
    if not method_name:
        return

    # require / require_relative
    if method_name in ("require", "require_relative"):
        args_node = ts_node.child_by_field_name("arguments")
        if args_node is not None:
            arg_text = _get_first_string_arg(args_node)
            if arg_text is not None:
                result.raw_imports.append(RawImport(
                    importer_file=rel_path,
                    module_path=arg_text,
                    imported_names=[],
                    is_from_import=(method_name == "require_relative"),
                ))
        return

    # include / extend / prepend
    if method_name in ("include", "extend", "prepend"):
        args_node = ts_node.child_by_field_name("arguments")
        if args_node is not None:
            for child in args_node.children:
                if child.type in ("constant", "scope_resolution"):
                    mod_name = _text(child)
                    result.raw_imports.append(RawImport(
                        importer_file=rel_path,
                        module_path=mod_name,
                        imported_names=[],
                        is_from_import=True,
                    ))
        return

    # attr_reader / attr_writer / attr_accessor — no nodes emitted, but we
    # could track them. For now these are detected in tag computation.


# ── Parameter text building ──────────────────────────────────────────

def _build_params_text(params_node) -> str:
    """Build a parameter list string from a method_parameters node."""
    if params_node is None:
        return ""
    raw = _text(params_node).strip()
    # Remove surrounding parentheses if present
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
    return raw


# ── Docstring extraction (preceding comments) ───────────────────────

def _extract_preceding_comments(ts_node) -> str | None:
    """Extract consecutive comment lines immediately preceding a node."""
    comments: list[str] = []
    sibling = ts_node.prev_named_sibling
    while sibling is not None and sibling.type == "comment":
        text = _text(sibling)
        # Strip the leading # and optional space
        if text.startswith("#"):
            text = text[1:]
            if text.startswith(" "):
                text = text[1:]
        comments.append(text)
        sibling = sibling.prev_named_sibling

    if not comments:
        return None

    # Comments were collected in reverse order
    comments.reverse()
    return "\n".join(comments)


# ── Tag computation ──────────────────────────────────────────────────

def _compute_class_tags(
    class_name: str, superclass: str, is_test_file: bool,
) -> list[str]:
    """Compute semantic tags for a Ruby class."""
    tags: list[str] = []

    # extends tag
    if superclass:
        tags.append(f"extends:{superclass}")

    # data_model: inherits from ActiveRecord::Base, ApplicationRecord, etc.
    if superclass in _RUBY_DATA_MODEL_BASES:
        tags.append("data_model")

    # test: class name starts with Test or contains Spec
    if class_name.startswith("Test") or "Spec" in class_name:
        tags.append("test")
    elif is_test_file:
        # Don't auto-tag the class as test just because it's in a test file
        pass

    return tags


def _compute_method_tags(
    method_name: str, is_test_file: bool, is_private: bool,
    is_singleton: bool, is_abstract: bool = False,
) -> list[str]:
    """Compute semantic tags for a Ruby method."""
    tags: list[str] = []

    # static / classmethod for singleton methods
    if is_singleton:
        tags.append("static")
        tags.append("classmethod")

    # private: after `private` keyword OR method name starts with "_"
    if is_private or method_name.startswith("_"):
        tags.append("private")

    # test
    if method_name.startswith("test_"):
        tags.append("test")
    elif is_test_file and method_name.startswith("it_"):
        tags.append("test")

    # callback
    for prefix in _CALLBACK_PREFIXES:
        if method_name.startswith(prefix):
            tags.append("callback")
            break

    # factory: named "new", "create", "build", "initialize", or starts with factory prefix
    if method_name in ("new", "create", "build", "initialize"):
        tags.append("factory")
    else:
        for prefix in _FACTORY_PREFIXES:
            if method_name.startswith(prefix):
                tags.append("factory")
                break

    # initialize is like __init__ — always present, not private per se
    if method_name == "initialize":
        if "constructor" not in tags:
            tags.append("constructor")

    # abstract: methods that raise NotImplementedError
    if is_abstract:
        tags.append("abstract")

    return tags


# ── Abstract method detection ───────────────────────────────────────

def _body_raises_not_implemented(body_node) -> bool:
    """Check if a method body raises NotImplementedError.

    Looks for patterns like:
        raise NotImplementedError
        raise NotImplementedError, "message"
        raise NotImplementedError.new("message")
    """
    if body_node is None:
        return False
    return _walk_for_not_implemented(body_node)


def _walk_for_not_implemented(ts_node) -> bool:
    """Recursively check if any child is a raise of NotImplementedError."""
    # In Ruby, `raise NotImplementedError` parses as a call node
    # where the method is "raise" and the argument is NotImplementedError
    if ts_node.type == "call":
        method_name = _get_call_method_name(ts_node)
        if method_name in ("raise", "fail"):
            args = ts_node.child_by_field_name("arguments")
            if args is not None:
                args_text = _text(args)
                if "NotImplementedError" in args_text:
                    return True
    for child in ts_node.children:
        if _walk_for_not_implemented(child):
            return True
    return False


# ── Call extraction ──────────────────────────────────────────────────

def _extract_calls_from_body(body_node, caller_id: str, result: ParseResult):
    """Walk a method body to find all call expressions and record them."""
    for child in body_node.children:
        _walk_for_calls(child, caller_id, result, error_context=None)


def _walk_for_calls(
    ts_node, caller_id: str, result: ParseResult,
    error_context: dict | None,
):
    """Recursively walk to find call expressions, tracking error context."""
    node_type = ts_node.type

    # Don't descend into nested method/class/module definitions
    if node_type in ("method", "singleton_method", "class", "module"):
        return

    # Handle begin/rescue (error handling)
    if node_type == "begin":
        _process_begin_rescue(ts_node, caller_id, result)
        return

    # Handle exception_handler (inline rescue)
    if node_type == "exception_handler":
        _process_begin_rescue(ts_node, caller_id, result)
        return

    # Handle call nodes
    if node_type == "call":
        _record_call(ts_node, caller_id, result, error_context)
        # Walk arguments for nested calls
        args = ts_node.child_by_field_name("arguments")
        if args is not None:
            for child in args.children:
                _walk_for_calls(child, caller_id, result, error_context)
        # Walk block if present
        block = ts_node.child_by_field_name("block")
        if block is not None:
            _walk_for_calls(block, caller_id, result, error_context)
        return

    # Handle bare method calls (identifiers followed by argument_list, etc.)
    # In Ruby, simple method calls like `puts "hello"` are often just `call` nodes,
    # but some may be `identifier` nodes in certain contexts.

    # Recurse into children
    for child in ts_node.children:
        _walk_for_calls(child, caller_id, result, error_context)


# ── Begin/Rescue processing ─────────────────────────────────────────

def _process_begin_rescue(ts_node, caller_id: str, result: ParseResult):
    """Process a begin/rescue/ensure block, tracking exception types."""
    # Collect exception types from rescue clauses
    exception_types: list[str] = []
    rescue_nodes: list = []
    ensure_node = None
    body_children: list = []

    for child in ts_node.children:
        if child.type == "rescue":
            rescue_nodes.append(child)
            # Extract exception classes from rescue
            exc = _extract_rescue_exception(child)
            if exc:
                exception_types.extend(exc)
        elif child.type == "ensure":
            ensure_node = child
        elif child.type not in ("begin", "end", "rescue", "ensure",
                                 "else", "keyword"):
            body_children.append(child)

    # Build error context
    if exception_types:
        error_context: dict | None = {
            "pattern": "begin_rescue",
            "exception": ",".join(exception_types),
        }
    elif rescue_nodes:
        error_context = {
            "pattern": "begin_rescue",
            "exception": "StandardError",
        }
    else:
        error_context = None

    # Walk body children with error context
    for child in body_children:
        _walk_for_calls(child, caller_id, result, error_context)

    # Walk rescue blocks without error context
    for rescue_node in rescue_nodes:
        body = rescue_node.child_by_field_name("body")
        if body is not None:
            for child in body.children:
                _walk_for_calls(child, caller_id, result, error_context=None)
        else:
            # Walk rescue children directly (body may not be a named field)
            for child in rescue_node.children:
                if child.type not in ("rescue", "exceptions", "exception_variable",
                                       "then", "keyword"):
                    _walk_for_calls(child, caller_id, result, error_context=None)

    # Walk ensure block
    if ensure_node is not None:
        for child in ensure_node.children:
            if child.type not in ("ensure", "keyword"):
                _walk_for_calls(child, caller_id, result, error_context=None)


def _extract_rescue_exception(rescue_node) -> list[str]:
    """Extract exception class names from a rescue clause."""
    exceptions: list[str] = []
    exceptions_node = rescue_node.child_by_field_name("exceptions")
    if exceptions_node is not None:
        for child in exceptions_node.children:
            if child.type in ("constant", "scope_resolution"):
                exceptions.append(_text(child))
    else:
        # Walk children for exception types
        for child in rescue_node.children:
            if child.type == "exceptions":
                for sub in child.children:
                    if sub.type in ("constant", "scope_resolution"):
                        exceptions.append(_text(sub))
    return exceptions


# ── Call recording ───────────────────────────────────────────────────

def _record_call(call_node, caller_id: str, result: ParseResult,
                 error_context: dict | None):
    """Record a single call expression as a RawCall."""
    # Get the method name
    method_node = call_node.child_by_field_name("method")
    receiver_node = call_node.child_by_field_name("receiver")

    if method_node is None:
        return

    method_name = _text(method_node)
    receiver_text = _text(receiver_node) if receiver_node else ""

    # Build callee text
    if receiver_text:
        callee_text = f"{receiver_text}.{method_name}"
    else:
        callee_text = method_name

    line = _line(call_node)
    return_usage = _determine_return_usage(call_node)

    result.raw_calls.append(RawCall(
        caller_id=caller_id,
        callee_text=callee_text,
        line=line,
        error_handling=error_context,
        return_usage=return_usage,
    ))


def _determine_return_usage(call_node) -> str:
    """Determine how the return value of a call is used based on AST context."""
    parent = call_node.parent
    if parent is None:
        return "ignored"

    parent_type = parent.type

    # Assignment: x = foo()
    if parent_type == "assignment":
        return "assigned"

    # Operator assignment: x += foo()
    if parent_type == "operator_assignment":
        return "assigned"

    # Return: return foo()
    if parent_type == "return":
        return "returned"

    # Condition of if/unless/while/until
    if parent_type in ("if", "unless", "while", "until", "if_modifier",
                        "unless_modifier", "while_modifier", "until_modifier"):
        return "checked"

    # Boolean / comparison
    if parent_type in ("binary", "unary", "conditional"):
        return "checked"

    # As argument to another call (argument_list or bare argument)
    if parent_type in ("argument_list", "arguments"):
        return "assigned"

    # In array/hash literal
    if parent_type in ("array", "hash", "pair", "element_reference"):
        return "assigned"

    # Method chain: foo().bar
    if parent_type == "call":
        # If this call is the receiver of the parent call, value is used
        receiver = parent.child_by_field_name("receiver")
        if receiver is not None and receiver == call_node:
            return "assigned"
        return "ignored"

    # Bare expression (body of a method, top-level program)
    if parent_type in ("body_statement", "program", "then", "do",
                        "begin", "else", "block"):
        return "ignored"

    return "ignored"


# ── Helper: get method name from a call node ─────────────────────────

def _get_call_method_name(call_node) -> str:
    """Get the method name from a call node, or empty string."""
    method_node = call_node.child_by_field_name("method")
    if method_node is not None:
        return _text(method_node)
    # Fallback: for some call forms the first child might be the method
    # In Ruby tree-sitter, calls without a receiver may have the method name
    # as the first identifier child
    for child in call_node.children:
        if child.type == "identifier":
            return _text(child)
    return ""


def _get_first_string_arg(args_node) -> str | None:
    """Get the text content of the first string argument in an argument list."""
    for child in args_node.children:
        if child.type == "string":
            return _strip_ruby_string(child)
        if child.type == "string_content":
            return _text(child)
        # Handle simple_string / bare string
        if child.type in ("simple_string",):
            return _strip_quotes(_text(child))
    return None


def _strip_ruby_string(string_node) -> str:
    """Extract the text content from a Ruby string node."""
    # The string node typically has children: string_beginning, string_content, string_end
    for child in string_node.children:
        if child.type == "string_content":
            return _text(child)
    # Fallback: strip quotes from the raw text
    return _strip_quotes(_text(string_node))


def _strip_quotes(text: str) -> str:
    """Strip surrounding quotes from a string literal."""
    if len(text) >= 2:
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            return text[1:-1]
    return text
