"""Tree-sitter AST parser for Go source files (.go)."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_go as tsgo
from tree_sitter import Language, Parser

from nervx.perception.parser import (
    Node, RawCall, RawImport, ParseResult,
    _relative_path, _text, _line, _end_line,
)

GO_LANGUAGE = Language(tsgo.language())

# ── Public entry point ───────────────────────────────────────────────


def parse_go(file_path: str, repo_root: str) -> ParseResult:
    """Parse a Go source file and extract structural information."""
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

    # Read source
    try:
        source_bytes = Path(file_path).read_bytes()
    except (OSError, IOError):
        return _empty_result()

    # Parse with tree-sitter
    try:
        parser = Parser(GO_LANGUAGE)
        tree = parser.parse(source_bytes)
    except Exception:
        return _empty_result()

    root = tree.root_node
    if root is None:
        return _empty_result()

    # Total lines
    source_text = source_bytes.decode("utf8", errors="replace")
    total_lines = source_text.count("\n") + (
        1 if source_text and not source_text.endswith("\n") else 0
    )
    if total_lines == 0:
        total_lines = 1

    # Detect package name (needed for entrypoint tagging)
    package_name = _extract_package_name(root)

    # File tags
    file_tags: list[str] = []
    if package_name == "main":
        file_tags.append("entrypoint")

    # Build file node
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

    # Track struct/type nodes by name for method receiver resolution
    type_node_ids: dict[str, str] = {}

    # First pass: extract types, functions, imports
    for child in root.children:
        _process_top_level(child, rel_path, file_node.id, result,
                           package_name, type_node_ids)

    # Second pass: fix method parent_ids now that all types are known
    for node in result.nodes:
        if node.kind == "method" and node.parent_id.startswith("__receiver:"):
            receiver_type = node.parent_id[len("__receiver:"):]
            if receiver_type in type_node_ids:
                node.parent_id = type_node_ids[receiver_type]
            else:
                # Receiver type not found as a declared type in this file;
                # fall back to file-level parent
                node.parent_id = file_node.id

    # Build error_handling summary dict
    for rc in result.raw_calls:
        if rc.error_handling is not None:
            eh_dict = result.error_handling.setdefault(rc.caller_id, {})
            eh_dict[rc.callee_text] = rc.error_handling

    return result


# ── Package name extraction ──────────────────────────────────────────


def _extract_package_name(root) -> str:
    """Extract the package name from the package_clause."""
    for child in root.children:
        if child.type == "package_clause":
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                return _text(name_node)
            # Fallback: second child is typically the package name
            for sub in child.children:
                if sub.type == "package_identifier":
                    return _text(sub)
    return ""


# ── Top-level processing ─────────────────────────────────────────────


def _process_top_level(ts_node, rel_path: str, parent_id: str,
                       result: ParseResult, package_name: str,
                       type_node_ids: dict[str, str]):
    """Process a top-level tree-sitter node in a Go file."""
    node_type = ts_node.type

    if node_type == "type_declaration":
        _process_type_declaration(ts_node, rel_path, parent_id, result,
                                  type_node_ids)
    elif node_type == "function_declaration":
        _process_function(ts_node, rel_path, parent_id, result,
                          package_name)
    elif node_type == "method_declaration":
        _process_method(ts_node, rel_path, parent_id, result,
                        package_name, type_node_ids)
    elif node_type == "import_declaration":
        _process_import(ts_node, rel_path, result)


# ── Type declaration processing ──────────────────────────────────────


def _process_type_declaration(ts_node, rel_path: str, parent_id: str,
                              result: ParseResult,
                              type_node_ids: dict[str, str]):
    """Process a type_declaration which may contain one or more type_spec children."""
    for child in ts_node.children:
        if child.type == "type_spec":
            _process_type_spec(child, ts_node, rel_path, parent_id,
                               result, type_node_ids)
        elif child.type == "type_spec_list":
            # Grouped type declarations: type ( ... )
            for spec in child.children:
                if spec.type == "type_spec":
                    _process_type_spec(spec, ts_node, rel_path, parent_id,
                                       result, type_node_ids)


def _process_type_spec(type_spec, type_decl, rel_path: str,
                       parent_id: str, result: ParseResult,
                       type_node_ids: dict[str, str]):
    """Process a single type_spec node."""
    name_node = type_spec.child_by_field_name("name")
    if name_node is None:
        return
    type_name = _text(name_node)
    node_id = f"{rel_path}::{type_name}"
    type_node_ids[type_name] = node_id

    # Determine the underlying type node (struct, interface, or alias)
    type_body = type_spec.child_by_field_name("type")
    if type_body is None:
        return

    tags: list[str] = []
    signature = ""
    docstring = _get_go_doc_comment(type_decl)

    if type_body.type == "struct_type":
        signature = f"type {type_name} struct"
        # Extract embedded types (extends)
        embedded = _extract_embedded_fields(type_body)
        for emb in embedded:
            tags.append(f"extends:{emb}")
        # Unexported check
        if type_name and type_name[0].islower():
            tags.append("private")

    elif type_body.type == "interface_type":
        signature = f"type {type_name} interface"
        tags.append("interface")
        if type_name and type_name[0].islower():
            tags.append("private")

    else:
        # Type alias or type definition (e.g., type MyString string)
        signature = f"type {type_name} {_text(type_body)}"
        tags.append("type_alias")
        if type_name and type_name[0].islower():
            tags.append("private")

    type_node = Node(
        id=node_id,
        kind="class",
        name=type_name,
        file_path=rel_path,
        line_start=_line(type_decl),
        line_end=_end_line(type_decl),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(type_node)


def _extract_embedded_fields(struct_type_node) -> list[str]:
    """Extract embedded (anonymous) field type names from a struct body.

    Embedded fields are field_declaration nodes with no field name,
    only a type.
    """
    embedded: list[str] = []
    field_list = struct_type_node.child_by_field_name("fields") or \
        _find_child_by_type(struct_type_node, "field_declaration_list")
    if field_list is None:
        return embedded

    for child in field_list.children:
        if child.type == "field_declaration":
            # An embedded field has a type but no name list.
            # In tree-sitter-go, embedded fields use the "type" field
            # without a preceding "name" field.
            name_node = child.child_by_field_name("name")
            type_node = child.child_by_field_name("type")
            if name_node is None and type_node is not None:
                # The type could be a pointer, qualified ident, etc.
                type_text = _extract_type_name(type_node)
                if type_text:
                    embedded.append(type_text)
    return embedded


def _extract_type_name(type_node) -> str:
    """Extract a clean type name from a type node, stripping pointers/packages."""
    text = _text(type_node)
    # Strip pointer prefix
    text = text.lstrip("*")
    # If it's a qualified name like pkg.Type, take just the Type part
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


# ── Function processing ──────────────────────────────────────────────


def _process_function(ts_node, rel_path: str, parent_id: str,
                      result: ParseResult, package_name: str):
    """Process a top-level function_declaration."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    func_name = _text(name_node)
    node_id = f"{rel_path}::{func_name}"

    # Signature: funcName(params) returnType
    params_text = _build_params_text(ts_node.child_by_field_name("parameters"))
    result_text = _build_result_text(ts_node.child_by_field_name("result"))
    signature = f"{func_name}({params_text}){result_text}"

    # Tags
    tags = _compute_function_tags(func_name, package_name)

    # Docstring
    docstring = _get_go_doc_comment(ts_node)

    func_node = Node(
        id=node_id,
        kind="function",
        name=func_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(func_node)

    # Extract calls from the function body
    body = ts_node.child_by_field_name("body")
    if body is not None:
        _extract_calls_from_body(body, node_id, result)


# ── Method processing ────────────────────────────────────────────────


def _process_method(ts_node, rel_path: str, parent_id: str,
                    result: ParseResult, package_name: str,
                    type_node_ids: dict[str, str]):
    """Process a method_declaration (function with receiver)."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    method_name = _text(name_node)

    # Extract receiver type
    receiver_type = _extract_receiver_type(ts_node)

    # Node ID: path::ReceiverType.MethodName
    if receiver_type:
        node_id = f"{rel_path}::{receiver_type}.{method_name}"
        # Use a placeholder parent_id that will be resolved after all types are processed
        method_parent_id = f"__receiver:{receiver_type}"
    else:
        node_id = f"{rel_path}::{method_name}"
        method_parent_id = parent_id

    # Signature: (t *Type) methodName(params) returnType
    receiver_text = _build_receiver_text(ts_node.child_by_field_name("receiver"))
    params_text = _build_params_text(ts_node.child_by_field_name("parameters"))
    result_text = _build_result_text(ts_node.child_by_field_name("result"))
    signature = f"({receiver_text}) {method_name}({params_text}){result_text}"

    # Tags
    tags = _compute_function_tags(method_name, package_name)

    # Docstring
    docstring = _get_go_doc_comment(ts_node)

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
        parent_id=method_parent_id,
    )
    result.nodes.append(method_node)

    # Extract calls from the method body
    body = ts_node.child_by_field_name("body")
    if body is not None:
        _extract_calls_from_body(body, node_id, result)


def _extract_receiver_type(method_node) -> str:
    """Extract the receiver type name from a method_declaration.

    The receiver field contains a parameter_list with one parameter
    whose type is the receiver type (possibly a pointer).
    """
    receiver = method_node.child_by_field_name("receiver")
    if receiver is None:
        return ""

    # receiver is a parameter_list: (t *Type) or (*Type) or (Type)
    for child in receiver.children:
        if child.type == "parameter_declaration":
            type_node = child.child_by_field_name("type")
            if type_node is not None:
                return _extract_type_name(type_node)
    return ""


def _build_receiver_text(receiver_node) -> str:
    """Build the receiver text like 't *Type' from a receiver parameter_list."""
    if receiver_node is None:
        return ""
    # Strip outer parens from the receiver text
    text = _text(receiver_node).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text


def _build_params_text(params_node) -> str:
    """Build parameter list text from a parameter_list node, without outer parens."""
    if params_node is None:
        return ""
    text = _text(params_node).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text


def _build_result_text(result_node) -> str:
    """Build return type text from a result node."""
    if result_node is None:
        return ""
    text = _text(result_node).strip()
    return f" {text}"


# ── Import processing ────────────────────────────────────────────────


def _process_import(ts_node, rel_path: str, result: ParseResult):
    """Process an import_declaration (single or grouped)."""
    for child in ts_node.children:
        if child.type == "import_spec":
            _process_import_spec(child, rel_path, result)
        elif child.type == "import_spec_list":
            for spec in child.children:
                if spec.type == "import_spec":
                    _process_import_spec(spec, rel_path, result)
        elif child.type == "interpreted_string_literal":
            # Single import without spec list: import "fmt"
            module_path = _strip_quotes(_text(child))
            result.raw_imports.append(RawImport(
                importer_file=rel_path,
                module_path=module_path,
                imported_names=[],
                is_from_import=False,
            ))


def _process_import_spec(spec_node, rel_path: str, result: ParseResult):
    """Process a single import_spec node."""
    alias_node = spec_node.child_by_field_name("name")
    path_node = spec_node.child_by_field_name("path")

    if path_node is None:
        # Fallback: look for interpreted_string_literal child
        for child in spec_node.children:
            if child.type == "interpreted_string_literal":
                path_node = child
                break

    if path_node is None:
        return

    module_path = _strip_quotes(_text(path_node))
    imported_names: list[str] = []

    if alias_node is not None:
        alias_text = _text(alias_node)
        if alias_text == ".":
            # Dot import: import . "pkg" — all names available directly
            imported_names = ["*"]
        elif alias_text == "_":
            # Blank import: import _ "pkg" — side-effects only
            imported_names = []
        # else: named alias — no special handling needed

    result.raw_imports.append(RawImport(
        importer_file=rel_path,
        module_path=module_path,
        imported_names=imported_names,
        is_from_import=False,
    ))


# ── Tag computation ──────────────────────────────────────────────────


def _compute_function_tags(func_name: str, package_name: str) -> list[str]:
    """Compute semantic tags for a Go function or method."""
    tags: list[str] = []

    # entrypoint: main function in package main
    if func_name == "main" and package_name == "main":
        tags.append("entrypoint")

    # test: Go test conventions
    if (func_name.startswith("Test") or func_name.startswith("Bench")
            or func_name.startswith("Example")):
        tags.append("test")

    # callback: Go PascalCase conventions for handlers
    if func_name.startswith("On") or func_name.startswith("Handle"):
        tags.append("callback")

    # factory: Go convention for constructors
    if (func_name.startswith("New") or func_name.startswith("Create")
            or func_name.startswith("Make")):
        tags.append("factory")

    # private: unexported (starts with lowercase)
    if func_name and func_name[0].islower():
        tags.append("private")

    return tags


# ── Docstring extraction ─────────────────────────────────────────────


def _get_go_doc_comment(ts_node) -> str | None:
    """Extract Go doc comments (// comments immediately preceding a declaration).

    Go doc comments are consecutive // comment lines immediately before
    the declaration with no blank lines in between.
    """
    comments: list[str] = []
    sibling = ts_node.prev_named_sibling

    # Walk backwards through previous siblings collecting comment nodes
    while sibling is not None and sibling.type == "comment":
        comment_text = _text(sibling)
        # Strip the // prefix
        if comment_text.startswith("//"):
            comment_text = comment_text[2:]
            if comment_text.startswith(" "):
                comment_text = comment_text[1:]
        comments.append(comment_text)

        # Check that there's no blank line gap between this comment and the next
        next_sib = sibling.next_named_sibling
        if next_sib is not None:
            gap = next_sib.start_point.row - sibling.end_point.row
            if gap > 1:
                # There's a blank line — stop collecting
                break

        sibling = sibling.prev_named_sibling

    if not comments:
        return None

    # Comments were collected in reverse order
    comments.reverse()
    return "\n".join(comments).strip() or None


# ── Call extraction ──────────────────────────────────────────────────


def _extract_calls_from_body(body_node, caller_id: str,
                             result: ParseResult):
    """Walk a function/method body to find all call expressions."""
    for child in body_node.children:
        _walk_for_calls(child, caller_id, result, error_context=None)


def _walk_for_calls(ts_node, caller_id: str, result: ParseResult,
                    error_context: dict | None):
    """Recursively walk to find call_expression nodes, tracking error context."""
    node_type = ts_node.type

    # Don't descend into nested function literals
    if node_type == "func_literal":
        return

    # Handle if statements that check err != nil (Go error handling pattern)
    if node_type == "if_statement":
        _process_if_err_check(ts_node, caller_id, result)
        return

    # Handle call expressions
    if node_type == "call_expression":
        _record_call(ts_node, caller_id, result, error_context)
        # Still walk arguments for nested calls
        args = ts_node.child_by_field_name("arguments")
        if args is not None:
            for child in args.children:
                _walk_for_calls(child, caller_id, result, error_context)
        return

    # Recurse into children
    for child in ts_node.children:
        _walk_for_calls(child, caller_id, result, error_context)


def _process_if_err_check(if_node, caller_id: str, result: ParseResult):
    """Process an if statement, detecting `if err != nil` patterns.

    When we find `if err != nil { ... }`, calls inside the consequence
    block are marked with error handling context.
    """
    condition = if_node.child_by_field_name("condition")
    consequence = if_node.child_by_field_name("consequence")
    alternative = if_node.child_by_field_name("alternative")

    # Check if condition matches the err != nil pattern
    is_err_check = False
    if condition is not None:
        cond_text = _text(condition)
        if "err" in cond_text and "nil" in cond_text:
            is_err_check = True

    if is_err_check:
        err_context = {
            "pattern": "if_err_check",
            "exception": "error",
        }
        # Calls inside the err-handling block get error context
        if consequence is not None:
            for child in consequence.children:
                _walk_for_calls(child, caller_id, result,
                                error_context=err_context)
        # Walk the else branch without error context
        if alternative is not None:
            for child in alternative.children:
                _walk_for_calls(child, caller_id, result,
                                error_context=None)
    else:
        # Not an error check — walk all branches normally
        if consequence is not None:
            for child in consequence.children:
                _walk_for_calls(child, caller_id, result,
                                error_context=None)
        if alternative is not None:
            for child in alternative.children:
                _walk_for_calls(child, caller_id, result,
                                error_context=None)

    # Also check for calls in the condition itself (e.g., if err := foo(); err != nil)
    if condition is not None:
        _walk_for_calls(condition, caller_id, result, error_context=None)

    # Handle the initializer in `if val, err := foo(); err != nil`
    initializer = if_node.child_by_field_name("initializer")
    if initializer is not None:
        _walk_for_calls(initializer, caller_id, result, error_context=None)


def _record_call(call_node, caller_id: str, result: ParseResult,
                 error_context: dict | None):
    """Record a single call_expression as a RawCall."""
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return

    callee_text = _text(func_node)
    line = _line(call_node)

    # Determine return usage
    return_usage = _determine_return_usage(call_node)

    # Check if this call is part of a multi-value assignment with err
    err_handling = error_context
    if err_handling is None:
        err_handling = _detect_err_assignment(call_node)

    raw_call = RawCall(
        caller_id=caller_id,
        callee_text=callee_text,
        line=line,
        error_handling=err_handling,
        return_usage=return_usage,
    )
    result.raw_calls.append(raw_call)


def _detect_err_assignment(call_node) -> dict | None:
    """Detect if a call is in a `val, err := foo()` pattern.

    If the call's return value is assigned to a variable list that
    includes 'err', mark it as having error handling.
    """
    parent = call_node.parent
    if parent is None:
        return None

    # Check for short_var_declaration or assignment_statement
    if parent.type in ("short_var_declaration", "assignment_statement"):
        left = parent.child_by_field_name("left")
        if left is not None:
            left_text = _text(left)
            if "err" in left_text:
                return {
                    "pattern": "err_assignment",
                    "exception": "error",
                }
    return None


def _determine_return_usage(call_node) -> str:
    """Determine how the return value of a call is used."""
    parent = call_node.parent
    if parent is None:
        return "ignored"

    parent_type = parent.type

    # Assignment: x := foo() or x = foo()
    if parent_type in ("short_var_declaration", "assignment_statement"):
        return "assigned"

    # Return statement: return foo()
    if parent_type == "return_statement":
        return "returned"

    # Condition in if/for: if foo() { ... }
    if parent_type in ("if_statement", "for_statement"):
        return "checked"

    # Binary expression (comparisons, etc.)
    if parent_type == "binary_expression":
        return "checked"

    # Expression statement (bare call): foo()
    if parent_type == "expression_statement":
        return "ignored"

    # As argument to another call: bar(foo())
    if parent_type == "argument_list":
        return "assigned"

    # Selector expression: foo().Bar
    if parent_type == "selector_expression":
        return "assigned"

    # Index expression: foo()[0]
    if parent_type == "index_expression":
        return "assigned"

    # Type assertion: foo().(Type)
    if parent_type == "type_assertion_expression":
        return "assigned"

    # Defer/go statements: defer foo() or go foo()
    if parent_type in ("defer_statement", "go_statement"):
        return "ignored"

    return "ignored"


# ── Utility helpers ──────────────────────────────────────────────────


def _strip_quotes(s: str) -> str:
    """Strip surrounding quote characters from a string literal."""
    if len(s) >= 2:
        if (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
            return s[1:-1]
        if (s[0] == '`' and s[-1] == '`'):
            return s[1:-1]
    return s


def _find_child_by_type(node, type_name: str):
    """Find the first child with the given type."""
    for child in node.children:
        if child.type == type_name:
            return child
    return None
