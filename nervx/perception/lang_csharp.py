"""Tree-sitter AST parser for C# source files (.cs)."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_c_sharp as tscs
from tree_sitter import Language, Parser

from nervx.perception.parser import (
    Node, RawCall, RawImport, ParseResult,
    _relative_path, _text, _line, _end_line,
)

# ── Language object ──────────────────────────────────────────────────

CS_LANGUAGE = Language(tscs.language())

# ── Type declaration node types ──────────────────────────────────────

_TYPE_DECL_TYPES = frozenset({
    "class_declaration",
    "interface_declaration",
    "struct_declaration",
    "enum_declaration",
    "record_declaration",
})

# ── Test attribute names (NUnit, MSTest, xUnit) ─────────────────────

_TEST_ATTRIBUTES = frozenset({
    "Test", "TestMethod", "Fact", "Theory",
    "TestCase", "TestCaseSource", "DataTestMethod",
})

# ── Data model attribute names ───────────────────────────────────────

_DATA_MODEL_ATTRIBUTES = frozenset({"Serializable", "DataContract", "Table", "Entity"})

# ── Factory method prefixes ──────────────────────────────────────────

_FACTORY_PREFIXES = ("Create", "Build", "Make")

# ── Public entry point ───────────────────────────────────────────────


def parse_csharp(file_path: str, repo_root: str) -> ParseResult:
    """Parse a C# source file and extract structural information."""
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
        parser = Parser(CS_LANGUAGE)
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

    # Detect namespace (for node ID construction)
    namespace = _extract_namespace(root)

    # File tags
    file_tags: list[str] = []

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

    # Walk top-level children
    for child in root.children:
        _process_top_level(child, rel_path, file_node.id, result, namespace)

    # Build error_handling summary dict
    for rc in result.raw_calls:
        if rc.error_handling is not None:
            eh_dict = result.error_handling.setdefault(rc.caller_id, {})
            eh_dict[rc.callee_text] = rc.error_handling

    # Tag file as entrypoint if any method named Main exists
    for node in result.nodes:
        if node.kind == "method" and node.name == "Main":
            if "entrypoint" not in file_tags:
                file_tags.append("entrypoint")
            break

    return result


# ── Namespace extraction ────────────────────────────────────────────


def _extract_namespace(root) -> str:
    """Extract the namespace name from the compilation unit.

    Handles both block-scoped ``namespace Foo.Bar { }`` and file-scoped
    ``namespace Foo.Bar;`` declarations.
    """
    for child in root.children:
        if child.type in ("namespace_declaration",
                          "file_scoped_namespace_declaration"):
            for sub in child.children:
                if sub.type in ("qualified_name", "identifier"):
                    return _text(sub)
    return ""


# ── Top-level processing ─────────────────────────────────────────────


def _process_top_level(ts_node, rel_path: str, parent_id: str,
                       result: ParseResult, namespace: str):
    """Process a top-level tree-sitter node in a C# file."""
    node_type = ts_node.type

    if node_type == "using_directive":
        _process_using(ts_node, rel_path, result)
    elif node_type in ("namespace_declaration",
                        "file_scoped_namespace_declaration"):
        _process_namespace(ts_node, rel_path, parent_id, result, namespace)
    elif node_type in _TYPE_DECL_TYPES:
        _process_type_declaration(ts_node, rel_path, parent_id, result,
                                  namespace)
    elif node_type == "global_statement":
        # C# top-level statements (C# 9+) — scan for calls
        for child in ts_node.children:
            _walk_for_calls(child, parent_id, result, error_context=None)


# ── Namespace processing ─────────────────────────────────────────────


def _process_namespace(ts_node, rel_path: str, parent_id: str,
                       result: ParseResult, namespace: str):
    """Process a namespace_declaration -- recurse into its body for type declarations."""
    # Determine namespace name from this node
    ns_name = namespace
    for sub in ts_node.children:
        if sub.type in ("qualified_name", "identifier"):
            ns_name = _text(sub)
            break

    # Process children inside the namespace body (declaration_list)
    for child in ts_node.children:
        if child.type == "declaration_list":
            for decl in child.children:
                _process_namespace_member(decl, rel_path, parent_id,
                                          result, ns_name)
        elif child.type in _TYPE_DECL_TYPES:
            _process_type_declaration(child, rel_path, parent_id, result,
                                      ns_name)
        elif child.type == "using_directive":
            _process_using(child, rel_path, result)
        elif child.type in ("namespace_declaration",
                            "file_scoped_namespace_declaration"):
            _process_namespace(child, rel_path, parent_id, result, ns_name)


def _process_namespace_member(ts_node, rel_path: str, parent_id: str,
                              result: ParseResult, namespace: str):
    """Process a member inside a namespace declaration_list."""
    node_type = ts_node.type

    if node_type in _TYPE_DECL_TYPES:
        _process_type_declaration(ts_node, rel_path, parent_id, result,
                                  namespace)
    elif node_type in ("namespace_declaration",
                        "file_scoped_namespace_declaration"):
        _process_namespace(ts_node, rel_path, parent_id, result, namespace)
    elif node_type == "using_directive":
        _process_using(ts_node, rel_path, result)


# ── Type declaration processing (class/interface/struct/enum/record) ──


def _process_type_declaration(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    namespace: str,
):
    """Extract a class, interface, struct, enum, or record declaration."""
    node_type = ts_node.type
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    type_name = _text(name_node)

    # Node ID: rel_path::Namespace.ClassName when in a namespace
    if namespace:
        node_id = f"{rel_path}::{namespace}.{type_name}"
    else:
        node_id = f"{rel_path}::{type_name}"

    # Collect modifiers
    modifiers = _get_modifiers(ts_node)

    # Collect attributes
    attributes = _get_attributes(ts_node)

    # Base types from base_list
    base_types = _extract_base_types(ts_node)

    # Build signature
    signature = _build_type_signature(node_type, type_name, base_types,
                                       modifiers, ts_node)

    # Tags
    tags = _compute_type_tags(
        node_type, type_name, modifiers, attributes, base_types,
    )

    # XML doc comment
    docstring = _get_xml_doc_for_node(ts_node)

    type_node = Node(
        id=node_id,
        kind="class",
        name=type_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(type_node)

    # Process body members (declaration_list)
    body = ts_node.child_by_field_name("body")
    if body is None:
        # Try finding declaration_list as a direct child
        for child in ts_node.children:
            if child.type in ("declaration_list",
                              "enum_member_declaration_list"):
                body = child
                break

    if body is not None:
        for child in body.children:
            _process_type_member(child, rel_path, node_id, type_name,
                                 result, namespace)


def _process_type_member(
    ts_node, rel_path: str, class_id: str, class_name: str,
    result: ParseResult, namespace: str,
):
    """Process a member inside a type declaration body."""
    node_type = ts_node.type

    if node_type == "method_declaration":
        _process_method(ts_node, rel_path, class_id, class_name, result)
    elif node_type == "constructor_declaration":
        _process_constructor(ts_node, rel_path, class_id, class_name, result)
    elif node_type == "property_declaration":
        _process_property(ts_node, rel_path, class_id, class_name, result)
    elif node_type in _TYPE_DECL_TYPES:
        # Nested type declaration
        _process_type_declaration(ts_node, rel_path, class_id, result,
                                  namespace)


# ── Method processing ────────────────────────────────────────────────


def _process_method(
    ts_node, rel_path: str, class_id: str, class_name: str,
    result: ParseResult,
):
    """Extract a method_declaration inside a type body."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    method_name = _text(name_node)
    node_id = f"{class_id}.{method_name}"

    # Modifiers
    modifiers = _get_modifiers(ts_node)

    # Attributes
    attributes = _get_attributes(ts_node)

    # Return type: collect type nodes between modifiers and the name identifier
    return_type_text = _extract_return_type(ts_node, method_name)

    # Parameters
    params_node = _find_child_by_type(ts_node, "parameter_list")
    params_text = _text(params_node) if params_node else "()"
    params_inner = _strip_parens(params_text)

    # Signature: "public async Task<int> MethodName(string param1, int param2)"
    sig_parts: list[str] = list(modifiers)
    if return_type_text:
        sig_parts.append(return_type_text)
    sig_parts.append(f"{method_name}({params_inner})")
    signature = " ".join(sig_parts)

    # Tags
    tags = _compute_method_tags(method_name, modifiers, attributes)

    # XML doc comment
    docstring = _get_xml_doc_for_node(ts_node)

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
        parent_id=class_id,
    )
    result.nodes.append(method_node)

    # Extract calls from the method body
    body = _find_child_by_type(ts_node, "block")
    if body is not None:
        _extract_calls_from_body(body, node_id, result)

    # Expression-bodied methods: => expression;
    arrow = _find_child_by_type(ts_node, "arrow_expression_clause")
    if arrow is not None:
        _walk_for_calls(arrow, node_id, result, error_context=None)


# ── Constructor processing ───────────────────────────────────────────


def _process_constructor(
    ts_node, rel_path: str, class_id: str, class_name: str,
    result: ParseResult,
):
    """Extract a constructor_declaration inside a type body."""
    # Constructor name = class name
    constructor_name = class_name
    node_id = f"{class_id}.{constructor_name}"

    # Modifiers
    modifiers = _get_modifiers(ts_node)

    # Parameters
    params_node = _find_child_by_type(ts_node, "parameter_list")
    params_text = _text(params_node) if params_node else "()"
    params_inner = _strip_parens(params_text)

    # Signature: "public ClassName(string param1, int param2)"
    sig_parts: list[str] = list(modifiers)
    sig_parts.append(f"{constructor_name}({params_inner})")
    signature = " ".join(sig_parts)

    # Tags
    tags: list[str] = []
    _apply_access_modifier_tags(modifiers, tags, default_private=True)
    if "static" in modifiers:
        tags.append("static")

    # XML doc comment
    docstring = _get_xml_doc_for_node(ts_node)

    ctor_node = Node(
        id=node_id,
        kind="method",
        name=constructor_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=class_id,
    )
    result.nodes.append(ctor_node)

    # Extract calls from the constructor body
    body = _find_child_by_type(ts_node, "block")
    if body is not None:
        _extract_calls_from_body(body, node_id, result)


# ── Property processing ──────────────────────────────────────────────


def _process_property(
    ts_node, rel_path: str, class_id: str, class_name: str,
    result: ParseResult,
):
    """Extract a property_declaration inside a type body."""
    name_node = _find_child_by_type(ts_node, "identifier")
    if name_node is None:
        return
    prop_name = _text(name_node)
    node_id = f"{class_id}.{prop_name}"

    # Modifiers
    modifiers = _get_modifiers(ts_node)

    # Return type (property type)
    type_text = _extract_return_type(ts_node, prop_name)

    # Build accessor summary: { get; set; }
    accessor_summary = _build_accessor_summary(ts_node)

    # Signature: "public string Name { get; set; }"
    sig_parts: list[str] = list(modifiers)
    if type_text:
        sig_parts.append(type_text)
    sig_parts.append(prop_name)
    if accessor_summary:
        sig_parts.append(accessor_summary)

    # Handle expression-bodied property
    arrow = _find_child_by_type(ts_node, "arrow_expression_clause")
    if arrow is not None and not accessor_summary:
        sig_parts.append("=> ...")

    signature = " ".join(sig_parts)

    # Tags -- property + access modifiers
    tags: list[str] = ["property"]
    _apply_access_modifier_tags(modifiers, tags, default_private=True)
    if "static" in modifiers:
        tags.append("static")
    if "abstract" in modifiers:
        tags.append("abstract")

    # XML doc comment
    docstring = _get_xml_doc_for_node(ts_node)

    prop_node = Node(
        id=node_id,
        kind="method",
        name=prop_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=class_id,
    )
    result.nodes.append(prop_node)

    # Extract calls from accessor bodies or expression body
    if arrow is not None:
        _walk_for_calls(arrow, node_id, result, error_context=None)

    accessor_list = _find_child_by_type(ts_node, "accessor_list")
    if accessor_list is not None:
        for accessor in accessor_list.children:
            if accessor.type == "accessor_declaration":
                body = _find_child_by_type(accessor, "block")
                if body is not None:
                    _extract_calls_from_body(body, node_id, result)
                acc_arrow = _find_child_by_type(accessor,
                                                 "arrow_expression_clause")
                if acc_arrow is not None:
                    _walk_for_calls(acc_arrow, node_id, result,
                                    error_context=None)


# ── Using directive processing ───────────────────────────────────────


def _process_using(ts_node, rel_path: str, result: ParseResult):
    """Process a using_directive."""
    # using System.Collections.Generic;
    # using static System.Math;
    # using Alias = Namespace.Type;

    is_static = False
    module_path = ""

    for child in ts_node.children:
        if child.type == "static":
            is_static = True
        elif not child.is_named and _text(child) == "static":
            is_static = True
        elif child.type in ("identifier", "qualified_name"):
            module_path = _text(child)
        elif child.type == "using_directive":
            # Nested — shouldn't happen normally, skip
            continue

    if not module_path:
        # Try to extract from the full text as a fallback
        full_text = _text(ts_node).strip()
        # Remove "using " prefix and ";" suffix
        if full_text.startswith("using "):
            remainder = full_text[6:].rstrip(";").strip()
            if remainder.startswith("static "):
                is_static = True
                remainder = remainder[7:].strip()
            # Handle alias: using Foo = Bar.Baz;
            if "=" in remainder:
                remainder = remainder.split("=", 1)[1].strip()
            module_path = remainder

    imported_names: list[str] = []
    if is_static:
        imported_names.append("*")  # static import imports all static members

    result.raw_imports.append(RawImport(
        importer_file=rel_path,
        module_path=module_path,
        imported_names=imported_names,
        is_from_import=is_static,
    ))


# ── Call extraction ──────────────────────────────────────────────────


def _extract_calls_from_body(body_node, caller_id: str, result: ParseResult):
    """Walk a method/constructor body to find all call expressions."""
    for child in body_node.children:
        _walk_for_calls(child, caller_id, result, error_context=None)


def _walk_for_calls(
    ts_node, caller_id: str, result: ParseResult,
    error_context: dict | None,
):
    """Recursively walk to find call expressions, tracking error context."""
    node_type = ts_node.type

    # Don't descend into nested type or method definitions
    if node_type in _TYPE_DECL_TYPES | {
        "method_declaration", "constructor_declaration",
        "local_function_statement", "lambda_expression",
        "anonymous_method_expression",
    }:
        return

    # Handle try_statement: process with error context
    if node_type == "try_statement":
        _process_try_statement(ts_node, caller_id, result)
        return

    # Handle invocation_expression (regular method calls)
    if node_type == "invocation_expression":
        _record_invocation(ts_node, caller_id, result, error_context)
        # Walk arguments for nested calls
        args = ts_node.child_by_field_name("arguments")
        if args is None:
            # Try to find argument_list child
            for child in ts_node.children:
                if child.type == "argument_list":
                    args = child
                    break
        if args is not None:
            for child in args.children:
                _walk_for_calls(child, caller_id, result, error_context)
        return

    # Handle object_creation_expression (new Foo())
    if node_type == "object_creation_expression":
        _record_object_creation(ts_node, caller_id, result, error_context)
        # Walk arguments for nested calls
        args = ts_node.child_by_field_name("arguments")
        if args is None:
            for child in ts_node.children:
                if child.type == "argument_list":
                    args = child
                    break
        if args is not None:
            for child in args.children:
                _walk_for_calls(child, caller_id, result, error_context)
        return

    # Recurse into children
    for child in ts_node.children:
        _walk_for_calls(child, caller_id, result, error_context)


def _process_try_statement(ts_node, caller_id: str, result: ParseResult):
    """Process a try/catch/finally statement, tracking exception types for calls in the try block."""
    # Collect exception types from catch clauses
    exception_types: list[str] = []

    for child in ts_node.children:
        if child.type == "catch_clause":
            exc_type = _extract_catch_exception_type(child)
            if exc_type:
                exception_types.append(exc_type)

    error_context: dict | None
    if exception_types:
        error_context = {
            "pattern": "try_catch",
            "exception": ",".join(exception_types),
        }
    else:
        error_context = {
            "pattern": "try_catch",
            "exception": "Exception",
        }

    # Walk the try block body with error context
    for child in ts_node.children:
        if child.type == "block":
            # First block child of try_statement is the try body
            for block_child in child.children:
                _walk_for_calls(block_child, caller_id, result, error_context)
            break

    # Walk catch/finally blocks without the try error context
    for child in ts_node.children:
        if child.type == "catch_clause":
            for sub in child.children:
                if sub.type == "block":
                    for block_child in sub.children:
                        _walk_for_calls(block_child, caller_id, result, error_context=None)
        elif child.type == "finally_clause":
            for sub in child.children:
                if sub.type == "block":
                    for block_child in sub.children:
                        _walk_for_calls(block_child, caller_id, result, error_context=None)


def _extract_catch_exception_type(catch_node) -> str | None:
    """Extract the exception type from a catch_clause's catch_declaration."""
    for child in catch_node.children:
        if child.type == "catch_declaration":
            # catch_declaration contains the exception type and optional variable name
            type_node = child.child_by_field_name("type")
            if type_node is not None:
                return _text(type_node)
            # Fallback: look for identifier or qualified_name children
            for sub in child.children:
                if sub.type in ("identifier", "qualified_name"):
                    return _text(sub)
    return None


def _record_invocation(call_node, caller_id: str, result: ParseResult,
                       error_context: dict | None):
    """Record a single invocation_expression as a RawCall."""
    # invocation_expression has a function child (member_access_expression or identifier)
    # and an argument_list child
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        # Fallback: first named child
        for child in call_node.children:
            if child.type in ("identifier", "member_access_expression",
                              "qualified_name", "generic_name"):
                func_node = child
                break
    if func_node is None and call_node.children:
        func_node = call_node.children[0]

    if func_node is None:
        return

    callee_text = _text(func_node)
    line = _line(call_node)
    return_usage = _determine_return_usage(call_node)

    result.raw_calls.append(RawCall(
        caller_id=caller_id,
        callee_text=callee_text,
        line=line,
        error_handling=error_context,
        return_usage=return_usage,
    ))


def _record_object_creation(new_node, caller_id: str, result: ParseResult,
                            error_context: dict | None):
    """Record an object_creation_expression (new Foo()) as a RawCall."""
    # The type being constructed
    type_node = new_node.child_by_field_name("type")
    if type_node is None:
        # Fallback: look for identifier or qualified name after 'new'
        for child in new_node.children:
            if child.type in ("identifier", "qualified_name", "generic_name"):
                type_node = child
                break
    if type_node is None:
        return

    callee_text = _text(type_node)
    line = _line(new_node)
    return_usage = _determine_return_usage(new_node)

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

    # await wrapping
    if parent_type == "await_expression":
        grandparent = parent.parent
        if grandparent is None:
            return "awaited_ignored"

        gp_type = grandparent.type
        if gp_type in ("variable_declaration", "assignment_expression",
                        "equals_value_clause"):
            return "awaited"
        if gp_type == "return_statement":
            return "awaited"
        if gp_type == "expression_statement":
            return "awaited_ignored"
        return "awaited"

    # Assignment
    if parent_type in ("variable_declaration", "assignment_expression",
                        "equals_value_clause", "variable_declarator"):
        return "assigned"

    # Return
    if parent_type == "return_statement":
        return "returned"

    # Condition of if/while
    if parent_type in ("if_statement", "while_statement", "do_statement",
                        "switch_statement"):
        return "checked"

    # Binary / unary expression
    if parent_type in ("binary_expression", "prefix_unary_expression"):
        return "checked"

    # Bare expression statement
    if parent_type == "expression_statement":
        return "ignored"

    # As argument to another call
    if parent_type in ("argument_list", "argument"):
        return "assigned"

    # Part of array/object initializer
    if parent_type in ("initializer_expression", "array_creation_expression"):
        return "assigned"

    # Member access: foo().Bar
    if parent_type == "member_access_expression":
        return "assigned"

    # Element access: foo()[0]
    if parent_type == "element_access_expression":
        return "assigned"

    return "ignored"


# ── Signature building ───────────────────────────────────────────────


def _build_type_signature(
    node_type: str, type_name: str, base_types: list[str],
    modifiers: list[str], ts_node,
) -> str:
    """Build a type signature like 'class ClassName : BaseClass, IInterface'."""
    keyword_map = {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "struct_declaration": "struct",
        "enum_declaration": "enum",
        "record_declaration": "record",
    }
    keyword = keyword_map.get(node_type, "class")

    sig = f"{keyword} {type_name}"

    # Type parameters (generics)
    type_params = _find_child_by_type(ts_node, "type_parameter_list")
    if type_params is not None:
        sig = f"{keyword} {type_name}{_text(type_params)}"

    if base_types:
        sig += f" : {', '.join(base_types)}"

    # For records with parameter lists
    if node_type == "record_declaration":
        param_list = _find_child_by_type(ts_node, "parameter_list")
        if param_list is not None:
            sig += _text(param_list)

    return sig


def _extract_return_type(ts_node, name: str) -> str:
    """Extract the return type text from a method or property declaration.

    Collects type nodes that appear between the last modifier/attribute
    and the identifier matching ``name``.  In tree-sitter-c-sharp, the
    return type can be a ``predefined_type`` (e.g. ``void``, ``int``),
    a ``generic_name`` (e.g. ``Task<int>``), or a plain ``identifier``
    (e.g. a user-defined class name like ``Person``).
    """
    _type_node_types = frozenset({
        "predefined_type", "generic_name", "qualified_name",
        "nullable_type", "array_type", "tuple_type", "pointer_type",
        "ref_type",
    })

    # Walk children: skip modifiers and attributes, then collect the first
    # type node we see before hitting the name identifier.
    past_modifiers = False
    type_texts: list[str] = []

    for child in ts_node.children:
        if child.type in ("attribute_list", "modifier"):
            past_modifiers = True
            continue

        if not past_modifiers:
            past_modifiers = True

        # If this is the name identifier, stop
        if child.type == "identifier" and _text(child) == name:
            break

        if child.type in _type_node_types:
            type_texts.append(_text(child))
        elif child.type == "identifier":
            # User-defined type name used as return type (e.g. Person)
            type_texts.append(_text(child))

    return " ".join(type_texts)


def _build_accessor_summary(ts_node) -> str:
    """Build accessor summary like '{ get; set; }' from an accessor_list."""
    accessor_list = _find_child_by_type(ts_node, "accessor_list")
    if accessor_list is None:
        return ""

    accessors: list[str] = []
    for child in accessor_list.children:
        if child.type == "accessor_declaration":
            acc_mods = _get_modifiers(child)
            keyword = ""
            for sub in child.children:
                if sub.type in ("get", "set", "init"):
                    keyword = _text(sub)
                    break
            if keyword:
                if acc_mods:
                    accessors.append(f"{' '.join(acc_mods)} {keyword}")
                else:
                    accessors.append(keyword)

    if accessors:
        return "{ " + "; ".join(accessors) + "; }"
    return ""


# ── Base type extraction ─────────────────────────────────────────────


def _extract_base_types(ts_node) -> list[str]:
    """Extract base types from a type declaration's base_list."""
    bases: list[str] = []
    for child in ts_node.children:
        if child.type == "base_list":
            for sub in child.children:
                if sub.type in ("identifier", "qualified_name", "generic_name"):
                    bases.append(_text(sub))
                elif sub.type == "simple_base_type":
                    bases.append(_text(sub))
                # Skip punctuation like ':', ','
            break
    return bases


# ── Modifier extraction ──────────────────────────────────────────────


def _get_modifiers(ts_node) -> list[str]:
    """Get all modifier keywords (public, static, abstract, etc.) from a declaration."""
    modifiers: list[str] = []
    for child in ts_node.children:
        if child.type == "modifier":
            modifiers.append(_text(child))
        # Some grammars put modifiers as bare keyword tokens
        elif child.type in (
            "public", "private", "protected", "internal",
            "static", "abstract", "virtual", "override",
            "sealed", "async", "readonly", "new", "partial",
            "extern", "volatile", "unsafe",
        ):
            modifiers.append(child.type)
        elif not child.is_named and _text(child) in (
            "public", "private", "protected", "internal",
            "static", "abstract", "virtual", "override",
            "sealed", "async", "readonly", "partial",
        ):
            modifiers.append(_text(child))
    return modifiers


# ── Attribute extraction ─────────────────────────────────────────────


def _get_attributes(ts_node) -> list[str]:
    """Get attribute names from attribute_list nodes on a declaration.

    Returns a list of attribute name strings (e.g., ["Test", "DataContract"]).
    """
    attrs: list[str] = []
    for child in ts_node.children:
        if child.type == "attribute_list":
            for sub in child.children:
                if sub.type == "attribute":
                    name_node = sub.child_by_field_name("name")
                    if name_node is not None:
                        attr_name = _text(name_node)
                        attrs.append(attr_name)
                    else:
                        # Fallback: get identifier children
                        for attr_child in sub.children:
                            if attr_child.type in ("identifier", "qualified_name"):
                                attrs.append(_text(attr_child))
                                break
    return attrs


# ── Tag computation ──────────────────────────────────────────────────


def _compute_type_tags(
    node_type: str, type_name: str, modifiers: list[str],
    attributes: list[str], base_types: list[str],
) -> list[str]:
    """Compute semantic tags for a type declaration."""
    tags: list[str] = []

    # Type-specific tags
    if node_type == "interface_declaration":
        tags.append("interface")
    elif node_type == "enum_declaration":
        tags.append("enum")
    elif node_type == "struct_declaration":
        tags.append("struct")

    # Modifier-based tags
    if "abstract" in modifiers:
        tags.append("abstract")
    if "static" in modifiers:
        tags.append("static")
    if "sealed" in modifiers:
        tags.append("sealed")

    # Access modifier tags (types don't default to private)
    if "private" in modifiers:
        tags.append("private")

    # extends tags for base types
    for base in base_types:
        tags.append(f"extends:{base}")

    # data_model: records, or [Serializable] / [DataContract] / etc.
    is_data_model = False
    if node_type == "record_declaration":
        is_data_model = True
    for attr in attributes:
        attr_base = attr.replace("Attribute", "") if attr.endswith("Attribute") else attr
        if attr_base in _DATA_MODEL_ATTRIBUTES:
            is_data_model = True
            break
    if is_data_model:
        tags.append("data_model")

    # test: class name contains "Test"
    if "Test" in type_name:
        tags.append("test")

    return tags


def _compute_method_tags(
    method_name: str, modifiers: list[str], attributes: list[str],
) -> list[str]:
    """Compute semantic tags for a method declaration."""
    tags: list[str] = []

    # entrypoint: method named Main
    if method_name == "Main":
        tags.append("entrypoint")

    # test: methods with [Test], [TestMethod], [Fact], [Theory] attributes
    for attr in attributes:
        attr_base = attr.replace("Attribute", "") if attr.endswith("Attribute") else attr
        if attr_base in _TEST_ATTRIBUTES:
            tags.append("test")
            break

    # async
    if "async" in modifiers:
        tags.append("async")

    # static
    if "static" in modifiers:
        tags.append("static")

    # abstract
    if "abstract" in modifiers:
        tags.append("abstract")

    # private: explicit or default (class members without access modifier
    # default to private in C#)
    _apply_access_modifier_tags(modifiers, tags, default_private=True)

    # factory: methods starting with Create, Build, Make
    for prefix in _FACTORY_PREFIXES:
        if method_name.startswith(prefix):
            tags.append("factory")
            break

    # callback: methods starting with On or Handle
    if (len(method_name) > 2 and method_name.startswith("On")
            and method_name[2].isupper()):
        tags.append("callback")
    elif (len(method_name) > 6 and method_name.startswith("Handle")
          and method_name[6].isupper()):
        tags.append("callback")

    return tags


def _apply_access_modifier_tags(modifiers: list[str], tags: list[str],
                                default_private: bool = False):
    """Add access modifier tags to the tags list.

    When ``default_private`` is True and no access modifier is present,
    the ``private`` tag is added (C# class members default to private).
    """
    if "private" in modifiers:
        tags.append("private")
    elif not any(m in modifiers for m in ("public", "protected", "internal")):
        if default_private:
            tags.append("private")


# ── XML doc comment extraction ───────────────────────────────────────


def _get_xml_doc_for_node(ts_node) -> str | None:
    """Extract C# XML doc comments (``///`` lines) preceding a declaration.

    Walks backward through previous named siblings collecting consecutive
    comment nodes that start with ``///``.
    """
    comments: list[str] = []
    sibling = ts_node.prev_named_sibling

    while sibling is not None and sibling.type == "comment":
        comment_text = _text(sibling)
        if comment_text.startswith("///"):
            # Strip the /// prefix
            line_text = comment_text[3:]
            if line_text.startswith(" "):
                line_text = line_text[1:]
            comments.append(line_text)

            # Check for blank line gap
            next_sib = sibling.next_named_sibling
            if next_sib is not None:
                gap = next_sib.start_point.row - sibling.end_point.row
                if gap > 1:
                    break

            sibling = sibling.prev_named_sibling
        else:
            # Non-doc comment -- stop
            break

    if not comments:
        return None

    # Collected in reverse order
    comments.reverse()
    raw = "\n".join(comments).strip()
    if not raw:
        return None

    # Strip XML tags to get plain text content
    cleaned_lines: list[str] = []
    for line in raw.split("\n"):
        stripped = _strip_xml_tags(line).strip()
        if stripped:
            cleaned_lines.append(stripped)

    result = " ".join(cleaned_lines).strip()
    return result if result else None


def _strip_xml_tags(text: str) -> str:
    """Remove XML tags from a string, keeping only the text content."""
    result: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "<":
            # Skip until closing >
            j = text.find(">", i)
            if j == -1:
                break
            i = j + 1
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


# ── Utility helpers ──────────────────────────────────────────────────


def _strip_parens(s: str) -> str:
    """Strip outer parentheses from a parameter list string."""
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        return s[1:-1].strip()
    return s


def _find_child_by_type(node, type_name: str):
    """Find the first child with the given type."""
    for child in node.children:
        if child.type == type_name:
            return child
    return None
