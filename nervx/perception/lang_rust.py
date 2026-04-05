"""Tree-sitter AST parser for Rust files (.rs)."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_rust as tsrust
from tree_sitter import Language, Parser

from nervx.perception.parser import (
    Node, RawCall, RawImport, ParseResult,
    _relative_path, _text, _line, _end_line,
)

# ── Language object ──────────────────────────────────────────────────

RUST_LANGUAGE = Language(tsrust.language())

# ── Factory-name prefixes ────────────────────────────────────────────

_FACTORY_PREFIXES = ("new", "create", "build", "from")


# ── Public entry point ───────────────────────────────────────────────


def parse_rust(file_path: str, repo_root: str) -> ParseResult:
    """Parse a Rust source file using tree-sitter and extract structural info."""
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
        parser = Parser(RUST_LANGUAGE)
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

    # File-level tags
    file_tags: list[str] = []
    lower_path = rel_path.lower().replace("\\", "/")
    if "test" in lower_path:
        file_tags.append("test")
    if file_name == "main.rs":
        file_tags.append("entrypoint")

    # Module-level doc comment (//! lines at top)
    module_doc = _extract_module_doc(root)

    # Build the file node
    file_node = Node(
        id=rel_path,
        kind="file",
        name=file_name,
        file_path=rel_path,
        line_start=1,
        line_end=total_lines,
        signature="",
        docstring=module_doc,
        tags=file_tags,
        parent_id="",
    )

    result = ParseResult(file_path=rel_path, nodes=[file_node])

    # Walk top-level children
    for child in root.children:
        _process_top_level(child, rel_path, file_node.id, result)

    # Build the error_handling summary dict
    for rc in result.raw_calls:
        if rc.error_handling is not None:
            eh_dict = result.error_handling.setdefault(rc.caller_id, {})
            eh_dict[rc.callee_text] = rc.error_handling

    return result


# ── Module doc extraction ────────────────────────────────────────────


def _extract_module_doc(root) -> str | None:
    """Extract //! inner doc comments from the top of the file."""
    lines: list[str] = []
    for child in root.children:
        if child.type == "line_comment":
            text = _text(child)
            if text.startswith("//!"):
                lines.append(text[3:].strip())
                continue
        # Once we hit a non-comment node, stop
        if child.type != "line_comment":
            break
    return "\n".join(lines) if lines else None


# ── Top-level dispatch ───────────────────────────────────────────────


def _process_top_level(ts_node, rel_path: str, parent_id: str, result: ParseResult):
    """Process a top-level tree-sitter node."""
    node_type = ts_node.type

    if node_type == "function_item":
        _process_function(ts_node, rel_path, parent_id, result,
                          class_name=None)
    elif node_type == "struct_item":
        _process_struct(ts_node, rel_path, parent_id, result)
    elif node_type == "enum_item":
        _process_enum(ts_node, rel_path, parent_id, result)
    elif node_type == "trait_item":
        _process_trait(ts_node, rel_path, parent_id, result)
    elif node_type == "impl_item":
        _process_impl(ts_node, rel_path, parent_id, result)
    elif node_type == "use_declaration":
        _process_use(ts_node, rel_path, result)
    elif node_type == "macro_definition":
        _process_macro_definition(ts_node, rel_path, parent_id, result)
    elif node_type == "mod_item":
        # Inline module: mod name { ... }
        _process_mod(ts_node, rel_path, parent_id, result)


# ── Struct processing ────────────────────────────────────────────────


def _process_struct(ts_node, rel_path: str, parent_id: str, result: ParseResult):
    """Extract a struct_item."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    struct_name = _text(name_node)
    node_id = f"{rel_path}::{struct_name}"

    is_pub = _has_visibility(ts_node)
    signature = f"struct {struct_name}"

    tags: list[str] = []
    if is_pub:
        tags.append("exported")
    else:
        tags.append("private")

    # Check for attributes like #[derive(...)], #[cfg(test)]
    attrs = _collect_attributes(ts_node)
    _apply_attribute_tags(attrs, tags)

    docstring = _get_doc_comment(ts_node)

    struct_node = Node(
        id=node_id,
        kind="class",
        name=struct_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(struct_node)


# ── Enum processing ──────────────────────────────────────────────────


def _process_enum(ts_node, rel_path: str, parent_id: str, result: ParseResult):
    """Extract an enum_item."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    enum_name = _text(name_node)
    node_id = f"{rel_path}::{enum_name}"

    is_pub = _has_visibility(ts_node)
    signature = f"enum {enum_name}"

    tags: list[str] = ["enum"]
    if is_pub:
        tags.append("exported")
    else:
        tags.append("private")

    attrs = _collect_attributes(ts_node)
    _apply_attribute_tags(attrs, tags)

    docstring = _get_doc_comment(ts_node)

    enum_node = Node(
        id=node_id,
        kind="class",
        name=enum_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(enum_node)


# ── Trait processing ─────────────────────────────────────────────────


def _process_trait(ts_node, rel_path: str, parent_id: str, result: ParseResult):
    """Extract a trait_item and its method signatures."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    trait_name = _text(name_node)
    node_id = f"{rel_path}::{trait_name}"

    is_pub = _has_visibility(ts_node)
    signature = f"trait {trait_name}"

    tags: list[str] = ["interface"]
    if is_pub:
        tags.append("exported")
    else:
        tags.append("private")

    attrs = _collect_attributes(ts_node)
    _apply_attribute_tags(attrs, tags)

    docstring = _get_doc_comment(ts_node)

    trait_node = Node(
        id=node_id,
        kind="class",
        name=trait_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(trait_node)

    # Process trait body for method signatures / default methods
    body = ts_node.child_by_field_name("body")
    if body is None:
        return

    for child in body.children:
        if child.type == "function_item":
            _process_function(child, rel_path, node_id, result,
                              class_name=trait_name)
        elif child.type == "function_signature_item":
            _process_function_signature(child, rel_path, node_id, result,
                                        class_name=trait_name)


# ── Impl block processing ───────────────────────────────────────────


def _process_impl(ts_node, rel_path: str, parent_id: str, result: ParseResult):
    """Extract an impl_item and its methods, associating with the type."""
    # Determine the type being implemented
    type_name = _get_impl_type_name(ts_node)
    if not type_name:
        return

    # Check for trait impl: impl Trait for Type
    trait_name = _get_impl_trait_name(ts_node)

    # The parent for methods is the type's node ID
    type_node_id = f"{rel_path}::{type_name}"

    # Process body
    body = ts_node.child_by_field_name("body")
    if body is None:
        return

    for child in body.children:
        if child.type == "function_item":
            _process_function(child, rel_path, type_node_id, result,
                              class_name=type_name, impl_trait=trait_name)
        elif child.type == "function_signature_item":
            _process_function_signature(child, rel_path, type_node_id, result,
                                        class_name=type_name,
                                        impl_trait=trait_name)


def _get_impl_type_name(impl_node) -> str:
    """Extract the type name from an impl_item.

    For `impl Type`, returns "Type".
    For `impl Trait for Type`, returns "Type".
    """
    # tree-sitter rust grammar: impl_item has `type` field for the implementing type
    type_node = impl_node.child_by_field_name("type")
    if type_node is not None:
        return _extract_type_name(type_node)

    # Fallback: walk children looking for the type after 'for' keyword or the sole type
    found_for = False
    last_type = ""
    for child in impl_node.children:
        if child.type == "for":
            found_for = True
            continue
        if found_for and child.type in ("type_identifier", "scoped_type_identifier",
                                         "generic_type"):
            return _extract_type_name(child)
        if child.type in ("type_identifier", "scoped_type_identifier", "generic_type"):
            last_type = _extract_type_name(child)
    return last_type


def _get_impl_trait_name(impl_node) -> str | None:
    """Extract the trait name from `impl Trait for Type`, or None if bare impl."""
    # tree-sitter rust grammar: impl_item has `trait` field
    trait_node = impl_node.child_by_field_name("trait")
    if trait_node is not None:
        return _extract_type_name(trait_node)

    # Fallback: look for pattern "impl TypeA for TypeB"
    found_impl = False
    for child in impl_node.children:
        if child.type == "impl":
            found_impl = True
            continue
        if found_impl and child.type == "for":
            # The type before 'for' is the trait
            break
        if found_impl and child.type in ("type_identifier", "scoped_type_identifier",
                                          "generic_type"):
            # Check if 'for' keyword follows
            next_sib = child.next_named_sibling
            if next_sib is not None:
                # Check unnamed siblings too
                for sib in impl_node.children:
                    if sib.start_byte > child.end_byte and _text(sib) == "for":
                        return _extract_type_name(child)
            break
    return None


def _extract_type_name(type_node) -> str:
    """Extract a clean type name from a type node (stripping generics)."""
    if type_node is None:
        return ""
    node_type = type_node.type
    if node_type == "type_identifier":
        return _text(type_node)
    if node_type == "scoped_type_identifier":
        return _text(type_node)
    if node_type == "generic_type":
        # generic_type has a 'type' child that is the base type
        inner = type_node.child_by_field_name("type")
        if inner is not None:
            return _extract_type_name(inner)
        # Fallback: first child
        for child in type_node.children:
            if child.type in ("type_identifier", "scoped_type_identifier"):
                return _text(child)
    return _text(type_node)


# ── Function processing ──────────────────────────────────────────────


def _process_function(ts_node, rel_path: str, parent_id: str, result: ParseResult,
                      class_name: str | None, impl_trait: str | None = None):
    """Extract a function_item (top-level function or method inside impl/trait)."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    func_name = _text(name_node)

    # Determine kind: method if inside an impl or trait (class_name is set)
    is_method = class_name is not None
    kind = "method" if is_method else "function"

    # Build node ID
    if class_name:
        node_id = f"{rel_path}::{class_name}.{func_name}"
    else:
        node_id = f"{rel_path}::{func_name}"

    # Visibility
    is_pub = _has_visibility(ts_node)

    # Async/unsafe checks
    is_async = _has_keyword(ts_node, "async")
    is_unsafe = _has_keyword(ts_node, "unsafe")

    # Signature: fn name(params) -> ReturnType
    signature = _build_fn_signature(ts_node, func_name)

    # Tags
    tags = _compute_function_tags(func_name, is_pub, is_async, is_unsafe,
                                  is_method, impl_trait)

    # Attributes
    attrs = _collect_attributes(ts_node)
    _apply_attribute_tags(attrs, tags)

    # Docstring (/// comments above the function)
    docstring = _get_doc_comment(ts_node)

    func_node = Node(
        id=node_id,
        kind=kind,
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


def _process_function_signature(ts_node, rel_path: str, parent_id: str,
                                result: ParseResult, class_name: str,
                                impl_trait: str | None = None):
    """Extract a function_signature_item (trait method without body)."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    func_name = _text(name_node)
    node_id = f"{rel_path}::{class_name}.{func_name}"

    is_pub = _has_visibility(ts_node)
    is_async = _has_keyword(ts_node, "async")
    is_unsafe = _has_keyword(ts_node, "unsafe")

    signature = _build_fn_signature(ts_node, func_name)

    tags = _compute_function_tags(func_name, is_pub, is_async, is_unsafe,
                                  is_method=True, impl_trait=impl_trait)
    tags.append("abstract")

    attrs = _collect_attributes(ts_node)
    _apply_attribute_tags(attrs, tags)

    docstring = _get_doc_comment(ts_node)

    sig_node = Node(
        id=node_id,
        kind="method",
        name=func_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(sig_node)


# ── Signature building ───────────────────────────────────────────────


def _build_fn_signature(ts_node, func_name: str) -> str:
    """Build a signature like: fn name(params) -> ReturnType"""
    params_node = ts_node.child_by_field_name("parameters")
    params_text = _text(params_node) if params_node else "()"

    # Strip outer parens and rebuild
    inner = params_text.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()

    # Return type
    return_type = ts_node.child_by_field_name("return_type")
    ret_text = ""
    if return_type is not None:
        raw_ret = _text(return_type)
        # tree-sitter may include '-> ' prefix
        if raw_ret.startswith("->"):
            raw_ret = raw_ret[2:].strip()
        ret_text = f" -> {raw_ret}"

    return f"fn {func_name}({inner}){ret_text}"


# ── Use / import processing ──────────────────────────────────────────


def _process_use(ts_node, rel_path: str, result: ParseResult):
    """Extract a use_declaration.

    Handles:
      use std::collections::HashMap;
      use crate::module::{Foo, Bar};
      use crate::module::*;
      use std::io::{self, Read};
    """
    # Walk into the use_tree child(ren)
    for child in ts_node.children:
        if child.type in ("use_as_clause", "scoped_identifier",
                          "identifier", "use_wildcard", "use_list",
                          "scoped_use_list"):
            _collect_use_paths(child, "", rel_path, result)


def _collect_use_paths(node, prefix: str, rel_path: str, result: ParseResult):
    """Recursively collect import paths from a use_tree."""
    node_type = node.type

    if node_type == "identifier":
        # Simple: use foo;
        name = _text(node)
        full_path = f"{prefix}{name}" if prefix else name
        # Split into module_path and name
        parts = full_path.rsplit("::", 1)
        if len(parts) == 2:
            result.raw_imports.append(RawImport(
                importer_file=rel_path,
                module_path=parts[0],
                imported_names=[parts[1]],
                is_from_import=True,
            ))
        else:
            result.raw_imports.append(RawImport(
                importer_file=rel_path,
                module_path=full_path,
                imported_names=[],
                is_from_import=False,
            ))

    elif node_type == "scoped_identifier":
        # use std::collections::HashMap;
        full = _text(node)
        full_path = f"{prefix}{full}" if prefix else full
        parts = full_path.rsplit("::", 1)
        if len(parts) == 2:
            result.raw_imports.append(RawImport(
                importer_file=rel_path,
                module_path=parts[0],
                imported_names=[parts[1]],
                is_from_import=True,
            ))
        else:
            result.raw_imports.append(RawImport(
                importer_file=rel_path,
                module_path=full_path,
                imported_names=[],
                is_from_import=False,
            ))

    elif node_type == "use_as_clause":
        # use foo::bar as baz;
        path_node = node.child_by_field_name("path")
        if path_node is not None:
            full = _text(path_node)
            full_path = f"{prefix}{full}" if prefix else full
            parts = full_path.rsplit("::", 1)
            if len(parts) == 2:
                result.raw_imports.append(RawImport(
                    importer_file=rel_path,
                    module_path=parts[0],
                    imported_names=[parts[1]],
                    is_from_import=True,
                ))
            else:
                result.raw_imports.append(RawImport(
                    importer_file=rel_path,
                    module_path=full_path,
                    imported_names=[],
                    is_from_import=False,
                ))

    elif node_type == "use_wildcard":
        # use crate::module::*;
        full = _text(node)
        full_path = f"{prefix}{full}" if prefix else full
        # Strip the trailing ::*
        module_path = full_path.rsplit("::*", 1)[0] if "::*" in full_path else full_path.rstrip("*")
        if module_path.endswith("::"):
            module_path = module_path[:-2]
        result.raw_imports.append(RawImport(
            importer_file=rel_path,
            module_path=module_path,
            imported_names=["*"],
            is_from_import=True,
        ))

    elif node_type == "scoped_use_list":
        # use std::io::{self, Read, Write};
        # Has a path and a use_list
        path_node = node.child_by_field_name("path")
        list_node = node.child_by_field_name("list")

        base_path = _text(path_node) if path_node else ""
        full_base = f"{prefix}{base_path}" if prefix else base_path

        if list_node is not None:
            names: list[str] = []
            for child in list_node.children:
                if child.type in ("identifier", "self"):
                    names.append(_text(child))
                elif child.type == "scoped_identifier":
                    # Nested path inside the list: e.g. collections::HashMap
                    names.append(_text(child))
                elif child.type == "use_as_clause":
                    path_c = child.child_by_field_name("path")
                    if path_c is not None:
                        names.append(_text(path_c))
                elif child.type == "use_wildcard":
                    names.append("*")
                elif child.type == "scoped_use_list":
                    # Nested scoped list — recurse
                    _collect_use_paths(child, f"{full_base}::" if full_base else "",
                                       rel_path, result)

            if names:
                result.raw_imports.append(RawImport(
                    importer_file=rel_path,
                    module_path=full_base,
                    imported_names=names,
                    is_from_import=True,
                ))

    elif node_type == "use_list":
        # Bare use list (unusual at top level but handle it)
        for child in node.children:
            if child.type not in (",", "{", "}"):
                _collect_use_paths(child, prefix, rel_path, result)


# ── Macro definition processing ──────────────────────────────────────


def _process_macro_definition(ts_node, rel_path: str, parent_id: str,
                              result: ParseResult):
    """Extract a macro_definition (macro_rules!)."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    macro_name = _text(name_node)
    node_id = f"{rel_path}::{macro_name}"

    tags: list[str] = ["macro"]
    docstring = _get_doc_comment(ts_node)

    macro_node = Node(
        id=node_id,
        kind="function",
        name=macro_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=f"macro_rules! {macro_name}",
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(macro_node)


# ── Module processing ────────────────────────────────────────────────


def _process_mod(ts_node, rel_path: str, parent_id: str, result: ParseResult):
    """Process an inline mod item: mod name { ... }."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return

    # Only process inline modules (those with a body / declaration_list)
    body = ts_node.child_by_field_name("body")
    if body is None:
        return

    # Recurse into the module body as if it were top-level
    for child in body.children:
        _process_top_level(child, rel_path, parent_id, result)


# ── Call extraction ──────────────────────────────────────────────────


def _extract_calls_from_body(body_node, caller_id: str, result: ParseResult):
    """Walk a function body to find all call expressions."""
    for child in body_node.children:
        _walk_for_calls(child, caller_id, result, error_context=None)


def _walk_for_calls(ts_node, caller_id: str, result: ParseResult,
                    error_context: dict | None):
    """Recursively walk to find call and method-call expressions."""
    node_type = ts_node.type

    # Don't descend into nested function definitions (closures are ok)
    if node_type == "function_item":
        return

    # Handle ? operator: try_expression wraps an expression with ?
    if node_type == "try_expression":
        _process_try_expression(ts_node, caller_id, result)
        return

    # Handle match expressions for Result pattern detection
    if node_type == "match_expression":
        _process_match_expression(ts_node, caller_id, result)
        return

    # Handle call_expression
    if node_type == "call_expression":
        _record_call(ts_node, caller_id, result, error_context)
        # Walk arguments for nested calls
        args = ts_node.child_by_field_name("arguments")
        if args is not None:
            for child in args.children:
                _walk_for_calls(child, caller_id, result, error_context)
        return

    # Handle method_call_expression (.method() syntax)
    if node_type == "method_call_expression":
        _record_method_call(ts_node, caller_id, result, error_context)
        # Walk arguments for nested calls
        args = ts_node.child_by_field_name("arguments")
        if args is not None:
            for child in args.children:
                _walk_for_calls(child, caller_id, result, error_context)
        # Walk the receiver for chained calls
        receiver = ts_node.child_by_field_name("value")
        if receiver is not None:
            _walk_for_calls(receiver, caller_id, result, error_context)
        return

    # Handle macro invocation: e.g. println!(...), vec![...]
    if node_type == "macro_invocation":
        _record_macro_call(ts_node, caller_id, result, error_context)
        # Walk inside the token tree for nested calls
        for child in ts_node.children:
            if child.type == "token_tree":
                for sub in child.children:
                    _walk_for_calls(sub, caller_id, result, error_context)
        return

    # Recurse into children
    for child in ts_node.children:
        _walk_for_calls(child, caller_id, result, error_context)


def _process_try_expression(ts_node, caller_id: str, result: ParseResult):
    """Handle a try_expression (expr?) — walk the inner expression with ? error context."""
    error_context = {"pattern": "question_mark", "exception": ""}
    # The inner expression is the first child (before the ? token)
    for child in ts_node.children:
        if child.type != "?":
            _walk_for_calls(child, caller_id, result, error_context)


def _process_match_expression(ts_node, caller_id: str, result: ParseResult):
    """Handle match expressions — detect match on call results."""
    value_node = ts_node.child_by_field_name("value")
    if value_node is not None:
        # If the match value is a call, record it with checked return usage
        if value_node.type in ("call_expression", "method_call_expression"):
            error_context = {"pattern": "match", "exception": ""}
            _walk_for_calls(value_node, caller_id, result, error_context)
        else:
            _walk_for_calls(value_node, caller_id, result, error_context=None)

    # Walk the match arms for calls
    body = ts_node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            _walk_for_calls(child, caller_id, result, error_context=None)


def _record_call(call_node, caller_id: str, result: ParseResult,
                 error_context: dict | None):
    """Record a call_expression as a RawCall."""
    func_node = call_node.child_by_field_name("function")
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


def _record_method_call(call_node, caller_id: str, result: ParseResult,
                        error_context: dict | None):
    """Record a method_call_expression (.method()) as a RawCall."""
    method_node = call_node.child_by_field_name("name")
    if method_node is None:
        return

    # Build the callee text including the receiver if available
    value_node = call_node.child_by_field_name("value")
    method_name = _text(method_node)

    if value_node is not None:
        receiver_text = _text(value_node)
        # Keep it reasonable in length
        if len(receiver_text) > 60:
            receiver_text = "..."
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


def _record_macro_call(call_node, caller_id: str, result: ParseResult,
                       error_context: dict | None):
    """Record a macro_invocation as a RawCall."""
    macro_node = call_node.child_by_field_name("macro")
    if macro_node is None:
        # Fallback: first child
        if call_node.children:
            macro_node = call_node.children[0]
        else:
            return

    callee_text = _text(macro_node)
    # Append ! for macro calls if not already present
    if not callee_text.endswith("!"):
        callee_text += "!"

    line = _line(call_node)
    return_usage = _determine_return_usage(call_node)

    result.raw_calls.append(RawCall(
        caller_id=caller_id,
        callee_text=callee_text,
        line=line,
        error_handling=error_context,
        return_usage=return_usage,
    ))


# ── Return usage determination ───────────────────────────────────────


def _determine_return_usage(call_node) -> str:
    """Determine how the return value of a call is used."""
    parent = call_node.parent
    if parent is None:
        return "ignored"

    parent_type = parent.type

    # let x = call(); or x = call();
    if parent_type in ("let_declaration", "assignment_expression"):
        return "assigned"

    # return call();
    if parent_type == "return_expression":
        return "returned"

    # Last expression in block (implicit return in Rust)
    if parent_type == "block" and _is_last_expression_in_block(call_node, parent):
        return "returned"

    # Part of if condition
    if parent_type in ("if_expression", "while_expression"):
        return "checked"

    # match value(call()) { ... }
    if parent_type == "match_expression":
        return "checked"

    # Expression statement (bare call)
    if parent_type == "expression_statement":
        return "ignored"

    # As argument to another call
    if parent_type == "arguments":
        return "assigned"

    # try_expression: call()?
    if parent_type == "try_expression":
        # Check the try_expression's own parent
        return _determine_return_usage(parent)

    # await: call().await
    if parent_type == "await_expression":
        return _determine_return_usage(parent)

    # Part of a method chain: call().method()
    if parent_type == "field_expression":
        return "assigned"

    # Index expression: call()[0]
    if parent_type == "index_expression":
        return "assigned"

    # Binary expression
    if parent_type == "binary_expression":
        return "checked"

    # Tuple / array / struct literal
    if parent_type in ("tuple_expression", "array_expression"):
        return "assigned"

    return "ignored"


def _is_last_expression_in_block(node, block) -> bool:
    """Check if a node is the last expression in a block (implicit return)."""
    children = [c for c in block.children if c.type not in ("{", "}")]
    if not children:
        return False
    last = children[-1]
    # The last child might be an expression_statement or a bare expression
    if last is node:
        return True
    # If the last child is an expression_statement containing our node
    if last.type == "expression_statement":
        for c in last.children:
            if c is node:
                return True
    return False


# ── Doc comment extraction ───────────────────────────────────────────


def _get_doc_comment(ts_node) -> str | None:
    """Extract /// doc comments immediately preceding a node.

    In tree-sitter rust, doc comments (/// ...) are line_comment nodes that
    appear as siblings before the item node.
    """
    parent = ts_node.parent
    if parent is None:
        return None

    # Find our position among siblings
    siblings = list(parent.children)
    try:
        idx = siblings.index(ts_node)
    except ValueError:
        return None

    # Collect consecutive /// comments immediately before this node
    doc_lines: list[str] = []
    i = idx - 1
    while i >= 0:
        sib = siblings[i]
        if sib.type == "line_comment":
            text = _text(sib)
            if text.startswith("///"):
                doc_lines.insert(0, text[3:].strip())
                i -= 1
                continue
        # Also skip attribute_item nodes (they can interleave)
        if sib.type == "attribute_item":
            i -= 1
            continue
        break

    return "\n".join(doc_lines) if doc_lines else None


# ── Attribute handling ───────────────────────────────────────────────


def _collect_attributes(ts_node) -> list[str]:
    """Collect #[...] attribute texts for a node.

    Attributes appear as preceding sibling attribute_item nodes.
    """
    parent = ts_node.parent
    if parent is None:
        return []

    siblings = list(parent.children)
    try:
        idx = siblings.index(ts_node)
    except ValueError:
        return []

    attrs: list[str] = []
    i = idx - 1
    while i >= 0:
        sib = siblings[i]
        if sib.type == "attribute_item":
            attrs.insert(0, _text(sib))
            i -= 1
            continue
        if sib.type == "line_comment":
            i -= 1
            continue
        break

    return attrs


def _apply_attribute_tags(attrs: list[str], tags: list[str]):
    """Apply tags based on #[...] attributes."""
    for attr in attrs:
        attr_lower = attr.lower()
        # #[test]
        if "#[test]" in attr_lower:
            if "test" not in tags:
                tags.append("test")
        # #[cfg(test)]
        if "#[cfg(test)]" in attr_lower:
            if "test" not in tags:
                tags.append("test")
        # #[tokio::test] or #[async_std::test]
        if "::test" in attr_lower:
            if "test" not in tags:
                tags.append("test")
        # #[derive(...)] — note derives for tags
        if "derive" in attr_lower:
            if "derive" not in tags:
                tags.append("derive")
            # Detect data_model: Serialize, Deserialize, or serde-related derives
            _DATA_MODEL_DERIVES = ("serialize", "deserialize", "serde")
            for dm in _DATA_MODEL_DERIVES:
                if dm in attr_lower:
                    if "data_model" not in tags:
                        tags.append("data_model")
                    break


# ── Visibility helpers ───────────────────────────────────────────────


def _has_visibility(ts_node) -> bool:
    """Check if a node has a visibility_modifier (pub, pub(crate), etc.)."""
    for child in ts_node.children:
        if child.type == "visibility_modifier":
            return True
    return False


def _has_keyword(ts_node, keyword: str) -> bool:
    """Check if a node has a specific keyword as a direct child."""
    for child in ts_node.children:
        if _text(child) == keyword:
            return True
    return False


# ── Tag computation ──────────────────────────────────────────────────


def _compute_function_tags(func_name: str, is_pub: bool, is_async: bool,
                           is_unsafe: bool, is_method: bool,
                           impl_trait: str | None = None) -> list[str]:
    """Compute semantic tags for a Rust function or method."""
    tags: list[str] = []

    if is_pub:
        tags.append("exported")
    else:
        tags.append("private")

    if is_async:
        tags.append("async")

    if is_unsafe:
        tags.append("unsafe")

    if func_name == "main" and not is_method:
        tags.append("entrypoint")

    # Factory pattern: new, create, build, from
    for prefix in _FACTORY_PREFIXES:
        if func_name == prefix or func_name.startswith(f"{prefix}_"):
            tags.append("factory")
            break

    # Trait implementation
    if impl_trait:
        tags.append(f"impl:{impl_trait}")

    # Callback pattern: on_ or handle_ prefixes
    if func_name.startswith("on_") or func_name.startswith("handle_"):
        tags.append("callback")

    return tags
