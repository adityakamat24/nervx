"""Tree-sitter AST parser for JavaScript, JSX, TypeScript, and TSX files."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser

from nervx.perception.parser import (
    Node, RawCall, RawImport, ParseResult,
    _relative_path, _text, _line, _end_line, _split_identifier,
    STOP_WORDS, ROUTE_DECORATOR_KEYWORDS, DATA_MODEL_BASES,
)

# ── Language objects ──────────────────────────────────────────────────

JS_LANGUAGE = Language(tsjs.language())
TS_LANGUAGE = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())

# ── Entrypoint file basenames ─────────────────────────────────────────

_ENTRYPOINT_FILES = frozenset({
    "index.js", "index.ts", "index.jsx", "index.tsx",
    "app.js", "app.ts", "app.jsx", "app.tsx",
    "server.js", "server.ts", "main.js", "main.ts",
})

# ── Public entry points ──────────────────────────────────────────────


def parse_javascript(file_path: str, repo_root: str) -> ParseResult:
    """Parse a .js or .jsx file."""
    return _parse_js_ts(file_path, repo_root, JS_LANGUAGE)


def parse_typescript(file_path: str, repo_root: str) -> ParseResult:
    """Parse a .ts file."""
    return _parse_js_ts(file_path, repo_root, TS_LANGUAGE)


def parse_tsx(file_path: str, repo_root: str) -> ParseResult:
    """Parse a .tsx file."""
    return _parse_js_ts(file_path, repo_root, TSX_LANGUAGE)


# ── Shared implementation ─────────────────────────────────────────────


def _parse_js_ts(file_path: str, repo_root: str, language: Language) -> ParseResult:
    """Core parser shared by JS, TS, and TSX."""
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
        parser = Parser(language)
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

    # Is this file a test file?
    lower_path = rel_path.lower().replace("\\", "/")
    is_test_file = (
        "test" in lower_path
        or "spec" in lower_path
        or "__tests__" in lower_path
    )

    # Is this file an entrypoint?
    is_entrypoint_file = file_name.lower() in _ENTRYPOINT_FILES

    # File tags
    file_tags: list[str] = []
    if is_test_file:
        file_tags.append("test")
    if is_entrypoint_file:
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
        docstring=_extract_leading_jsdoc(root, 0),
        tags=file_tags,
        parent_id="",
    )

    result = ParseResult(file_path=rel_path, nodes=[file_node])

    # Walk top-level children
    for child in root.children:
        _process_top_level(
            child, rel_path, file_node.id, result,
            is_test_file=is_test_file,
            is_entrypoint_file=is_entrypoint_file,
        )

    # Build error_handling summary dict
    for rc in result.raw_calls:
        if rc.error_handling is not None:
            eh_dict = result.error_handling.setdefault(rc.caller_id, {})
            eh_dict[rc.callee_text] = rc.error_handling

    return result


# ── Top-level processing ──────────────────────────────────────────────


def _process_top_level(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_test_file: bool, is_entrypoint_file: bool,
):
    """Process a top-level tree-sitter node (JS/TS)."""
    node_type = ts_node.type

    # Unwrap export statements to get the inner declaration
    is_exported = False
    actual_node = ts_node
    if node_type == "export_statement":
        is_exported = True
        actual_node = _unwrap_export(ts_node)
        if actual_node is None:
            # export default <expression> or module.exports — still scan for calls
            _scan_export_for_calls_and_imports(ts_node, rel_path, parent_id, result)
            return
        node_type = actual_node.type

    if node_type == "class_declaration":
        _process_class(actual_node, rel_path, parent_id, result,
                       is_exported=is_exported,
                       is_test_file=is_test_file,
                       is_entrypoint_file=is_entrypoint_file)
    elif node_type == "abstract_class_declaration":
        _process_class(actual_node, rel_path, parent_id, result,
                       is_exported=is_exported,
                       is_test_file=is_test_file,
                       is_entrypoint_file=is_entrypoint_file,
                       is_abstract=True)
    elif node_type == "function_declaration":
        _process_function_decl(actual_node, rel_path, parent_id, result,
                               is_exported=is_exported,
                               is_test_file=is_test_file,
                               is_entrypoint_file=is_entrypoint_file)
    elif node_type == "generator_function_declaration":
        _process_function_decl(actual_node, rel_path, parent_id, result,
                               is_exported=is_exported,
                               is_test_file=is_test_file,
                               is_entrypoint_file=is_entrypoint_file,
                               extra_tags=["generator"])
    elif node_type in ("lexical_declaration", "variable_declaration"):
        _process_variable_declaration(actual_node, rel_path, parent_id, result,
                                      is_exported=is_exported,
                                      is_test_file=is_test_file,
                                      is_entrypoint_file=is_entrypoint_file)
    elif node_type == "import_statement":
        _process_import_statement(actual_node, rel_path, result)
    elif node_type == "interface_declaration":
        _process_interface(actual_node, rel_path, parent_id, result,
                           is_exported=is_exported)
    elif node_type == "type_alias_declaration":
        _process_type_alias(actual_node, rel_path, parent_id, result,
                            is_exported=is_exported)
    elif node_type == "enum_declaration":
        _process_enum(actual_node, rel_path, parent_id, result,
                      is_exported=is_exported)
    elif node_type == "expression_statement":
        _process_expression_statement(actual_node, rel_path, parent_id, result,
                                      is_test_file=is_test_file,
                                      is_entrypoint_file=is_entrypoint_file)


def _unwrap_export(export_node):
    """Return the inner declaration of an export_statement, or None."""
    # export default class Foo {} / export function bar() {} / export const x = ...
    for child in export_node.children:
        if child.type in (
            "class_declaration", "abstract_class_declaration",
            "function_declaration", "generator_function_declaration",
            "lexical_declaration", "variable_declaration",
            "interface_declaration", "type_alias_declaration",
            "enum_declaration",
        ):
            return child
    return None


def _scan_export_for_calls_and_imports(export_node, rel_path, parent_id, result):
    """Handle 'export default <expr>' or re-exports for calls/imports."""
    for child in export_node.children:
        if child.type == "import_statement":
            _process_import_statement(child, rel_path, result)
        # e.g.  export { foo } from 'bar'  — treat as an import
        source_node = export_node.child_by_field_name("source")
        if source_node is not None:
            module_path = _strip_quotes(_text(source_node))
            names: list[str] = []
            for c in export_node.children:
                if c.type == "export_clause":
                    for spec in c.children:
                        if spec.type == "export_specifier":
                            name_c = spec.child_by_field_name("name")
                            if name_c:
                                names.append(_text(name_c))
            result.raw_imports.append(RawImport(
                importer_file=rel_path,
                module_path=module_path,
                imported_names=names,
                is_from_import=True,
            ))
            return


# ── Class processing ──────────────────────────────────────────────────


def _process_class(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_exported: bool,
    is_test_file: bool,
    is_entrypoint_file: bool,
    is_abstract: bool = False,
):
    """Extract a class declaration and its methods."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    class_name = _text(name_node)
    node_id = f"{rel_path}::{class_name}"

    # Base classes from extends clause
    base_classes = _extract_extends(ts_node)

    # Implements clause (TypeScript)
    implements = _extract_implements(ts_node)

    # Signature
    if base_classes:
        signature = f"class {class_name} extends {', '.join(base_classes)}"
    else:
        signature = f"class {class_name}"

    # Tags
    tags = _compute_class_tags_js(
        class_name, base_classes, implements,
        is_exported=is_exported,
        is_abstract=is_abstract,
        is_test_file=is_test_file,
    )

    # JSDoc
    docstring = _get_jsdoc_for_node(ts_node)

    class_node = Node(
        id=node_id,
        kind="class",
        name=class_name,
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
    if body is None:
        return

    for child in body.children:
        child_type = child.type
        if child_type == "method_definition":
            _process_method(child, rel_path, node_id, class_name, result,
                            is_test_file=is_test_file,
                            is_entrypoint_file=is_entrypoint_file)
        elif child_type == "abstract_method_signature":
            _process_abstract_method_signature(
                child, rel_path, node_id, class_name, result)
        elif child_type in ("public_field_definition", "field_definition",
                            "property_definition"):
            # Check if the value is an arrow function
            value = child.child_by_field_name("value")
            if value is not None and value.type == "arrow_function":
                _process_arrow_in_class(child, value, rel_path, node_id,
                                        class_name, result,
                                        is_test_file=is_test_file,
                                        is_entrypoint_file=is_entrypoint_file)
        elif child_type == "class_declaration":
            _process_class(child, rel_path, node_id, result,
                           is_exported=False,
                           is_test_file=is_test_file,
                           is_entrypoint_file=is_entrypoint_file)
        elif child_type == "abstract_class_declaration":
            _process_class(child, rel_path, node_id, result,
                           is_exported=False,
                           is_test_file=is_test_file,
                           is_entrypoint_file=is_entrypoint_file,
                           is_abstract=True)


def _extract_extends(class_node) -> list[str]:
    """Extract base class names from an extends/heritage clause."""
    bases: list[str] = []
    for child in class_node.children:
        # TypeScript uses extends_clause directly
        if child.type == "extends_clause":
            for sub in child.children:
                if sub.type in ("identifier", "member_expression",
                                "generic_type", "type_identifier"):
                    bases.append(_text(sub))
            break
        # JavaScript uses class_heritage which may contain extends keyword
        # and the base class identifier(s) directly, or may wrap an extends_clause
        if child.type == "class_heritage":
            for sub in child.children:
                if sub.type == "extends_clause":
                    # TS-style nested extends_clause inside class_heritage
                    for sub2 in sub.children:
                        if sub2.type in ("identifier", "member_expression",
                                         "generic_type", "type_identifier"):
                            bases.append(_text(sub2))
                elif sub.type in ("identifier", "member_expression",
                                  "generic_type", "type_identifier"):
                    # JS-style: identifier is direct child of class_heritage
                    bases.append(_text(sub))
    return bases


def _extract_implements(class_node) -> list[str]:
    """Extract interface names from an implements clause (TypeScript)."""
    impls: list[str] = []
    for child in class_node.children:
        if child.type == "implements_clause":
            for sub in child.children:
                if sub.type in ("identifier", "type_identifier", "generic_type"):
                    impls.append(_text(sub))
        elif child.type == "class_heritage":
            for sub in child.children:
                if sub.type == "implements_clause":
                    for sub2 in sub.children:
                        if sub2.type in ("identifier", "type_identifier",
                                         "generic_type"):
                            impls.append(_text(sub2))
    return impls


# ── Method processing ─────────────────────────────────────────────────


def _process_method(
    ts_node, rel_path: str, class_id: str, class_name: str,
    result: ParseResult,
    is_test_file: bool,
    is_entrypoint_file: bool,
):
    """Extract a method_definition inside a class body."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    method_name = _text(name_node)
    node_id = f"{rel_path}::{class_name}.{method_name}"

    # Async check
    is_async = _has_keyword_child(ts_node, "async")

    # Static check
    is_static = _has_keyword_child(ts_node, "static")

    # Abstract check (TypeScript)
    is_abstract = _has_keyword_child(ts_node, "abstract")

    # Accessibility (private/protected/public via TS keywords)
    accessibility = _get_accessibility(ts_node)

    # Signature
    signature = _build_method_signature(ts_node, method_name)

    # Tags
    tags = _compute_method_tags(
        method_name, is_async, is_static, is_abstract, accessibility,
        is_test_file=is_test_file,
        is_entrypoint_file=is_entrypoint_file,
    )

    # JSDoc
    docstring = _get_jsdoc_for_node(ts_node)

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
    body = ts_node.child_by_field_name("body")
    if body is not None:
        _extract_calls_from_body(body, node_id, result)


def _process_abstract_method_signature(
    ts_node, rel_path: str, class_id: str, class_name: str,
    result: ParseResult,
):
    """Process a TypeScript abstract_method_signature (abstract method with no body)."""
    # The name is a property_identifier child (not a named field in this node type)
    method_name = ""
    params_text = "()"
    ret_suffix = ""

    for child in ts_node.children:
        if child.type == "property_identifier":
            method_name = _text(child)
        elif child.type == "formal_parameters":
            params_text = _text(child)
        elif child.type == "type_annotation":
            ret_text = _text(child).lstrip(": ")
            if ret_text:
                ret_suffix = f": {ret_text}"

    if not method_name:
        return

    node_id = f"{rel_path}::{class_name}.{method_name}"
    params_inner = _strip_parens(params_text)
    signature = f"{method_name}({params_inner}){ret_suffix}"

    tags = ["abstract"]

    # Accessibility
    accessibility = _get_accessibility(ts_node)
    if accessibility == "private":
        tags.append("private")
    elif accessibility == "protected":
        tags.append("protected")

    docstring = _get_jsdoc_for_node(ts_node)

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


def _process_arrow_in_class(
    field_node, arrow_node, rel_path: str, class_id: str, class_name: str,
    result: ParseResult,
    is_test_file: bool,
    is_entrypoint_file: bool,
):
    """Process a class field whose value is an arrow function."""
    name_node = field_node.child_by_field_name("name")
    if name_node is None:
        return
    method_name = _text(name_node)
    node_id = f"{rel_path}::{class_name}.{method_name}"

    is_async = _has_keyword_child(arrow_node, "async")
    is_static = _has_keyword_child(field_node, "static")
    accessibility = _get_accessibility(field_node)

    params_node = arrow_node.child_by_field_name("parameters")
    params_text = _text(params_node) if params_node else "()"
    params_inner = _strip_parens(params_text)
    signature = f"{method_name}({params_inner})"

    tags = _compute_method_tags(
        method_name, is_async, is_static, False, accessibility,
        is_test_file=is_test_file,
        is_entrypoint_file=is_entrypoint_file,
    )

    docstring = _get_jsdoc_for_node(field_node)

    method_node = Node(
        id=node_id,
        kind="method",
        name=method_name,
        file_path=rel_path,
        line_start=_line(field_node),
        line_end=_end_line(field_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=class_id,
    )
    result.nodes.append(method_node)

    body = arrow_node.child_by_field_name("body")
    if body is not None:
        _extract_calls_from_body(body, node_id, result)


# ── Function declaration processing ──────────────────────────────────


def _process_function_decl(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_exported: bool,
    is_test_file: bool,
    is_entrypoint_file: bool,
    extra_tags: list[str] | None = None,
):
    """Extract a function_declaration or generator_function_declaration."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    func_name = _text(name_node)
    node_id = f"{rel_path}::{func_name}"

    is_async = _has_keyword_child(ts_node, "async")

    # Signature: funcName(params): ReturnType
    signature = _build_function_signature(ts_node, func_name)

    tags = _compute_function_tags_js(
        func_name, is_async, is_exported,
        is_test_file=is_test_file,
        is_entrypoint_file=is_entrypoint_file,
    )
    if extra_tags:
        tags.extend(extra_tags)

    docstring = _get_jsdoc_for_node(ts_node)

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

    body = ts_node.child_by_field_name("body")
    if body is not None:
        _extract_calls_from_body(body, node_id, result)


# ── Variable declaration (arrow functions, etc.) ──────────────────────


def _process_variable_declaration(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_exported: bool,
    is_test_file: bool,
    is_entrypoint_file: bool,
):
    """Process const/let/var declarations, looking for arrow functions."""
    for child in ts_node.children:
        if child.type == "variable_declarator":
            _process_variable_declarator(
                child, rel_path, parent_id, result,
                is_exported=is_exported,
                is_test_file=is_test_file,
                is_entrypoint_file=is_entrypoint_file,
            )


def _process_variable_declarator(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_exported: bool,
    is_test_file: bool,
    is_entrypoint_file: bool,
):
    """Process a single variable_declarator — emit a function node if RHS is an arrow/function."""
    name_node = ts_node.child_by_field_name("name")
    value_node = ts_node.child_by_field_name("value")
    if name_node is None or value_node is None:
        return

    var_name = _text(name_node)

    # Unwrap type assertion / as expression wrapping the value
    actual_value = value_node
    if actual_value.type in ("as_expression", "type_assertion", "satisfies_expression"):
        for c in actual_value.children:
            if c.type in ("arrow_function", "function_expression",
                          "generator_function"):
                actual_value = c
                break

    if actual_value.type in ("arrow_function", "function_expression",
                             "generator_function"):
        _process_arrow_or_func_expr(
            actual_value, var_name, ts_node, rel_path, parent_id, result,
            is_exported=is_exported,
            is_test_file=is_test_file,
            is_entrypoint_file=is_entrypoint_file,
        )
    elif actual_value.type == "call_expression":
        # Check for require('module') calls — treat as imports
        callee = actual_value.child_by_field_name("function")
        if callee is not None and _text(callee) == "require":
            _process_require_call(actual_value, rel_path, result)
        # Other call expressions at top level (e.g. const router = express.Router())
        # are not function definitions — skip for now


def _process_arrow_or_func_expr(
    func_node, var_name: str, declarator_node,
    rel_path: str, parent_id: str, result: ParseResult,
    is_exported: bool,
    is_test_file: bool,
    is_entrypoint_file: bool,
):
    """Process an arrow_function or function_expression assigned to a variable."""
    node_id = f"{rel_path}::{var_name}"

    is_async = _has_keyword_child(func_node, "async")

    params_node = func_node.child_by_field_name("parameters")
    params_text = _text(params_node) if params_node else "()"
    params_inner = _strip_parens(params_text)

    # Return type annotation (TypeScript)
    return_type = func_node.child_by_field_name("return_type")
    ret_suffix = ""
    if return_type is not None:
        ret_text = _text(return_type)
        # tree-sitter may include the colon in the return type text
        ret_text = ret_text.lstrip(": ")
        if ret_text:
            ret_suffix = f": {ret_text}"

    signature = f"{var_name}({params_inner}){ret_suffix}"

    extra_tags: list[str] = []
    if func_node.type == "generator_function":
        extra_tags.append("generator")

    tags = _compute_function_tags_js(
        var_name, is_async, is_exported,
        is_test_file=is_test_file,
        is_entrypoint_file=is_entrypoint_file,
    )
    tags.extend(extra_tags)

    docstring = _get_jsdoc_for_node(declarator_node)
    if docstring is None:
        # Try the parent (lexical_declaration / variable_declaration)
        parent = declarator_node.parent
        if parent is not None:
            docstring = _get_jsdoc_for_node(parent)

    fn_node = Node(
        id=node_id,
        kind="function",
        name=var_name,
        file_path=rel_path,
        line_start=_line(declarator_node),
        line_end=_end_line(func_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(fn_node)

    body = func_node.child_by_field_name("body")
    if body is not None:
        _extract_calls_from_body(body, node_id, result)


# ── Expression statement processing ──────────────────────────────────


def _process_expression_statement(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_test_file: bool,
    is_entrypoint_file: bool,
):
    """Handle expression statements — module.exports, require(), test calls, etc."""
    if not ts_node.children:
        return

    expr = ts_node.children[0]

    # module.exports = ...
    if expr.type == "assignment_expression":
        left = expr.child_by_field_name("left")
        if left is not None and _text(left) == "module.exports":
            # Scan the RHS for calls
            right = expr.child_by_field_name("right")
            if right is not None:
                _walk_for_calls(right, parent_id, result, error_context=None)
            return

    # Top-level call expressions (e.g. require(), test calls at module level)
    if expr.type == "call_expression":
        callee = expr.child_by_field_name("function")
        if callee is not None:
            callee_text = _text(callee)
            # require() at top level — treat as import
            if callee_text == "require":
                _process_require_call(expr, rel_path, result)
                return


# ── TypeScript-specific: interface, type alias, enum ──────────────────


def _process_interface(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_exported: bool,
):
    """Process a TypeScript interface declaration as a class-kind node."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    iface_name = _text(name_node)
    node_id = f"{rel_path}::{iface_name}"

    # Extends
    extends: list[str] = []
    for child in ts_node.children:
        if child.type == "extends_type_clause":
            for sub in child.children:
                if sub.type in ("identifier", "type_identifier", "generic_type"):
                    extends.append(_text(sub))

    sig = f"interface {iface_name}"
    if extends:
        sig += f" extends {', '.join(extends)}"

    tags: list[str] = ["interface"]
    if is_exported:
        tags.append("export")
    if extends:
        tags.append(f"extends:{','.join(extends)}")

    docstring = _get_jsdoc_for_node(ts_node)

    iface_node = Node(
        id=node_id,
        kind="class",
        name=iface_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=sig,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(iface_node)


def _process_type_alias(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_exported: bool,
):
    """Process a TypeScript type alias declaration."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    type_name = _text(name_node)
    node_id = f"{rel_path}::{type_name}"

    sig = f"type {type_name}"
    # Try to get the value text (abbreviated)
    value_node = ts_node.child_by_field_name("value")
    if value_node is not None:
        val_text = _text(value_node)
        if len(val_text) <= 80:
            sig = f"type {type_name} = {val_text}"

    tags: list[str] = ["type_alias"]
    if is_exported:
        tags.append("export")

    docstring = _get_jsdoc_for_node(ts_node)

    type_node = Node(
        id=node_id,
        kind="class",
        name=type_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=sig,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(type_node)


def _process_enum(
    ts_node, rel_path: str, parent_id: str, result: ParseResult,
    is_exported: bool,
):
    """Process a TypeScript enum declaration."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    enum_name = _text(name_node)
    node_id = f"{rel_path}::{enum_name}"

    sig = f"enum {enum_name}"

    tags: list[str] = ["enum"]
    if is_exported:
        tags.append("export")

    docstring = _get_jsdoc_for_node(ts_node)

    enum_node = Node(
        id=node_id,
        kind="class",
        name=enum_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=sig,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(enum_node)


# ── Import processing ─────────────────────────────────────────────────


def _process_import_statement(ts_node, rel_path: str, result: ParseResult):
    """Process an ES6 import_statement."""
    source_node = ts_node.child_by_field_name("source")
    if source_node is None:
        return

    module_path = _strip_quotes(_text(source_node))
    imported_names: list[str] = []

    for child in ts_node.children:
        if child.type == "import_clause":
            for sub in child.children:
                if sub.type == "identifier":
                    # default import: import foo from 'bar'
                    imported_names.append(_text(sub))
                elif sub.type == "named_imports":
                    # import { a, b as c } from 'bar'
                    for spec in sub.children:
                        if spec.type == "import_specifier":
                            name_c = spec.child_by_field_name("name")
                            if name_c:
                                imported_names.append(_text(name_c))
                elif sub.type == "namespace_import":
                    # import * as foo from 'bar'
                    for ns_child in sub.children:
                        if ns_child.type == "identifier":
                            imported_names.append(f"* as {_text(ns_child)}")

    result.raw_imports.append(RawImport(
        importer_file=rel_path,
        module_path=module_path,
        imported_names=imported_names,
        is_from_import=True,
    ))


def _process_require_call(call_node, rel_path: str, result: ParseResult):
    """Process a require('module') call as an import."""
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return

    for arg in args.children:
        if arg.type == "string":
            module_path = _strip_quotes(_text(arg))
            # Try to get the assigned variable name
            imported_names: list[str] = []
            parent = call_node.parent
            if parent is not None and parent.type == "variable_declarator":
                name_node = parent.child_by_field_name("name")
                if name_node is not None:
                    name_text = _text(name_node)
                    if name_text:
                        imported_names.append(name_text)

            result.raw_imports.append(RawImport(
                importer_file=rel_path,
                module_path=module_path,
                imported_names=imported_names,
                is_from_import=False,
            ))
            break


# ── Call extraction ───────────────────────────────────────────────────


def _extract_calls_from_body(body_node, caller_id: str, result: ParseResult):
    """Walk a function/method body to find all call expressions."""
    for child in body_node.children:
        _walk_for_calls(child, caller_id, result, error_context=None)


def _walk_for_calls(
    ts_node, caller_id: str, result: ParseResult,
    error_context: dict | None,
):
    """Recursively walk to find call expressions, tracking error context."""
    node_type = ts_node.type

    # Don't descend into nested function/class definitions
    if node_type in ("function_declaration", "class_declaration",
                     "abstract_class_declaration", "generator_function_declaration",
                     "arrow_function", "function_expression",
                     "generator_function"):
        return

    # Handle try_statement: process with error context
    if node_type == "try_statement":
        _process_try_statement(ts_node, caller_id, result)
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

    # Handle new_expression (constructor calls)
    if node_type == "new_expression":
        _record_new_call(ts_node, caller_id, result, error_context)
        # Walk arguments for nested calls
        args = ts_node.child_by_field_name("arguments")
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
    handler = ts_node.child_by_field_name("handler")
    if handler is not None and handler.type == "catch_clause":
        param = handler.child_by_field_name("parameter")
        if param is not None:
            param_text = _text(param)
            if param_text:
                exception_types.append(param_text)

    error_context: dict | None
    if exception_types:
        error_context = {
            "pattern": "try_catch",
            "exception": ",".join(exception_types),
        }
    else:
        error_context = {
            "pattern": "try_catch",
            "exception": "Error",
        }

    # Walk the try body with error context
    body = ts_node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            _walk_for_calls(child, caller_id, result, error_context)

    # Walk handler body (catch block) without error context
    if handler is not None:
        handler_body = handler.child_by_field_name("body")
        if handler_body is not None:
            for child in handler_body.children:
                _walk_for_calls(child, caller_id, result, error_context=None)

    # Walk finalizer body
    finalizer = ts_node.child_by_field_name("finalizer")
    if finalizer is not None:
        # Finalizer is a finally_clause whose body is a statement_block
        for child in finalizer.children:
            if child.type == "statement_block":
                for sub in child.children:
                    _walk_for_calls(sub, caller_id, result, error_context=None)
            else:
                _walk_for_calls(child, caller_id, result, error_context=None)


def _record_call(call_node, caller_id: str, result: ParseResult,
                 error_context: dict | None):
    """Record a single call_expression as a RawCall."""
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        if call_node.children:
            func_node = call_node.children[0]
        else:
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


def _record_new_call(new_node, caller_id: str, result: ParseResult,
                     error_context: dict | None):
    """Record a new_expression as a RawCall (constructor invocation)."""
    # The constructor name is the first meaningful child after 'new'
    constructor = new_node.child_by_field_name("constructor")
    if constructor is None:
        # Fallback: iterate children for identifier/member_expression
        for child in new_node.children:
            if child.type in ("identifier", "member_expression"):
                constructor = child
                break
    if constructor is None:
        return

    callee_text = _text(constructor)
    line = _line(new_node)
    return_usage = _determine_return_usage(new_node)

    result.raw_calls.append(RawCall(
        caller_id=caller_id,
        callee_text=f"new {callee_text}",
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

    # Unwrap parenthesized_expression — JS wraps if/while conditions in parens
    if parent_type == "parenthesized_expression":
        grandparent = parent.parent
        if grandparent is not None:
            gp_type = grandparent.type
            if gp_type in ("if_statement", "while_statement", "do_statement",
                            "switch_statement", "for_statement"):
                return "checked"
            # Otherwise treat as whatever the grandparent implies
            parent = grandparent
            parent_type = parent.type

    # await wrapping — check the await's parent for actual usage
    if parent_type == "await_expression":
        grandparent = parent.parent
        if grandparent is None:
            return "awaited_ignored"

        gp_type = grandparent.type
        if gp_type in ("variable_declarator", "assignment_expression",
                        "augmented_assignment_expression"):
            return "awaited"
        if gp_type == "return_statement":
            return "awaited"
        if gp_type == "expression_statement":
            return "awaited_ignored"
        if gp_type in ("if_statement", "while_statement", "do_statement",
                        "ternary_expression", "binary_expression",
                        "unary_expression"):
            return "awaited"
        return "awaited"

    # Assignment
    if parent_type in ("variable_declarator", "assignment_expression",
                        "augmented_assignment_expression"):
        return "assigned"

    # Return
    if parent_type == "return_statement":
        return "returned"

    # Condition of if/while
    if parent_type in ("if_statement", "while_statement", "do_statement",
                        "switch_statement"):
        return "checked"

    # Ternary / binary / unary
    if parent_type in ("ternary_expression", "binary_expression",
                        "unary_expression"):
        return "checked"

    # Bare expression statement
    if parent_type == "expression_statement":
        return "ignored"

    # As argument to another call
    if parent_type == "arguments":
        return "assigned"

    # Yield
    if parent_type == "yield_expression":
        return "returned"

    # Part of array/object
    if parent_type in ("array", "object", "pair", "spread_element"):
        return "assigned"

    # Member access: foo().bar
    if parent_type == "member_expression":
        return "assigned"

    # Subscript: foo()[0]
    if parent_type == "subscript_expression":
        return "assigned"

    return "ignored"


# ── Signature building ────────────────────────────────────────────────


def _build_function_signature(ts_node, func_name: str) -> str:
    """Build signature for a function_declaration: name(params): RetType."""
    params_node = ts_node.child_by_field_name("parameters")
    params_text = _text(params_node) if params_node else "()"
    params_inner = _strip_parens(params_text)

    return_type = ts_node.child_by_field_name("return_type")
    ret_suffix = ""
    if return_type is not None:
        ret_text = _text(return_type)
        ret_text = ret_text.lstrip(": ")
        if ret_text:
            ret_suffix = f": {ret_text}"

    return f"{func_name}({params_inner}){ret_suffix}"


def _build_method_signature(ts_node, method_name: str) -> str:
    """Build signature for a method_definition: name(params): RetType."""
    params_node = ts_node.child_by_field_name("parameters")
    params_text = _text(params_node) if params_node else "()"
    params_inner = _strip_parens(params_text)

    return_type = ts_node.child_by_field_name("return_type")
    ret_suffix = ""
    if return_type is not None:
        ret_text = _text(return_type)
        ret_text = ret_text.lstrip(": ")
        if ret_text:
            ret_suffix = f": {ret_text}"

    return f"{method_name}({params_inner}){ret_suffix}"


# ── Tag computation ───────────────────────────────────────────────────


def _compute_class_tags_js(
    class_name: str, base_classes: list[str], implements: list[str],
    is_exported: bool,
    is_abstract: bool,
    is_test_file: bool,
) -> list[str]:
    """Compute semantic tags for a JS/TS class."""
    tags: list[str] = []

    if is_exported:
        tags.append("export")

    if is_abstract:
        tags.append("abstract")

    # extends tag
    if base_classes:
        tags.append(f"extends:{','.join(base_classes)}")

    # data_model — name ends with Model/Schema/Entity/DTO or extends known base
    _name_lower = class_name.lower()
    if any(_name_lower.endswith(suffix) for suffix in ("model", "schema", "entity", "dto")):
        tags.append("data_model")
    else:
        for base in base_classes:
            base_simple = base.split("<")[0].split(".")[-1]
            if base_simple in DATA_MODEL_BASES:
                tags.append("data_model")
                break

    # test
    if class_name.startswith("Test") or is_test_file:
        if "test" not in tags:
            pass  # file already tagged; don't double-tag class unless it is a Test class
    if class_name.startswith("Test"):
        tags.append("test")

    # private
    if class_name.startswith("_"):
        tags.append("private")

    return tags


def _compute_function_tags_js(
    func_name: str, is_async: bool, is_exported: bool,
    is_test_file: bool,
    is_entrypoint_file: bool,
) -> list[str]:
    """Compute semantic tags for a JS/TS function."""
    tags: list[str] = []

    if is_exported:
        tags.append("export")

    if is_async:
        tags.append("async")

    # entrypoint — only the main function, not every function in an entrypoint file
    if func_name == "main":
        tags.append("entrypoint")

    # test
    if func_name.startswith("test") or is_test_file:
        tags.append("test")

    # callback: name starts with 'on' + uppercase, or 'handle' + uppercase
    if (len(func_name) > 2 and func_name.startswith("on") and func_name[2].isupper()):
        tags.append("callback")
    elif (len(func_name) > 6 and func_name.startswith("handle") and func_name[6].isupper()):
        tags.append("callback")

    # factory
    for prefix in ("create", "build", "make"):
        if func_name.startswith(prefix) and (
            len(func_name) == len(prefix) or func_name[len(prefix)].isupper()
            or func_name[len(prefix)] == "_"
        ):
            tags.append("factory")
            break

    # private
    if func_name.startswith("_") or func_name.startswith("#"):
        tags.append("private")

    # route_handler — simple heuristic: function name matches route keywords
    if func_name.lower() in ROUTE_DECORATOR_KEYWORDS:
        tags.append("route_handler")

    return tags


def _compute_method_tags(
    method_name: str, is_async: bool, is_static: bool, is_abstract: bool,
    accessibility: str | None,
    is_test_file: bool,
    is_entrypoint_file: bool,
) -> list[str]:
    """Compute semantic tags for a method."""
    tags = _compute_function_tags_js(
        method_name, is_async, is_exported=False,
        is_test_file=is_test_file,
        is_entrypoint_file=is_entrypoint_file,
    )

    if is_static:
        tags.append("static")

    if is_abstract:
        tags.append("abstract")

    if accessibility == "private" and "private" not in tags:
        tags.append("private")
    elif accessibility == "protected":
        tags.append("protected")

    return tags


# ── JSDoc extraction ──────────────────────────────────────────────────


def _get_jsdoc_for_node(ts_node) -> str | None:
    """Get the JSDoc comment (/** ... */) immediately preceding a node.

    Tree-sitter places comment nodes as siblings in the parent's children list.
    """
    parent = ts_node.parent
    if parent is None:
        return None

    prev_sibling = ts_node.prev_named_sibling
    if prev_sibling is None:
        # Also check unnamed siblings (comments may be unnamed)
        prev_sibling = ts_node.prev_sibling

    if prev_sibling is not None and prev_sibling.type == "comment":
        text = _text(prev_sibling)
        if text.startswith("/**"):
            return _parse_jsdoc(text)

    return None


def _extract_leading_jsdoc(root, child_index: int) -> str | None:
    """Extract a leading JSDoc from the first child of the root (for file-level docs)."""
    if not root.children:
        return None

    for child in root.children:
        if child.type == "comment":
            text = _text(child)
            if text.startswith("/**"):
                return _parse_jsdoc(text)
            # Skip line comments at the top (e.g., shebangs, eslint directives)
            continue
        # First non-comment — stop looking
        break
    return None


def _parse_jsdoc(raw: str) -> str:
    """Parse a JSDoc comment string, stripping delimiters and leading asterisks."""
    # Remove /** and */
    content = raw
    if content.startswith("/**"):
        content = content[3:]
    if content.endswith("*/"):
        content = content[:-2]

    # Remove leading * on each line
    lines = content.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("* "):
            cleaned.append(stripped[2:])
        elif stripped.startswith("*"):
            cleaned.append(stripped[1:])
        else:
            cleaned.append(stripped)

    result = "\n".join(cleaned).strip()
    return result if result else None


# ── Utility helpers ───────────────────────────────────────────────────


def _strip_quotes(s: str) -> str:
    """Remove surrounding quotes from a string literal."""
    if len(s) >= 2:
        if (s.startswith('"') and s.endswith('"')) or \
           (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        if s.startswith("`") and s.endswith("`"):
            return s[1:-1]
    return s


def _strip_parens(s: str) -> str:
    """Strip outer parentheses from a parameter list string."""
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        return s[1:-1].strip()
    return s


def _has_keyword_child(ts_node, keyword: str) -> bool:
    """Check if a tree-sitter node has a direct child that is a keyword token."""
    for child in ts_node.children:
        if child.type == keyword:
            return True
        # Some grammars use a generic token — check text as fallback
        if not child.is_named and _text(child) == keyword:
            return True
    return False


def _get_accessibility(ts_node) -> str | None:
    """Get TypeScript accessibility modifier (public/private/protected)."""
    for child in ts_node.children:
        if child.type == "accessibility_modifier":
            return _text(child)
    return None
