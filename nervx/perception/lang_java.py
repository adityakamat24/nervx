"""Tree-sitter AST parser for Java source files (.java)."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from nervx.perception.parser import (
    Node, RawCall, RawImport, ParseResult,
    _relative_path, _text, _line, _end_line,
)

JAVA_LANGUAGE = Language(tsjava.language())

# ── Node types ──────────────────────────────────────────────────────

_CLASS_TYPES = frozenset({
    "class_declaration", "interface_declaration", "enum_declaration",
    "record_declaration", "annotation_type_declaration",
})

_METHOD_TYPES = frozenset({
    "method_declaration", "constructor_declaration",
})

_DATA_MODEL_ANNOTATIONS = frozenset({
    "Entity", "Data", "Value", "Table", "Immutable",
    "Document", "Embeddable", "MappedSuperclass",
})

_FACTORY_PREFIXES = ("create", "build", "make", "newInstance", "of", "from")

# ── Public entry point ───────────────────────────────────────────────


def parse_java(file_path: str, repo_root: str) -> ParseResult:
    """Parse a Java source file and extract structural information."""
    rel_path = _relative_path(file_path, repo_root)
    file_name = Path(file_path).name

    def _empty_result(doc: str | None = None, line_end: int = 1) -> ParseResult:
        file_node = Node(
            id=rel_path, kind="file", name=file_name,
            file_path=rel_path, line_start=1, line_end=line_end,
            signature="", docstring=doc, tags=[], parent_id="",
        )
        return ParseResult(file_path=rel_path, nodes=[file_node])

    try:
        source_bytes = Path(file_path).read_bytes()
    except (OSError, IOError):
        return _empty_result()

    try:
        parser = Parser(JAVA_LANGUAGE)
        tree = parser.parse(source_bytes)
    except Exception:
        return _empty_result()

    root = tree.root_node
    if root is None:
        return _empty_result()

    source_text = source_bytes.decode("utf8", errors="replace")
    total_lines = source_text.count("\n") + (
        1 if source_text and not source_text.endswith("\n") else 0
    )
    if total_lines == 0:
        total_lines = 1

    file_node = Node(
        id=rel_path, kind="file", name=file_name,
        file_path=rel_path, line_start=1, line_end=total_lines,
        signature="", docstring=None, tags=[], parent_id="",
    )
    result = ParseResult(file_path=rel_path, nodes=[file_node])

    for child in root.children:
        _process_top_level(child, rel_path, file_node.id, result)

    # Tag file as entrypoint if any method is an entrypoint
    for node in result.nodes:
        if "entrypoint" in node.tags:
            file_node.tags.append("entrypoint")
            break

    # Build error_handling summary dict
    for rc in result.raw_calls:
        if rc.error_handling is not None:
            eh_dict = result.error_handling.setdefault(rc.caller_id, {})
            eh_dict[rc.callee_text] = rc.error_handling

    return result


# ── Top-level processing ─────────────────────────────────────────────


def _process_top_level(ts_node, rel_path: str, parent_id: str,
                       result: ParseResult):
    """Process a top-level tree-sitter node in a Java file."""
    if ts_node.type in _CLASS_TYPES:
        _process_class(ts_node, rel_path, parent_id, result,
                       enclosing_class=None)
    elif ts_node.type == "import_declaration":
        _process_import(ts_node, rel_path, result)


# ── Class / interface / enum / record processing ────────────────────


def _process_class(ts_node, rel_path: str, parent_id: str,
                   result: ParseResult, enclosing_class: str | None):
    """Process any class-like declaration (class, interface, enum, record, annotation type)."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    class_name = _text(name_node)
    node_type = ts_node.type

    if enclosing_class:
        node_id = f"{rel_path}::{enclosing_class}.{class_name}"
        full_name = f"{enclosing_class}.{class_name}"
    else:
        node_id = f"{rel_path}::{class_name}"
        full_name = class_name

    modifiers = _get_modifiers(ts_node)
    annotations = _get_annotations(ts_node)

    # Superclass and interfaces
    superclass = _extract_superclass(ts_node)
    interfaces = _extract_super_interfaces(ts_node)
    extends_ifaces = _extract_extends_interfaces(ts_node)

    # Signature
    signature = _build_class_signature(class_name, node_type,
                                       superclass, interfaces,
                                       extends_ifaces)

    # Tags
    tags = _compute_class_tags(class_name, node_type, superclass,
                               interfaces, extends_ifaces, modifiers,
                               annotations)

    # Docstring
    docstring = _get_javadoc(ts_node)

    class_node = Node(
        id=node_id, kind="class", name=class_name,
        file_path=rel_path, line_start=_line(ts_node),
        line_end=_end_line(ts_node), signature=signature,
        docstring=docstring, tags=tags, parent_id=parent_id,
    )
    result.nodes.append(class_node)

    # Process body for methods and nested types
    body = ts_node.child_by_field_name("body")
    if body is None:
        # Some types use class_body, interface_body, enum_body, etc.
        for child in ts_node.children:
            if child.type.endswith("_body"):
                body = child
                break
    if body is not None:
        for child in body.children:
            if child.type in _METHOD_TYPES:
                _process_method(child, rel_path, node_id, full_name,
                                result)
            elif child.type in _CLASS_TYPES:
                _process_class(child, rel_path, node_id, result,
                               enclosing_class=full_name)


# ── Method / constructor processing ─────────────────────────────────


def _process_method(ts_node, rel_path: str, class_id: str,
                    class_name: str, result: ParseResult):
    """Process a method_declaration or constructor_declaration."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    method_name = _text(name_node)
    node_id = f"{rel_path}::{class_name}.{method_name}"
    is_constructor = ts_node.type == "constructor_declaration"

    modifiers = _get_modifiers(ts_node)
    annotations = _get_annotations(ts_node)

    # Build signature
    params_text = _extract_params(ts_node)
    if is_constructor:
        signature = f"{method_name}({params_text})"
    else:
        type_node = ts_node.child_by_field_name("type")
        return_type = _text(type_node).strip() if type_node else "void"
        signature = f"{return_type} {method_name}({params_text})"

    tags = _compute_method_tags(method_name, modifiers, annotations)
    docstring = _get_javadoc(ts_node)

    method_node = Node(
        id=node_id, kind="method", name=method_name,
        file_path=rel_path, line_start=_line(ts_node),
        line_end=_end_line(ts_node), signature=signature,
        docstring=docstring, tags=tags, parent_id=class_id,
    )
    result.nodes.append(method_node)

    body = ts_node.child_by_field_name("body")
    if body is not None:
        _extract_calls_from_body(body, node_id, result)


# ── Signature building ──────────────────────────────────────────────


def _build_class_signature(class_name: str, node_type: str,
                           superclass: str | None,
                           interfaces: list[str],
                           extends_ifaces: list[str]) -> str:
    """Build a class signature string."""
    keyword_map = {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "record_declaration": "record",
        "annotation_type_declaration": "@interface",
    }
    keyword = keyword_map.get(node_type, "class")
    parts = [f"{keyword} {class_name}"]

    if superclass:
        parts.append(f"extends {superclass}")
    if extends_ifaces:
        parts.append(f"extends {', '.join(extends_ifaces)}")
    if interfaces:
        parts.append(f"implements {', '.join(interfaces)}")

    return " ".join(parts)


def _extract_params(ts_node) -> str:
    """Extract parameter list text without outer parens."""
    params_node = ts_node.child_by_field_name("parameters")
    if params_node is None:
        return ""
    text = _text(params_node).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text


# ── Import processing ───────────────────────────────────────────────


def _process_import(ts_node, rel_path: str, result: ParseResult):
    """Process an import_declaration.

    import com.example.Foo;              -> module_path="com.example.Foo"
    import static com.example.Foo.bar;   -> module_path="com.example.Foo", imported_names=["bar"]
    import com.example.*;                -> module_path="com.example", imported_names=["*"]
    """
    is_static = any(c.type == "static" for c in ts_node.children)
    has_asterisk = any(c.type == "asterisk" for c in ts_node.children)

    # Find the scoped_identifier or identifier
    import_path = ""
    for child in ts_node.children:
        if child.type in ("scoped_identifier", "identifier"):
            import_path = _text(child)
            break

    if not import_path:
        return

    if has_asterisk:
        # import java.util.* -> module_path="java.util", names=["*"]
        result.raw_imports.append(RawImport(
            importer_file=rel_path,
            module_path=import_path,
            imported_names=["*"],
            is_from_import=False,
        ))
    elif is_static and "." in import_path:
        # import static com.example.Foo.bar -> module="com.example.Foo", names=["bar"]
        last_dot = import_path.rfind(".")
        module_path = import_path[:last_dot]
        member = import_path[last_dot + 1:]
        result.raw_imports.append(RawImport(
            importer_file=rel_path,
            module_path=module_path,
            imported_names=[member],
            is_from_import=False,
        ))
    else:
        # import com.example.Foo -> module_path="com.example.Foo"
        result.raw_imports.append(RawImport(
            importer_file=rel_path,
            module_path=import_path,
            imported_names=[],
            is_from_import=False,
        ))


# ── Call extraction ─────────────────────────────────────────────────


def _extract_calls_from_body(body_node, caller_id: str,
                             result: ParseResult):
    """Walk a method/constructor body to find all call expressions."""
    for child in body_node.children:
        _walk_for_calls(child, caller_id, result, error_context=None)


def _walk_for_calls(ts_node, caller_id: str, result: ParseResult,
                    error_context: dict | None):
    """Recursively walk to find calls, tracking error context."""
    node_type = ts_node.type

    # Don't descend into nested classes or lambdas
    if node_type in _CLASS_TYPES or node_type == "lambda_expression":
        return

    # Try/catch: track error context
    if node_type == "try_statement":
        _process_try_statement(ts_node, caller_id, result)
        return

    # method_invocation: obj.method(args) or method(args)
    if node_type == "method_invocation":
        _record_method_call(ts_node, caller_id, result, error_context)
        # Walk arguments for nested calls
        args = ts_node.child_by_field_name("arguments")
        if args is not None:
            for child in args.children:
                _walk_for_calls(child, caller_id, result, error_context)
        # Walk object for chained calls
        obj = ts_node.child_by_field_name("object")
        if obj is not None and obj.type in (
            "method_invocation", "object_creation_expression"
        ):
            _walk_for_calls(obj, caller_id, result, error_context)
        return

    # object_creation_expression: new Foo(args)
    if node_type == "object_creation_expression":
        _record_new_call(ts_node, caller_id, result, error_context)
        args = ts_node.child_by_field_name("arguments")
        if args is not None:
            for child in args.children:
                _walk_for_calls(child, caller_id, result, error_context)
        return

    for child in ts_node.children:
        _walk_for_calls(child, caller_id, result, error_context)


# ── Try / catch error handling ──────────────────────────────────────


def _process_try_statement(ts_node, caller_id: str,
                           result: ParseResult):
    """Process a try_statement with catch clauses for error context."""
    exception_types: list[str] = []
    for child in ts_node.children:
        if child.type == "catch_clause":
            exc = _extract_catch_type(child)
            if exc:
                exception_types.append(exc)

    error_context: dict | None = {
        "pattern": "try_catch",
        "exception": ",".join(exception_types) if exception_types else "Exception",
    }

    # Walk the try block body with error context
    for child in ts_node.children:
        if child.type == "block":
            for block_child in child.children:
                _walk_for_calls(block_child, caller_id, result, error_context)
            break  # First block is the try body

    # Walk catch/finally blocks without error context
    for child in ts_node.children:
        if child.type == "catch_clause":
            for sub in child.children:
                if sub.type == "block":
                    for bc in sub.children:
                        _walk_for_calls(bc, caller_id, result, None)
        elif child.type == "finally_clause":
            for sub in child.children:
                if sub.type == "block":
                    for bc in sub.children:
                        _walk_for_calls(bc, caller_id, result, None)

    # Try-with-resources
    resources = ts_node.child_by_field_name("resources")
    if resources is not None:
        for child in resources.children:
            _walk_for_calls(child, caller_id, result, error_context)


def _extract_catch_type(catch_node) -> str | None:
    """Extract exception type(s) from a catch_clause.

    Handles single types and multi-catch: catch (IOException | SQLException e).
    """
    for child in catch_node.children:
        if child.type == "catch_formal_parameter":
            for pc in child.children:
                if pc.type == "catch_type":
                    types = [_text(t) for t in pc.children
                             if t.type in ("type_identifier",
                                           "scoped_type_identifier")]
                    return ",".join(types) if types else None
    return None


# ── Call recording ──────────────────────────────────────────────────


def _record_method_call(call_node, caller_id: str,
                        result: ParseResult,
                        error_context: dict | None):
    """Record a method_invocation as a RawCall."""
    name_node = call_node.child_by_field_name("name")
    if name_node is None:
        return
    method_name = _text(name_node)
    obj_node = call_node.child_by_field_name("object")

    if obj_node is not None:
        callee_text = f"{_text(obj_node)}.{method_name}"
    else:
        callee_text = method_name

    result.raw_calls.append(RawCall(
        caller_id=caller_id,
        callee_text=callee_text,
        line=_line(call_node),
        error_handling=error_context,
        return_usage=_determine_return_usage(call_node),
    ))


def _record_new_call(new_node, caller_id: str, result: ParseResult,
                     error_context: dict | None):
    """Record an object_creation_expression (new Foo()) as a RawCall.

    Callee text is the type name (e.g., 'Foo'), not 'new Foo'.
    """
    type_node = new_node.child_by_field_name("type")
    if type_node is None:
        return

    result.raw_calls.append(RawCall(
        caller_id=caller_id,
        callee_text=_text(type_node),
        line=_line(new_node),
        error_handling=error_context,
        return_usage=_determine_return_usage(new_node),
    ))


# ── Return usage determination ──────────────────────────────────────


def _determine_return_usage(call_node) -> str:
    """Determine how the return value of a call is used."""
    parent = call_node.parent
    if parent is None:
        return "ignored"

    pt = parent.type

    # Parenthesized expression (conditions in if/while)
    if pt == "parenthesized_expression":
        gp = parent.parent
        if gp is not None and gp.type in (
            "if_statement", "while_statement", "do_statement",
            "for_statement",
        ):
            return "checked"
        parent = gp if gp is not None else parent
        pt = parent.type

    if pt in ("variable_declarator", "assignment_expression"):
        return "assigned"
    if pt == "return_statement":
        return "returned"
    if pt == "expression_statement":
        return "ignored"
    if pt in ("if_statement", "while_statement", "do_statement"):
        return "checked"
    if pt in ("binary_expression", "ternary_expression",
              "unary_expression"):
        return "checked"
    if pt == "argument_list":
        return "assigned"
    if pt in ("method_invocation", "field_access", "array_access"):
        return "assigned"
    if pt in ("throw_statement", "yield_statement"):
        return "ignored"

    return "ignored"


# ── Superclass / interface extraction ───────────────────────────────


def _extract_superclass(ts_node) -> str | None:
    """Extract superclass name from a class declaration."""
    for child in ts_node.children:
        if child.type == "superclass":
            for sub in child.children:
                if sub.type in ("type_identifier", "generic_type",
                                "scoped_type_identifier"):
                    return _text(sub)
    return None


def _extract_super_interfaces(ts_node) -> list[str]:
    """Extract implemented interface names (for classes/enums)."""
    for child in ts_node.children:
        if child.type == "super_interfaces":
            return _collect_type_list(child)
    return []


def _extract_extends_interfaces(ts_node) -> list[str]:
    """Extract extended interface names (for interfaces)."""
    for child in ts_node.children:
        if child.type == "extends_interfaces":
            return _collect_type_list(child)
    return []


def _collect_type_list(parent_node) -> list[str]:
    """Collect type names from a type_list child."""
    for child in parent_node.children:
        if child.type == "type_list":
            return [_text(t) for t in child.children
                    if t.type in ("type_identifier", "generic_type",
                                  "scoped_type_identifier")]
    return []


# ── Modifier / annotation helpers ───────────────────────────────────


def _get_modifiers(ts_node) -> set[str]:
    """Extract modifier keywords from a node's modifiers child."""
    mods: set[str] = set()
    for child in ts_node.children:
        if child.type == "modifiers":
            for sub in child.children:
                t = _text(sub).strip()
                if t in ("public", "private", "protected", "static",
                         "abstract", "final", "synchronized", "native",
                         "default", "strictfp", "transient", "volatile"):
                    mods.add(t)
            break
    return mods


def _get_annotations(ts_node) -> list[str]:
    """Extract annotation names (without @) from a node's modifiers child."""
    annotations: list[str] = []
    for child in ts_node.children:
        if child.type == "modifiers":
            for sub in child.children:
                if sub.type == "marker_annotation":
                    ann = _text(sub)
                    annotations.append(ann[1:] if ann.startswith("@") else ann)
                elif sub.type == "annotation":
                    name_node = sub.child_by_field_name("name")
                    if name_node is not None:
                        annotations.append(_text(name_node))
            break
    return annotations


# ── Tag computation ─────────────────────────────────────────────────


def _compute_class_tags(name: str, node_type: str,
                        superclass: str | None,
                        interfaces: list[str],
                        extends_ifaces: list[str],
                        modifiers: set[str],
                        annotations: list[str]) -> list[str]:
    """Compute semantic tags for a class-like declaration."""
    tags: list[str] = []

    if node_type == "interface_declaration":
        tags.append("interface")

    if "abstract" in modifiers:
        tags.append("abstract")
    if "private" in modifiers:
        tags.append("private")
    if "static" in modifiers:
        tags.append("static")

    # extends tags
    if superclass:
        tags.append(f"extends:{superclass}")
    for iface in interfaces:
        tags.append(f"extends:{iface}")
    for iface in extends_ifaces:
        tags.append(f"extends:{iface}")

    # data_model
    if node_type == "record_declaration":
        tags.append("data_model")
    else:
        for ann in annotations:
            simple = ann.rsplit(".", 1)[-1]
            if simple in _DATA_MODEL_ANNOTATIONS:
                tags.append("data_model")
                break

    # test class
    if name.startswith("Test"):
        tags.append("test")

    return tags


def _compute_method_tags(method_name: str, modifiers: set[str],
                         annotations: list[str]) -> list[str]:
    """Compute semantic tags for a method or constructor."""
    tags: list[str] = []

    # entrypoint: public static void main
    if (method_name == "main" and "public" in modifiers
            and "static" in modifiers):
        tags.append("entrypoint")

    # test
    test_anns = {"Test", "ParameterizedTest", "RepeatedTest"}
    if any(a in test_anns for a in annotations):
        tags.append("test")

    # callback
    if (method_name.startswith("on") and len(method_name) > 2
            and method_name[2].isupper()):
        tags.append("callback")
    elif (method_name.startswith("handle") and len(method_name) > 6
          and method_name[6].isupper()):
        tags.append("callback")

    # factory
    for prefix in _FACTORY_PREFIXES:
        if method_name == prefix or (
            method_name.startswith(prefix)
            and len(method_name) > len(prefix)
            and (method_name[len(prefix)].isupper()
                 or method_name[len(prefix)] == "_")
        ):
            tags.append("factory")
            break

    # modifiers
    if "private" in modifiers:
        tags.append("private")
    if "static" in modifiers:
        tags.append("static")
    if "abstract" in modifiers:
        tags.append("abstract")

    return tags


# ── Javadoc extraction ──────────────────────────────────────────────


def _get_javadoc(ts_node) -> str | None:
    """Get the Javadoc (/** ... */) preceding a declaration node."""
    # Check previous named sibling first
    prev = ts_node.prev_named_sibling
    if prev is not None and prev.type == "block_comment":
        text = _text(prev)
        if text.startswith("/**"):
            return _parse_javadoc(text)

    # Fall back to prev_sibling (unnamed)
    prev = ts_node.prev_sibling
    if prev is not None and prev.type == "block_comment":
        text = _text(prev)
        if text.startswith("/**"):
            return _parse_javadoc(text)

    return None


def _parse_javadoc(raw: str) -> str | None:
    """Parse a Javadoc comment, stripping delimiters and leading asterisks."""
    content = raw
    if content.startswith("/**"):
        content = content[3:]
    if content.endswith("*/"):
        content = content[:-2]

    lines: list[str] = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("* "):
            lines.append(stripped[2:])
        elif stripped.startswith("*"):
            lines.append(stripped[1:])
        else:
            lines.append(stripped)

    result = "\n".join(lines).strip()
    return result if result else None
