"""Tree-sitter AST parser for C and C++ source files (.c, .h, .cpp, .hpp, .cc, .cxx, .hh)."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

from nervx.perception.parser import (
    Node, RawCall, RawImport, ParseResult,
    _relative_path, _text, _line, _end_line,
)

# ── Language objects ────────────────────────────────────────────────

C_LANGUAGE = Language(tsc.language())
CPP_LANGUAGE = Language(tscpp.language())

# ── Name-based tag prefixes ─────────────────────────────────────────

_FACTORY_PREFIXES = ("create_", "make_", "new_", "alloc_")
_CALLBACK_PREFIXES = ("on_", "handle_")


# ── Public entry points ─────────────────────────────────────────────


def parse_c(file_path: str, repo_root: str) -> ParseResult:
    """Parse a C source file (.c, .h) and extract structural information."""
    return _parse_c_cpp(file_path, repo_root, C_LANGUAGE, is_cpp=False)


def parse_cpp(file_path: str, repo_root: str) -> ParseResult:
    """Parse a C++ source file (.cpp, .hpp, .cc, .cxx, .hh) and extract structural information."""
    return _parse_c_cpp(file_path, repo_root, CPP_LANGUAGE, is_cpp=True)


# ── Shared implementation ───────────────────────────────────────────


def _parse_c_cpp(file_path: str, repo_root: str, language: Language,
                 is_cpp: bool) -> ParseResult:
    """Core parser shared between C and C++."""
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

    # File-level docstring (leading /** ... */ comment)
    file_doc = _extract_leading_doc_comment(root)

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
        docstring=file_doc,
        tags=file_tags,
        parent_id="",
    )

    result = ParseResult(file_path=rel_path, nodes=[file_node])

    # Walk top-level children
    for child in root.children:
        _process_top_level(child, rel_path, file_node.id, result, is_cpp)

    # Build error_handling summary dict
    for rc in result.raw_calls:
        if rc.error_handling is not None:
            eh_dict = result.error_handling.setdefault(rc.caller_id, {})
            eh_dict[rc.callee_text] = rc.error_handling

    return result


# ── Top-level processing ────────────────────────────────────────────


def _process_top_level(ts_node, rel_path: str, parent_id: str,
                       result: ParseResult, is_cpp: bool):
    """Process a top-level tree-sitter node in a C/C++ file."""
    node_type = ts_node.type

    if node_type == "preproc_include":
        _process_include(ts_node, rel_path, result)
    elif node_type == "function_definition":
        _process_function(ts_node, rel_path, parent_id, result, is_cpp,
                          class_name=None)
    elif node_type == "declaration":
        # May contain struct/enum/class specifiers with bodies
        _process_declaration(ts_node, rel_path, parent_id, result, is_cpp)
    elif node_type == "struct_specifier":
        _process_struct(ts_node, rel_path, parent_id, result, is_cpp)
    elif node_type == "enum_specifier":
        _process_enum(ts_node, rel_path, parent_id, result)
    elif node_type == "class_specifier" and is_cpp:
        _process_class(ts_node, rel_path, parent_id, result)
    elif node_type == "namespace_definition" and is_cpp:
        _process_namespace(ts_node, rel_path, parent_id, result)
    elif node_type == "template_declaration" and is_cpp:
        _process_template(ts_node, rel_path, parent_id, result)
    elif node_type == "linkage_specification" and is_cpp:
        # extern "C" { ... }
        body = ts_node.child_by_field_name("body")
        if body is not None:
            for child in body.children:
                _process_top_level(child, rel_path, parent_id, result, is_cpp)


# ── Namespace processing (C++) ──────────────────────────────────────


def _process_namespace(ts_node, rel_path: str, parent_id: str,
                       result: ParseResult):
    """Process a C++ namespace definition and recurse into its body."""
    body = ts_node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            _process_top_level(child, rel_path, parent_id, result, is_cpp=True)


# ── Template processing (C++) ──────────────────────────────────────


def _process_template(ts_node, rel_path: str, parent_id: str,
                      result: ParseResult):
    """Process a C++ template declaration — unwrap to inner declaration."""
    for child in ts_node.children:
        child_type = child.type
        if child_type == "function_definition":
            _process_function(child, rel_path, parent_id, result,
                              is_cpp=True, class_name=None)
        elif child_type == "class_specifier":
            _process_class(child, rel_path, parent_id, result)
        elif child_type == "struct_specifier":
            _process_struct(child, rel_path, parent_id, result, is_cpp=True)
        elif child_type == "declaration":
            _process_declaration(child, rel_path, parent_id, result,
                                 is_cpp=True)


# ── Declaration processing ──────────────────────────────────────────


def _process_declaration(ts_node, rel_path: str, parent_id: str,
                         result: ParseResult, is_cpp: bool):
    """Process a declaration node which may contain struct/enum/class specifiers.

    Handles patterns like:
      typedef struct { ... } Name;
      struct Foo { ... } foo_instance;
    """
    for child in ts_node.children:
        if child.type == "struct_specifier":
            body = child.child_by_field_name("body")
            if body is not None:
                name = _get_struct_name(child, ts_node)
                if name:
                    _process_struct_with_name(child, name, rel_path,
                                              parent_id, result, is_cpp)
        elif child.type == "enum_specifier":
            _process_enum(child, rel_path, parent_id, result,
                          typedef_parent=ts_node)
        elif child.type == "class_specifier" and is_cpp:
            body = child.child_by_field_name("body")
            if body is not None:
                _process_class(child, rel_path, parent_id, result)


# ── Include processing ──────────────────────────────────────────────


def _process_include(ts_node, rel_path: str, result: ParseResult):
    """Process a #include preprocessor directive."""
    path_node = ts_node.child_by_field_name("path")
    if path_node is None:
        # Fallback: find system_lib_string or string_literal child
        for child in ts_node.children:
            if child.type in ("system_lib_string", "string_literal"):
                path_node = child
                break

    if path_node is None:
        return

    raw_path = _text(path_node)
    # Strip surrounding <> or ""
    module_path = raw_path.strip()
    if module_path.startswith("<") and module_path.endswith(">"):
        module_path = module_path[1:-1]
    elif module_path.startswith('"') and module_path.endswith('"'):
        module_path = module_path[1:-1]

    result.raw_imports.append(RawImport(
        importer_file=rel_path,
        module_path=module_path,
        imported_names=[],
        is_from_import=False,
    ))


# ── Struct processing ──────────────────────────────────────────────


def _get_struct_name(struct_node, parent_node=None) -> str:
    """Get the name of a struct from the name field, or from a typedef declarator."""
    name_node = struct_node.child_by_field_name("name")
    if name_node is not None:
        return _text(name_node)

    # If the struct has no name field, look for a typedef name in the parent declaration
    if parent_node is not None:
        for child in parent_node.children:
            if child.type == "type_identifier" and child is not struct_node:
                return _text(child)
    return ""


def _process_struct(ts_node, rel_path: str, parent_id: str,
                    result: ParseResult, is_cpp: bool):
    """Process a top-level struct_specifier."""
    body = ts_node.child_by_field_name("body")
    if body is None:
        # Forward declaration — skip
        return

    name = _get_struct_name(ts_node)
    if not name:
        # Anonymous struct at top level — skip
        return

    _process_struct_with_name(ts_node, name, rel_path, parent_id, result,
                              is_cpp)


def _process_struct_with_name(ts_node, name: str, rel_path: str,
                              parent_id: str, result: ParseResult,
                              is_cpp: bool):
    """Extract a struct with a known name."""
    node_id = f"{rel_path}::{name}"

    # Base classes (C++ structs can have base_class_clause)
    base_classes: list[str] = []
    if is_cpp:
        base_classes = _extract_base_classes(ts_node)

    # Signature
    if base_classes:
        signature = f"struct {name} : {', '.join(base_classes)}"
    else:
        signature = f"struct {name}"

    # Tags
    tags: list[str] = []
    for base in base_classes:
        tags.append(f"extends:{base}")

    # Check for abstract (C++ struct with pure virtual methods)
    if is_cpp:
        body = ts_node.child_by_field_name("body")
        if body is not None and _has_pure_virtual(body):
            tags.append("abstract")

    # Docstring
    docstring = _get_doc_comment_for_node(ts_node)

    struct_node = Node(
        id=node_id,
        kind="class",
        name=name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=signature,
        docstring=docstring,
        tags=tags,
        parent_id=parent_id,
    )
    result.nodes.append(struct_node)

    # Process body for methods (C++ structs can have methods)
    if is_cpp:
        body = ts_node.child_by_field_name("body")
        if body is not None:
            _process_class_body(body, rel_path, node_id, name, result,
                                default_access="public")


# ── Class processing (C++) ──────────────────────────────────────────


def _process_class(ts_node, rel_path: str, parent_id: str,
                   result: ParseResult):
    """Extract a C++ class_specifier and its methods."""
    name_node = ts_node.child_by_field_name("name")
    if name_node is None:
        return
    class_name = _text(name_node)
    node_id = f"{rel_path}::{class_name}"

    # Base classes
    base_classes = _extract_base_classes(ts_node)

    # Signature
    if base_classes:
        signature = f"class {class_name} : {', '.join(base_classes)}"
    else:
        signature = f"class {class_name}"

    # Tags
    tags: list[str] = []
    for base in base_classes:
        tags.append(f"extends:{base}")

    # Check for abstract (pure virtual methods)
    body = ts_node.child_by_field_name("body")
    if body is not None and _has_pure_virtual(body):
        tags.append("abstract")

    # Docstring
    docstring = _get_doc_comment_for_node(ts_node)

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
    if body is not None:
        _process_class_body(body, rel_path, node_id, class_name, result,
                            default_access="private")


def _extract_base_classes(ts_node) -> list[str]:
    """Extract base class names from a C++ base_class_clause."""
    bases: list[str] = []
    for child in ts_node.children:
        if child.type == "base_class_clause":
            for sub in child.children:
                if sub.type == "type_identifier":
                    bases.append(_text(sub))
                elif sub.type == "qualified_identifier":
                    bases.append(_text(sub))
                elif sub.type == "base_class_specifier":
                    # Contains access_specifier + type
                    for inner in sub.children:
                        if inner.type in ("type_identifier",
                                          "qualified_identifier",
                                          "template_type"):
                            bases.append(_text(inner))
            break
    return bases


def _has_pure_virtual(body_node) -> bool:
    """Check if a class body contains any pure virtual method (= 0)."""
    for child in body_node.children:
        if child.type in ("declaration", "field_declaration"):
            text = _text(child)
            if "= 0" in text and "virtual" in text:
                return True
    return False


# ── Enum processing ─────────────────────────────────────────────────


def _process_enum(ts_node, rel_path: str, parent_id: str,
                  result: ParseResult, typedef_parent=None):
    """Process an enum_specifier node."""
    name_node = ts_node.child_by_field_name("name")
    enum_name = _text(name_node) if name_node is not None else ""

    # Try typedef name if no direct name
    if not enum_name and typedef_parent is not None:
        for child in typedef_parent.children:
            if child.type == "type_identifier" and child is not ts_node:
                enum_name = _text(child)
                break

    if not enum_name:
        # Anonymous enum — skip
        return

    node_id = f"{rel_path}::{enum_name}"

    # Only emit if it has a body (definition, not forward decl)
    body = ts_node.child_by_field_name("body")
    if body is None:
        body = _find_child_by_type(ts_node, "enumerator_list")
    if body is None:
        return

    docstring = _get_doc_comment_for_node(ts_node)

    enum_node = Node(
        id=node_id,
        kind="class",
        name=enum_name,
        file_path=rel_path,
        line_start=_line(ts_node),
        line_end=_end_line(ts_node),
        signature=f"enum {enum_name}",
        docstring=docstring,
        tags=[],
        parent_id=parent_id,
    )
    result.nodes.append(enum_node)


# ── Class body processing ──────────────────────────────────────────


def _process_class_body(body_node, rel_path: str, class_id: str,
                        class_name: str, result: ParseResult,
                        default_access: str = "private"):
    """Process children of a class/struct body (field_declaration_list)."""
    current_access = default_access

    for child in body_node.children:
        child_type = child.type

        if child_type == "access_specifier":
            access_text = _text(child).rstrip(":").strip().lower()
            if access_text in ("public", "private", "protected"):
                current_access = access_text
            continue

        if child_type == "function_definition":
            _process_method(child, rel_path, class_id, class_name, result,
                            access=current_access)
        elif child_type == "declaration":
            # Check for nested struct/class/enum
            for sub in child.children:
                if sub.type == "class_specifier":
                    body = sub.child_by_field_name("body")
                    if body is not None:
                        _process_class(sub, rel_path, class_id, result)
                elif sub.type == "struct_specifier":
                    body = sub.child_by_field_name("body")
                    if body is not None:
                        name = _get_struct_name(sub, child)
                        if name:
                            _process_struct_with_name(sub, name, rel_path,
                                                      class_id, result,
                                                      is_cpp=True)
                elif sub.type == "enum_specifier":
                    _process_enum(sub, rel_path, class_id, result,
                                  typedef_parent=child)
        elif child_type == "template_declaration":
            for sub in child.children:
                if sub.type == "function_definition":
                    _process_method(sub, rel_path, class_id, class_name,
                                    result, access=current_access)
                elif sub.type == "class_specifier":
                    _process_class(sub, rel_path, class_id, result)
                elif sub.type == "struct_specifier":
                    name = _get_struct_name(sub)
                    if name:
                        _process_struct_with_name(sub, name, rel_path,
                                                  class_id, result,
                                                  is_cpp=True)
        elif child_type == "friend_declaration":
            pass  # Skip friend declarations


# ── Method processing (C++) ─────────────────────────────────────────


def _process_method(ts_node, rel_path: str, class_id: str,
                    class_name: str, result: ParseResult,
                    access: str = "private"):
    """Process a function_definition inside a C++ class body (method)."""
    func_name = _extract_function_name(ts_node)
    if not func_name:
        return

    node_id = f"{rel_path}::{class_name}::{func_name}"

    # Signature
    return_type = _extract_return_type(ts_node)
    params_text = _extract_params_text(ts_node)
    if return_type:
        signature = f"{return_type} {func_name}({params_text})"
    else:
        signature = f"{func_name}({params_text})"

    # Tags
    tags: list[str] = []

    # Access-based private tag
    if access in ("private", "protected"):
        tags.append("private")

    # Check for static/virtual/inline specifiers
    _add_specifier_tags(ts_node, tags)

    # Check for pure virtual (= 0)
    full_text = _text(ts_node)
    if "virtual" in full_text and "= 0" in full_text:
        if "abstract" not in tags:
            tags.append("abstract")

    # Name-based tags
    _add_name_based_tags(func_name, tags)

    # Docstring
    docstring = _get_doc_comment_for_node(ts_node)

    method_node = Node(
        id=node_id,
        kind="method",
        name=func_name,
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
        _extract_calls_from_body(body, node_id, result, is_cpp=True)


# ── Function processing ─────────────────────────────────────────────


def _process_function(ts_node, rel_path: str, parent_id: str,
                      result: ParseResult, is_cpp: bool,
                      class_name: str | None = None):
    """Process a function_definition at file scope."""
    func_name = _extract_function_name(ts_node)
    if not func_name:
        return

    # Check if this is a qualified method definition (e.g., ClassName::methodName)
    declarator = ts_node.child_by_field_name("declarator")
    qualified_class = _extract_qualified_class(declarator) if declarator else None

    if qualified_class and is_cpp:
        # Out-of-class method definition: void ClassName::method() { ... }
        short_name = _extract_short_name(declarator)
        if short_name:
            func_name = short_name

        node_id = f"{rel_path}::{qualified_class}::{func_name}"
        kind = "method"
        actual_parent_id = f"{rel_path}::{qualified_class}"
    elif class_name:
        node_id = f"{rel_path}::{class_name}::{func_name}"
        kind = "method"
        actual_parent_id = f"{rel_path}::{class_name}"
    else:
        node_id = f"{rel_path}::{func_name}"
        kind = "function"
        actual_parent_id = parent_id

    # Signature
    return_type = _extract_return_type(ts_node)
    params_text = _extract_params_text(ts_node)
    if return_type:
        signature = f"{return_type} {func_name}({params_text})"
    else:
        signature = f"{func_name}({params_text})"

    # Tags
    tags: list[str] = []
    is_static = _has_storage_class(ts_node, "static")
    _add_specifier_tags(ts_node, tags)
    _add_name_based_tags(func_name, tags)

    # entrypoint
    if func_name == "main":
        if "entrypoint" not in tags:
            tags.append("entrypoint")

    # static C functions are private (file-internal linkage)
    if is_static and kind == "function":
        if "private" not in tags:
            tags.append("private")

    # Docstring
    docstring = _get_doc_comment_for_node(ts_node)

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
        parent_id=actual_parent_id,
    )
    result.nodes.append(func_node)

    # Extract calls from the function body
    body = ts_node.child_by_field_name("body")
    if body is not None:
        _extract_calls_from_body(body, node_id, result, is_cpp=is_cpp)


# ── Function name extraction ───────────────────────────────────────


def _extract_function_name(ts_node) -> str:
    """Extract the function name from a function_definition.

    The name is nested inside the declarator tree. We dig through
    function_declarator -> declarator to find the actual identifier.
    """
    declarator = ts_node.child_by_field_name("declarator")
    if declarator is None:
        return ""
    return _dig_for_name(declarator)


def _dig_for_name(node) -> str:
    """Recursively dig into declarator nodes to find the function name."""
    node_type = node.type

    if node_type == "identifier":
        return _text(node)
    if node_type == "field_identifier":
        return _text(node)
    if node_type == "type_identifier":
        return _text(node)

    # qualified_identifier: ClassName::methodName
    if node_type == "qualified_identifier":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return _text(name_node)
        # Fallback: last identifier child
        for child in reversed(node.children):
            if child.type in ("identifier", "field_identifier",
                              "destructor_name", "operator_name",
                              "type_identifier"):
                return _text(child)
            if child.type == "template_function":
                return _dig_for_name(child)
        return _text(node)

    # destructor_name: ~ClassName
    if node_type == "destructor_name":
        return _text(node)

    # operator_name: operator+, operator==, etc.
    if node_type == "operator_name":
        return _text(node)

    # template_function: name<T>
    if node_type == "template_function":
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return _text(name_node)

    # function_declarator: has a declarator child (the name) and parameters
    if node_type == "function_declarator":
        inner = node.child_by_field_name("declarator")
        if inner is not None:
            return _dig_for_name(inner)

    # pointer_declarator: *funcName
    if node_type == "pointer_declarator":
        inner = node.child_by_field_name("declarator")
        if inner is not None:
            return _dig_for_name(inner)

    # reference_declarator: &funcName
    if node_type == "reference_declarator":
        inner = node.child_by_field_name("declarator")
        if inner is not None:
            return _dig_for_name(inner)

    # parenthesized_declarator: (funcName)
    if node_type == "parenthesized_declarator":
        for child in node.children:
            if child.is_named:
                found = _dig_for_name(child)
                if found:
                    return found

    # Fallback: try children
    for child in node.children:
        if child.type in ("function_declarator", "identifier",
                          "field_identifier", "qualified_identifier",
                          "destructor_name", "operator_name",
                          "template_function", "pointer_declarator",
                          "reference_declarator"):
            found = _dig_for_name(child)
            if found:
                return found

    return ""


def _extract_qualified_class(declarator_node) -> str | None:
    """Extract the class name from a qualified method definition.

    For `void ClassName::method()`, the declarator is a function_declarator
    whose declarator is a qualified_identifier with scope=ClassName.
    """
    if declarator_node is None:
        return None

    node = declarator_node
    while node is not None:
        if node.type == "qualified_identifier":
            scope = node.child_by_field_name("scope")
            if scope is not None:
                scope_text = _text(scope)
                return scope_text.rstrip(":")
            return None
        if node.type == "function_declarator":
            node = node.child_by_field_name("declarator")
        elif node.type in ("pointer_declarator", "reference_declarator"):
            node = node.child_by_field_name("declarator")
        else:
            break
    return None


def _extract_short_name(declarator_node) -> str | None:
    """Extract the short (unqualified) method name from a qualified declaration."""
    if declarator_node is None:
        return None

    node = declarator_node
    while node is not None:
        if node.type == "qualified_identifier":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                return _text(name_node)
            return None
        if node.type == "function_declarator":
            node = node.child_by_field_name("declarator")
        elif node.type in ("pointer_declarator", "reference_declarator"):
            node = node.child_by_field_name("declarator")
        else:
            break
    return None


# ── Signature helpers ───────────────────────────────────────────────


def _extract_return_type(func_node) -> str:
    """Extract the return type from a function_definition node."""
    type_node = func_node.child_by_field_name("type")
    if type_node is not None:
        return _text(type_node).strip()

    # Fallback: collect type specifier children before the declarator
    parts: list[str] = []
    for child in func_node.children:
        if child.type == "function_declarator":
            break
        if child.type in ("primitive_type", "type_identifier",
                          "sized_type_specifier", "type_qualifier",
                          "qualified_identifier", "template_type"):
            parts.append(_text(child))
    return " ".join(parts).strip()


def _extract_params_text(ts_node) -> str:
    """Extract the parameters text from a function_definition's declarator."""
    declarator = ts_node.child_by_field_name("declarator")
    if declarator is None:
        return ""

    func_decl = _find_function_declarator(declarator)
    if func_decl is None:
        return ""

    params = func_decl.child_by_field_name("parameters")
    if params is None:
        return ""

    params_raw = _text(params).strip()
    if params_raw.startswith("(") and params_raw.endswith(")"):
        return params_raw[1:-1].strip()
    return params_raw


def _find_function_declarator(node):
    """Find the function_declarator node within a declarator tree."""
    if node.type == "function_declarator":
        return node
    for child in node.children:
        if child.type == "function_declarator":
            return child
        found = _find_function_declarator(child)
        if found is not None:
            return found
    return None


def _has_storage_class(func_node, keyword: str) -> bool:
    """Check if a function_definition has a specific storage class specifier."""
    for child in func_node.children:
        if child.type == "storage_class_specifier" and _text(child) == keyword:
            return True
    return False


# ── Specifier / tag helpers ─────────────────────────────────────────


def _add_specifier_tags(ts_node, tags: list[str]):
    """Add tags based on storage class specifiers and type qualifiers."""
    for child in ts_node.children:
        child_type = child.type
        text = _text(child)

        if child_type == "storage_class_specifier":
            if text == "static" and "static" not in tags:
                tags.append("static")
        elif child_type in ("virtual_function_specifier", "virtual"):
            if "virtual" in text and "virtual" not in tags:
                tags.append("virtual")
        elif not child.is_named:
            if text == "virtual" and "virtual" not in tags:
                tags.append("virtual")
            elif text == "static" and "static" not in tags:
                tags.append("static")


def _add_name_based_tags(func_name: str, tags: list[str]):
    """Add semantic tags based on the function name."""
    lower = func_name.lower()

    # Factory
    for prefix in _FACTORY_PREFIXES:
        if lower.startswith(prefix):
            if "factory" not in tags:
                tags.append("factory")
            break

    # Callback
    for prefix in _CALLBACK_PREFIXES:
        if lower.startswith(prefix):
            if "callback" not in tags:
                tags.append("callback")
            break

    # Test: functions with "test" or "Test" in the name
    if "test" in lower:
        if "test" not in tags:
            tags.append("test")


# ── Call extraction ─────────────────────────────────────────────────


def _extract_calls_from_body(body_node, caller_id: str,
                             result: ParseResult, is_cpp: bool):
    """Walk a function/method body to find all call expressions."""
    for child in body_node.children:
        _walk_for_calls(child, caller_id, result, error_context=None,
                        is_cpp=is_cpp)


def _walk_for_calls(ts_node, caller_id: str, result: ParseResult,
                    error_context: dict | None, is_cpp: bool):
    """Recursively walk to find call_expression nodes, tracking error context."""
    node_type = ts_node.type

    # Don't descend into nested function/class/struct definitions
    if node_type in ("function_definition", "class_specifier",
                     "struct_specifier", "lambda_expression"):
        return

    # Handle C++ try/catch
    if node_type == "try_statement" and is_cpp:
        _process_try_statement(ts_node, caller_id, result)
        return

    # Handle call expressions
    if node_type == "call_expression":
        _record_call(ts_node, caller_id, result, error_context)
        # Walk arguments for nested calls
        args = ts_node.child_by_field_name("arguments")
        if args is not None:
            for child in args.children:
                _walk_for_calls(child, caller_id, result, error_context,
                                is_cpp)
        return

    # Recurse into children
    for child in ts_node.children:
        _walk_for_calls(child, caller_id, result, error_context, is_cpp)


def _process_try_statement(ts_node, caller_id: str, result: ParseResult):
    """Process a C++ try/catch statement, tracking exception types."""
    # Collect exception types from catch clauses
    exception_types: list[str] = []
    for child in ts_node.children:
        if child.type == "catch_clause":
            exc_type = _extract_catch_type(child)
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
            "exception": "...",
        }

    # Walk the try body with error context
    body = ts_node.child_by_field_name("body")
    if body is not None:
        for child in body.children:
            _walk_for_calls(child, caller_id, result,
                            error_context=error_context, is_cpp=True)

    # Walk catch handler bodies without error context
    for child in ts_node.children:
        if child.type == "catch_clause":
            catch_body = child.child_by_field_name("body")
            if catch_body is not None:
                for sub in catch_body.children:
                    _walk_for_calls(sub, caller_id, result,
                                    error_context=None, is_cpp=True)


def _extract_catch_type(catch_node) -> str | None:
    """Extract the exception type from a C++ catch clause.

    catch (const std::exception& e) -> 'std::exception'
    catch (...) -> '...'
    """
    params = catch_node.child_by_field_name("parameters")
    if params is None:
        # Look for parameter_list child
        for child in catch_node.children:
            if child.type in ("parameter_list", "parameter_declaration"):
                params = child
                break

    if params is None:
        # catch (...) — ellipsis may be a direct child
        text = _text(catch_node)
        if "..." in text:
            return "..."
        return None

    param_text = _text(params).strip()
    if param_text == "..." or param_text == "(...)":
        return "..."

    # Try to extract the type identifier from the parameter declaration
    for child in params.children:
        if child.type == "parameter_declaration":
            type_node = child.child_by_field_name("type")
            if type_node is not None:
                return _text(type_node).strip()
            # Fallback: first type-like child
            for sub in child.children:
                if sub.type in ("type_identifier", "qualified_identifier",
                                "template_type", "primitive_type"):
                    return _text(sub)
        elif child.type in ("type_identifier", "qualified_identifier"):
            return _text(child)

    # Fallback: strip parens and variable name from param text
    if param_text.startswith("(") and param_text.endswith(")"):
        param_text = param_text[1:-1].strip()
    parts = param_text.split()
    type_parts = [p for p in parts
                  if p not in ("const", "&", "&&") and not p.startswith("&")]
    if type_parts:
        return " ".join(type_parts).rstrip("&").strip()

    return param_text or None


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


def _determine_return_usage(call_node) -> str:
    """Determine how the return value of a call is used based on AST context."""
    parent = call_node.parent
    if parent is None:
        return "ignored"

    parent_type = parent.type

    # Parenthesized expression — check grandparent
    if parent_type == "parenthesized_expression":
        grandparent = parent.parent
        if grandparent is not None:
            gp_type = grandparent.type
            if gp_type in ("if_statement", "while_statement",
                           "do_statement", "switch_statement",
                           "for_statement"):
                return "checked"
            parent = grandparent
            parent_type = parent.type

    # Assignment
    if parent_type in ("assignment_expression", "init_declarator",
                       "declaration"):
        return "assigned"

    # Return
    if parent_type == "return_statement":
        return "returned"

    # Condition of if/while/for/switch
    if parent_type in ("if_statement", "while_statement", "do_statement",
                       "for_statement", "switch_statement"):
        return "checked"

    # Binary/unary/ternary expressions
    if parent_type in ("binary_expression", "unary_expression",
                       "conditional_expression"):
        return "checked"

    # Bare expression statement
    if parent_type == "expression_statement":
        return "ignored"

    # As argument to another call
    if parent_type == "argument_list":
        return "assigned"

    # Comma expression
    if parent_type == "comma_expression":
        return "ignored"

    # Initializer list
    if parent_type in ("initializer_list", "initializer_pair"):
        return "assigned"

    # Member access: foo().bar or foo()->bar
    if parent_type == "field_expression":
        return "assigned"

    # Subscript: foo()[0]
    if parent_type == "subscript_expression":
        return "assigned"

    # Pointer dereference: *foo()
    if parent_type == "pointer_expression":
        return "assigned"

    # Cast: (int)foo()
    if parent_type == "cast_expression":
        return "assigned"

    return "ignored"


# ── Doc-comment extraction ──────────────────────────────────────────


def _get_doc_comment_for_node(ts_node) -> str | None:
    """Get the doc comment (/** ... */ or ///) immediately preceding a node.

    Handles:
    - /** ... */ Doxygen block comments
    - /// triple-slash Doxygen line comments
    """
    # Try previous named sibling first, fall back to prev_sibling
    prev = ts_node.prev_named_sibling
    if prev is None:
        prev = ts_node.prev_sibling

    if prev is not None and prev.type == "comment":
        text = _text(prev)
        if text.startswith("/**"):
            return _parse_block_doc(text)
        if text.startswith("///"):
            return _collect_triple_slash_comments(ts_node)

    return None


def _collect_triple_slash_comments(ts_node) -> str | None:
    """Collect consecutive /// comments immediately above a node."""
    lines: list[str] = []
    current = ts_node.prev_sibling
    while current is not None and current.type == "comment":
        text = _text(current)
        if text.startswith("///"):
            content = text[3:]
            if content.startswith(" "):
                content = content[1:]
            lines.insert(0, content)

            # Check for blank line gap
            next_sib = current.next_named_sibling
            if next_sib is not None:
                gap = next_sib.start_point.row - current.end_point.row
                if gap > 1:
                    break

            current = current.prev_sibling
        else:
            break

    if lines:
        return "\n".join(lines).strip() or None
    return None


def _parse_block_doc(raw: str) -> str | None:
    """Parse a /** ... */ doc comment, stripping delimiters and leading asterisks."""
    content = raw
    if content.startswith("/**"):
        content = content[3:]
    if content.endswith("*/"):
        content = content[:-2]

    doc_lines = content.split("\n")
    cleaned: list[str] = []
    for line in doc_lines:
        stripped = line.strip()
        if stripped.startswith("* "):
            cleaned.append(stripped[2:])
        elif stripped.startswith("*"):
            cleaned.append(stripped[1:])
        else:
            cleaned.append(stripped)

    result_text = "\n".join(cleaned).strip()
    return result_text if result_text else None


def _extract_leading_doc_comment(root) -> str | None:
    """Extract a leading doc comment at the top of the file."""
    if not root.children:
        return None

    for child in root.children:
        if child.type == "comment":
            text = _text(child)
            if text.startswith("/**"):
                return _parse_block_doc(text)
            continue
        break
    return None


# ── Utility helpers ─────────────────────────────────────────────────


def _find_child_by_type(node, type_name: str):
    """Find the first child with the given type."""
    for child in node.children:
        if child.type == type_name:
            return child
    return None
