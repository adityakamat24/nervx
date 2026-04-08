"""Linker: resolves raw calls and imports into concrete graph edges."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .parser import Node, ParseResult, RawCall, RawImport


@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


def build_symbol_index(parse_results: list[ParseResult]) -> dict[str, list[Node]]:
    """Build a mapping from short name -> list of nodes with that name."""
    index: dict[str, list[Node]] = {}
    for pr in parse_results:
        for node in pr.nodes:
            if node.kind == "file":
                continue
            index.setdefault(node.name, []).append(node)
    return index


def _build_file_index(parse_results: list[ParseResult]) -> dict[str, list[Node]]:
    """Build a mapping from file_path -> list of non-file nodes in that file."""
    index: dict[str, list[Node]] = {}
    for pr in parse_results:
        for node in pr.nodes:
            if node.kind != "file":
                index.setdefault(node.file_path, []).append(node)
    return index


def _build_import_map(parse_results: list[ParseResult]) -> dict[str, set[str]]:
    """Build a mapping from file_path -> set of module paths it imports."""
    imp_map: dict[str, set[str]] = {}
    for pr in parse_results:
        for ri in pr.raw_imports:
            imp_map.setdefault(ri.importer_file, set()).add(ri.module_path)
    return imp_map


def _module_to_file_paths(module_path: str) -> list[str]:
    """Convert a dotted module path to possible relative file paths.

    e.g. 'agents.factory' -> ['agents/factory.py', 'agents/factory/__init__.py']
    Also handles relative imports starting with '.'.
    """
    cleaned = module_path.lstrip(".")
    if not cleaned:
        return []
    parts = cleaned.split(".")
    base = "/".join(parts)
    return [f"{base}.py", f"{base}/__init__.py"]


def resolve_imports(
    parse_results: list[ParseResult],
    symbol_index: dict[str, list[Node]],
) -> list[Edge]:
    """Resolve import statements into edges."""
    all_files = {pr.file_path for pr in parse_results}
    edges: list[Edge] = []

    for pr in parse_results:
        for ri in pr.raw_imports:
            # Try to resolve the module path to a file in the project
            possible_files = _module_to_file_paths(ri.module_path)
            target_file = None
            for pf in possible_files:
                if pf in all_files:
                    target_file = pf
                    break

            if target_file is None:
                continue

            # Edge from importing file to target file
            edges.append(Edge(
                source_id=ri.importer_file,
                target_id=target_file,
                edge_type="imports",
            ))

            # For 'from X import Y', also create edges to specific symbols
            if ri.is_from_import and ri.imported_names:
                for name in ri.imported_names:
                    if name == "*":
                        continue
                    candidates = symbol_index.get(name, [])
                    for c in candidates:
                        if c.file_path == target_file:
                            edges.append(Edge(
                                source_id=ri.importer_file,
                                target_id=c.id,
                                edge_type="imports",
                            ))
                            break

    return edges


def resolve_calls(
    parse_results: list[ParseResult],
    symbol_index: dict[str, list[Node]],
) -> list[Edge]:
    """Resolve raw calls into edges using disambiguation heuristics."""
    import_map = _build_import_map(parse_results)
    edges: list[Edge] = []
    seen: set[tuple[str, str, str]] = set()

    for pr in parse_results:
        for rc in pr.raw_calls:
            target = _resolve_single_call(
                rc, pr.file_path, symbol_index, import_map,
            )
            if target is None:
                continue

            key = (rc.caller_id, target.id, "calls")
            if key in seen:
                continue
            seen.add(key)

            meta = {}
            if rc.error_handling:
                meta["error_handling"] = rc.error_handling

            edges.append(Edge(
                source_id=rc.caller_id,
                target_id=target.id,
                edge_type="calls",
                metadata=meta,
            ))

    return edges


def _resolve_single_call(
    rc: RawCall,
    caller_file: str,
    symbol_index: dict[str, list[Node]],
    import_map: dict[str, set[str]],
) -> Node | None:
    """Resolve a single RawCall to a target Node, or None if unresolved."""
    callee = rc.callee_text

    # Step 1: Strip 'self.' prefix
    if callee.startswith("self."):
        callee = callee[5:]

    # Step 2: Take the last segment for qualified names
    # e.g. "memory.store" -> "store", "self.state.get_player" -> "get_player"
    if "." in callee:
        qualifier = callee.rsplit(".", 1)[0]
        short_name = callee.rsplit(".", 1)[1]
    else:
        qualifier = ""
        short_name = callee

    # Step 3: Look up in symbol index
    candidates = symbol_index.get(short_name, [])
    if not candidates:
        return None

    # Step 4: Exactly one match
    if len(candidates) == 1:
        return candidates[0]

    # Step 5: Disambiguate
    caller_imports = import_map.get(caller_file, set())
    caller_module = caller_file.split("/")[0] if "/" in caller_file else ""

    # 5a: Prefer same file
    same_file = [c for c in candidates if c.file_path == caller_file]
    if len(same_file) == 1:
        return same_file[0]

    # 5b: Prefer symbols in files imported by the caller
    imported_files = set()
    for mod in caller_imports:
        for pf in _module_to_file_paths(mod):
            imported_files.add(pf)
    in_imported = [c for c in candidates if c.file_path in imported_files]
    if len(in_imported) == 1:
        return in_imported[0]

    # 5c: Use qualifier to match against candidate node IDs
    if qualifier:
        qual_parts = qualifier.lower().split(".")
        scored = []
        for c in candidates:
            cid_lower = c.id.lower()
            match_score = sum(1 for qp in qual_parts if qp in cid_lower)
            scored.append((match_score, c))
        scored.sort(key=lambda x: -x[0])
        if scored and scored[0][0] > 0:
            if len(scored) == 1 or scored[0][0] > scored[1][0]:
                return scored[0][1]

    # 5d: Prefer same top-level module
    if caller_module:
        same_module = [c for c in candidates
                       if c.file_path.startswith(caller_module + "/")]
        if len(same_module) == 1:
            return same_module[0]

    # 5e: First match fallback
    return candidates[0]


def resolve_inheritance(
    parse_results: list[ParseResult],
    symbol_index: dict[str, list[Node]],
) -> list[Edge]:
    """Resolve class inheritance into edges."""
    edges: list[Edge] = []

    for pr in parse_results:
        for node in pr.nodes:
            if node.kind != "class":
                continue
            for tag in node.tags:
                if not tag.startswith("extends:"):
                    continue
                bases = tag[len("extends:"):].split(",")
                for base_name in bases:
                    base_name = base_name.strip()
                    candidates = symbol_index.get(base_name, [])
                    if candidates:
                        # Prefer same file, then first match
                        target = candidates[0]
                        for c in candidates:
                            if c.file_path == node.file_path:
                                target = c
                                break
                        edges.append(Edge(
                            source_id=node.id,
                            target_id=target.id,
                            edge_type="inherits",
                        ))

    return edges


def resolve_dispatches(
    parse_results: list[ParseResult],
    inheritance_edges: list[Edge],
) -> list[Edge]:
    """C18: link abstract/base methods to their concrete subclass overrides.

    This is a *soft* pass intended to give ``trace`` visibility into
    polymorphic dispatch that static call-linking can't resolve. For every
    subclass -> base edge in ``inheritance_edges``, we walk the transitive
    base chain and for each method on the subclass that shares its name with
    a method on an ancestor we emit a ``dispatches_to`` edge pointing from
    the ancestor method to the subclass override.

    ``dispatches_to`` is deliberately distinct from ``calls`` so that strict
    users (and ``trace --calls-only``) can ignore it. The reverse edge
    ``dispatched_from`` is populated automatically by the store.
    """
    # Index nodes by ID for quick lookup and group methods by their parent class.
    nodes_by_id: dict[str, Node] = {}
    methods_by_class: dict[str, dict[str, Node]] = {}
    for pr in parse_results:
        for node in pr.nodes:
            if node.kind == "file":
                continue
            nodes_by_id[node.id] = node
            if node.kind == "method" and node.parent_id:
                methods_by_class.setdefault(node.parent_id, {})[node.name] = node

    # Build class -> list of direct base class IDs from inheritance edges.
    class_bases: dict[str, list[str]] = {}
    for e in inheritance_edges:
        if e.edge_type != "inherits":
            continue
        class_bases.setdefault(e.source_id, []).append(e.target_id)

    def _ancestor_chain(class_id: str) -> list[str]:
        """Return transitive base classes of ``class_id`` (closest first)."""
        seen: set[str] = set()
        order: list[str] = []
        frontier = list(class_bases.get(class_id, []))
        while frontier:
            nxt = frontier.pop(0)
            if nxt in seen:
                continue
            seen.add(nxt)
            order.append(nxt)
            frontier.extend(class_bases.get(nxt, []))
        return order

    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()

    for subclass_id, subclass_methods in methods_by_class.items():
        ancestors = _ancestor_chain(subclass_id)
        if not ancestors:
            continue
        for method_name, concrete in subclass_methods.items():
            # Dunders are usually not polymorphic in a useful sense (e.g.
            # every subclass has __init__); skip to keep the edge count sane.
            if method_name.startswith("__") and method_name.endswith("__"):
                continue
            for ancestor_id in ancestors:
                ancestor_methods = methods_by_class.get(ancestor_id, {})
                base_method = ancestor_methods.get(method_name)
                if base_method is None:
                    continue
                if base_method.id == concrete.id:
                    continue
                key = (base_method.id, concrete.id)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(Edge(
                    source_id=base_method.id,
                    target_id=concrete.id,
                    edge_type="dispatches_to",
                    weight=0.5,  # soft edge — lower weight than direct calls
                    metadata={"via": ancestor_id},
                ))
                # Only link to the closest override in the ancestor chain,
                # otherwise a single method on a deep base class would get
                # flooded with every same-named method below it.
                break

    return edges


def resolve_all(parse_results: list[ParseResult]) -> list[Edge]:
    """Resolve all calls, imports, and inheritance into edges."""
    symbol_index = build_symbol_index(parse_results)
    imports = resolve_imports(parse_results, symbol_index)
    calls = resolve_calls(parse_results, symbol_index)
    inheritance = resolve_inheritance(parse_results, symbol_index)
    dispatches = resolve_dispatches(parse_results, inheritance)
    return imports + calls + inheritance + dispatches
