"""Architectural pattern detection from graph topology."""

from __future__ import annotations

import json

from nervx.memory.store import GraphStore


def detect_patterns(store: GraphStore):
    """Run all pattern detectors and store results.

    Pre-loads all data into memory for O(N) detection instead of O(N*M) queries.
    """
    store.clear_patterns()

    # Pre-load everything into memory once
    all_nodes = store.get_all_nodes()
    all_edges = store.get_all_edges()

    nodes_by_id = {n["id"]: n for n in all_nodes}
    classes = [n for n in all_nodes if n["kind"] == "class"]
    functions = [n for n in all_nodes if n["kind"] == "function"]

    # Build method-by-parent index
    methods_by_parent: dict[str, list[dict]] = {}
    for n in all_nodes:
        if n["kind"] == "method" and n["parent_id"]:
            methods_by_parent.setdefault(n["parent_id"], []).append(n)

    # Build edges-by-source and edges-by-type indexes
    edges_from: dict[str, list[dict]] = {}
    for e in all_edges:
        edges_from.setdefault(e["source_id"], []).append(e)

    with store.batch():
        _detect_event_bus(store, classes, methods_by_parent, edges_from, nodes_by_id)
        _detect_factory(store, functions, edges_from)
        _detect_strategy(store, classes, edges_from, nodes_by_id)
        _detect_middleware_chain(store, functions)
        _detect_repository(store, classes, methods_by_parent)
        _detect_singleton(store, classes, methods_by_parent)
        _detect_observer(store, classes, methods_by_parent)


def _detect_event_bus(store, classes, methods_by_parent, edges_from, nodes_by_id):
    emit_names = {"emit", "dispatch", "fire", "trigger", "publish", "notify"}
    listen_names = {"on", "subscribe", "listen", "register", "add_listener", "bind"}

    for cls in classes:
        methods = methods_by_parent.get(cls["id"], [])
        method_names = {m["name"] for m in methods}

        if not (method_names & emit_names and method_names & listen_names):
            continue

        # Count imported_by edges from this class
        imported_by = [e for e in edges_from.get(cls["id"], [])
                       if e["edge_type"] == "imported_by"]
        n_importers = len(imported_by)

        # Count listener call sites
        n_listeners = 0
        for lm in method_names & listen_names:
            method_id = f"{cls['id']}.{lm}"
            for e in edges_from.get(method_id, []):
                if e["edge_type"] == "called_by":
                    n_listeners += 1

        store.add_pattern(
            cls["id"], "event_bus",
            {"listeners": n_listeners, "importers": n_importers},
            f"Static call graph will NOT show all subscribers. "
            f"{n_listeners} listener registrations found across {n_importers} files. "
            f"Search for .on()/.subscribe() calls.",
        )


def _detect_factory(store, functions, edges_from):
    factory_prefixes = ("create_", "build_", "make_")

    for func in functions:
        if not any(func["name"].startswith(p) for p in factory_prefixes):
            continue
        called_by = [e for e in edges_from.get(func["id"], [])
                     if e["edge_type"] == "called_by"]
        if called_by:
            store.add_pattern(
                func["id"], "factory",
                {"callers": len(called_by)},
                f"Factory with {len(called_by)} call site(s). "
                f"When adding new types, add a case here.",
            )


def _detect_strategy(store, classes, edges_from, nodes_by_id):
    for cls in classes:
        tags = json.loads(cls["tags"]) if isinstance(cls["tags"], str) else cls["tags"]
        is_base = (
            cls["name"].startswith("Base")
            or cls["name"].startswith("Abstract")
            or "abstract" in tags
        )
        if not is_base:
            continue

        inherited_by = [e for e in edges_from.get(cls["id"], [])
                        if e["edge_type"] == "inherited_by"]
        if len(inherited_by) < 2:
            continue

        sub_names = []
        for e in inherited_by:
            sub = nodes_by_id.get(e["target_id"])
            if sub:
                sub_names.append(sub["name"])

        store.add_pattern(
            cls["id"], "strategy",
            {"implementations": sub_names},
            f"{cls['name']} is a strategy base with {len(inherited_by)} implementations: "
            f"{', '.join(sub_names)}. When modifying the interface, update all implementations.",
        )


def _detect_middleware_chain(store, functions):
    middleware_keywords = {"next", "handler", "request", "response", "middleware"}
    dir_funcs: dict[str, list[dict]] = {}
    for func in functions:
        d = "/".join(func["file_path"].split("/")[:-1]) if "/" in func["file_path"] else ""
        dir_funcs.setdefault(d, []).append(func)

    for d, funcs in dir_funcs.items():
        matches = []
        for f in funcs:
            sig_lower = (f["signature"] or "").lower()
            if any(kw in sig_lower for kw in middleware_keywords):
                matches.append(f)

        if len(matches) >= 3:
            for m in matches:
                store.add_pattern(
                    m["id"], "middleware_chain",
                    {"chain_size": len(matches)},
                    f"Middleware chain with {len(matches)} functions. Execution order matters.",
                )


def _detect_repository(store, classes, methods_by_parent):
    crud_patterns = {
        "read": {"get", "find", "read", "fetch", "load"},
        "create": {"create", "save", "write", "store"},
        "update": {"update"},
        "delete": {"delete", "remove"},
    }

    for cls in classes:
        methods = methods_by_parent.get(cls["id"], [])
        method_names = {m["name"] for m in methods}

        matched_categories = set()
        for category, names in crud_patterns.items():
            for mn in method_names:
                if any(mn.startswith(p) or mn == p for p in names):
                    matched_categories.add(category)
                    break

        if len(matched_categories) >= 3:
            store.add_pattern(
                cls["id"], "repository",
                {"crud_categories": list(matched_categories)},
                f"Data access centralized in {cls['name']}. "
                f"Don't bypass with direct storage calls.",
            )


def _detect_singleton(store, classes, methods_by_parent):
    singleton_names = {"get_instance", "instance", "get_singleton", "shared"}

    for cls in classes:
        methods = methods_by_parent.get(cls["id"], [])
        if {m["name"] for m in methods} & singleton_names:
            store.add_pattern(
                cls["id"], "singleton",
                {},
                f"{cls['name']} is a singleton. Use the class method, don't instantiate directly.",
            )


def _detect_observer(store, classes, methods_by_parent):
    add_names = {"add_observer", "register_observer", "attach"}
    remove_names = {"remove_observer", "detach"}

    for cls in classes:
        methods = methods_by_parent.get(cls["id"], [])
        method_names = {m["name"] for m in methods}

        if (method_names & add_names) and (method_names & remove_names):
            store.add_pattern(
                cls["id"], "observer",
                {},
                "Runtime callbacks — check all registration sites when modifying notifications.",
            )
