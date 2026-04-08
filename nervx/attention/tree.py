"""`nervx tree` — structural overview of a file.

Shows every symbol in a file with line counts, importance, caller counts
and notable tags. ~150 tokens instead of 4000 tokens of raw source.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from nervx.memory.store import GraphStore


def tree_file(
    store: GraphStore,
    file_path: str,
    repo_root: str = ".",
) -> dict | str:
    """Return a structured outline of one file.

    Resolves ``file_path`` relative to ``repo_root`` if it's not already
    a stored forward-slash path.
    """
    rel = _normalize_file_path(store, file_path, repo_root)
    if rel is None:
        return f"File not found in brain: {file_path}"

    nodes = store.get_nodes_by_file(rel)
    if not nodes:
        return f"No symbols indexed for {rel}. Run `nervx build` or `nervx update`."

    # Separate the synthetic file node from real symbols.
    file_node: dict | None = None
    symbols: list[dict] = []
    for n in nodes:
        if n["kind"] == "file":
            file_node = n
        else:
            symbols.append(n)
    symbols.sort(key=lambda n: n.get("line_start") or 0)

    # Caller counts via `calls` edges incoming to each node.
    caller_counts: dict[str, int] = {}
    for n in symbols:
        caller_counts[n["id"]] = sum(
            1 for e in store.get_edges_to(n["id"]) if e["edge_type"] == "calls"
        )

    # Classes first, so methods can be nested underneath.
    classes: list[dict] = []
    methods_by_parent: dict[str, list[dict]] = {}
    top_level: list[dict] = []
    for n in symbols:
        if n["kind"] == "class":
            classes.append(n)
        elif n["kind"] == "method" and n.get("parent_id"):
            methods_by_parent.setdefault(n["parent_id"], []).append(n)
        else:
            top_level.append(n)

    # Imports (from file node or top-level import-tagged nodes).
    imports: list[str] = []
    if file_node:
        for e in store.get_edges_from(file_node["id"]):
            if e["edge_type"] == "imports":
                imports.append(_short_name(e["target_id"]))

    # "Called by" summary — who pulls symbols from this file.
    called_by: set[str] = set()
    for n in symbols:
        for e in store.get_edges_to(n["id"]):
            if e["edge_type"] != "calls":
                continue
            src = store.get_node(e["source_id"])
            if src and src["file_path"] != rel:
                called_by.add(f"{src['file_path']}::{src['name']}")

    stats = store.get_file_stats(rel) or {}

    # Physical line count (if the file still exists on disk).
    total_lines = 0
    disk = Path(repo_root) / rel
    if disk.exists():
        try:
            total_lines = sum(1 for _ in disk.open("r", encoding="utf-8", errors="replace"))
        except OSError:
            total_lines = 0

    return {
        "file": rel,
        "total_lines": total_lines,
        "class_count": len(classes),
        "method_count": sum(1 for n in symbols if n["kind"] == "method"),
        "function_count": sum(1 for n in symbols if n["kind"] == "function"),
        "classes": [_summarize(c, methods_by_parent.get(c["id"], []), caller_counts) for c in classes],
        "top_level": [_summarize_single(n, caller_counts) for n in top_level],
        "imports": sorted(set(imports))[:20],
        "called_by": sorted(called_by)[:10],
        "commits_30d": stats.get("commits_30d", 0),
    }


def format_tree(tree: dict | str) -> str:
    """Render a tree dict as compact multi-line text."""
    if isinstance(tree, str):
        return tree

    lines: list[str] = []
    header = (
        f"file: {tree['file']} ({tree['total_lines']} lines, "
        f"{tree['method_count']} methods, {tree['class_count']} classes, "
        f"{tree['function_count']} functions)"
    )
    lines.append(header)
    lines.append("")

    for cls in tree["classes"]:
        lines.append(f"class {cls['name']}:")
        for m in cls["methods"]:
            lines.append("  " + _format_symbol_line(m))
        if not cls["methods"]:
            lines.append("  (no methods indexed)")
        lines.append("")

    if tree["top_level"]:
        lines.append("top-level:")
        for n in tree["top_level"]:
            lines.append("  " + _format_symbol_line(n))
        lines.append("")

    if tree["imports"]:
        lines.append(f"imports: {', '.join(tree['imports'])}")
    if tree["called_by"]:
        lines.append(f"called by: {', '.join(tree['called_by'][:5])}")
    if tree["commits_30d"]:
        lines.append(f"hotspot: {tree['commits_30d']} commits in last 30 days")

    return "\n".join(lines).rstrip() + "\n"


# ── helpers ──────────────────────────────────────────────────────────


def _normalize_file_path(
    store: GraphStore, file_path: str, repo_root: str
) -> str | None:
    """Map a CLI-provided path to the canonical relative path stored in the brain."""
    candidate = file_path.replace("\\", "/")
    if store.get_nodes_by_file(candidate):
        return candidate

    # Try relative-to-repo-root.
    try:
        abs_path = os.path.abspath(file_path)
        rel = os.path.relpath(abs_path, repo_root).replace("\\", "/")
        if store.get_nodes_by_file(rel):
            return rel
    except ValueError:
        pass

    # Try suffix match — handy when user just passed a basename.
    all_nodes = store.get_all_nodes()
    file_paths = {n["file_path"] for n in all_nodes}
    for fp in file_paths:
        if fp.endswith("/" + candidate) or fp == candidate:
            return fp
    return None


def _short_name(node_id: str) -> str:
    return node_id.split("::")[-1] if "::" in node_id else node_id


def _summarize(cls: dict, methods: list[dict], caller_counts: dict[str, int]) -> dict:
    return {
        "id": cls["id"],
        "name": cls["name"],
        "line_start": cls.get("line_start") or 0,
        "line_end": cls.get("line_end") or 0,
        "line_count": _line_count(cls),
        "caller_count": caller_counts.get(cls["id"], 0),
        "tags": _tags(cls),
        "methods": [_summarize_single(m, caller_counts) for m in sorted(
            methods, key=lambda m: m.get("line_start") or 0
        )],
    }


def _summarize_single(n: dict, caller_counts: dict[str, int]) -> dict:
    callees = []
    return {
        "id": n["id"],
        "name": n["name"],
        "kind": n["kind"],
        "line_start": n.get("line_start") or 0,
        "line_end": n.get("line_end") or 0,
        "line_count": _line_count(n),
        "importance": float(n.get("importance") or 0.0),
        "caller_count": caller_counts.get(n["id"], 0),
        "tags": _tags(n),
    }


def _line_count(n: dict) -> int:
    ls = n.get("line_start") or 0
    le = n.get("line_end") or 0
    return max(0, le - ls + 1) if ls and le else 0


def _tags(n: dict) -> list[str]:
    raw = n.get("tags") or "[]"
    try:
        tags = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (TypeError, ValueError):
        tags = []
    return [t for t in tags if not t.startswith("decorator:") and not t.startswith("extends:")]


def _format_symbol_line(n: dict) -> str:
    parts = [f"{n['name']:<22}"]
    parts.append(f"[{n['line_count']} lines]")
    extras: list[str] = []
    if n["importance"] > 10:
        extras.append(f"importance={n['importance']:.0f}")
    if n["caller_count"] > 0:
        extras.append(f"{n['caller_count']} callers")
    if "test" in n["tags"]:
        extras.append("test")
    if n["name"].startswith("_") and not n["name"].startswith("__"):
        extras.append("private")
    if extras:
        parts.append("  " + ", ".join(extras))
    return " ".join(parts)
