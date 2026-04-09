"""`nervx ask` family — micro-queries that answer in 5–30 tokens.

Every subcommand is a direct graph or AST lookup; none of them read source
files. Each returns either a small dict (when called programmatically) or
a compact string (via ``format_ask``).
"""

from __future__ import annotations

import json

from nervx.attention.fuzzy import resolve_symbol
from nervx.attention.test_coverage import test_coverage_for
from nervx.memory.store import GraphStore


# ── subcommand handlers ──────────────────────────────────────────────


def ask_exists(store: GraphStore, symbol: str, pick: int | None = None) -> dict:
    node, _ = resolve_symbol(store, symbol, pick=pick)
    return {"op": "exists", "query": symbol, "result": bool(node),
            "resolved_id": node["id"] if node else ""}


def ask_signature(store: GraphStore, symbol: str, pick: int | None = None) -> dict:
    node, err = resolve_symbol(store, symbol, pick=pick)
    if node is None:
        return {"op": "signature", "query": symbol, "error": err}
    return {
        "op": "signature",
        "query": symbol,
        "resolved_id": node["id"],
        "signature": (node.get("signature") or "").strip(),
    }


def ask_calls(
    store: GraphStore, caller: str, callee: str, pick: int | None = None,
) -> dict:
    # --pick applies to the caller only; ambiguity on the callee still
    # returns a did-you-mean.
    src, err_s = resolve_symbol(store, caller, pick=pick)
    if src is None:
        return {"op": "calls", "error": err_s}
    dst, err_t = resolve_symbol(store, callee)
    if dst is None:
        return {"op": "calls", "error": err_t}

    row = store.conn.execute(
        """
        SELECT metadata FROM edges
        WHERE source_id = ? AND target_id = ? AND edge_type = 'calls'
        LIMIT 1
        """,
        (src["id"], dst["id"]),
    ).fetchone()
    result = {
        "op": "calls",
        "caller": src["id"],
        "callee": dst["id"],
        "direct": bool(row),
    }
    if row:
        try:
            meta = json.loads(row["metadata"] or "{}")
            if "line" in meta:
                result["line"] = meta["line"]
        except (TypeError, ValueError):
            pass
    return result


def ask_imports(store: GraphStore, file_path: str) -> dict:
    rel = file_path.replace("\\", "/")
    # File nodes are keyed by their relative path as the ID.
    file_node = store.get_node(rel)
    if file_node is None:
        # Try suffix match.
        for n in store.get_nodes_by_kind("file"):
            if n["id"].endswith("/" + rel) or n["id"] == rel:
                file_node = n
                break
    if file_node is None:
        return {"op": "imports", "query": file_path,
                "error": f"File node not found: {file_path}"}
    edges = store.get_edges_from(file_node["id"])
    targets = [e["target_id"] for e in edges if e["edge_type"] == "imports"]
    return {
        "op": "imports",
        "file": file_node["id"],
        "imports": targets,
        "count": len(targets),
    }


def ask_is_async(store: GraphStore, symbol: str, pick: int | None = None) -> dict:
    node, err = resolve_symbol(store, symbol, pick=pick)
    if node is None:
        return {"op": "is_async", "error": err}
    tags = _parse_tags(node.get("tags"))
    sig = (node.get("signature") or "").lower()
    is_async = "async" in tags or sig.startswith("async ")
    return {"op": "is_async", "resolved_id": node["id"], "result": is_async}


def ask_returns_type(store: GraphStore, symbol: str, pick: int | None = None) -> dict:
    node, err = resolve_symbol(store, symbol, pick=pick)
    if node is None:
        return {"op": "returns_type", "error": err}
    sig = (node.get("signature") or "").strip()
    return_type = ""
    if "->" in sig:
        tail = sig.rsplit("->", 1)[1].strip()
        return_type = tail.rstrip(":").strip()
    return {
        "op": "returns_type",
        "resolved_id": node["id"],
        "signature": sig,
        "return_type": return_type,
    }


def ask_callers_count(store: GraphStore, symbol: str, pick: int | None = None) -> dict:
    node, err = resolve_symbol(store, symbol, pick=pick)
    if node is None:
        return {"op": "callers_count", "error": err}
    count = sum(
        1 for e in store.get_edges_to(node["id"]) if e["edge_type"] == "calls"
    )
    return {"op": "callers_count", "resolved_id": node["id"], "count": count}


def ask_has_tests(store: GraphStore, symbol: str, pick: int | None = None) -> dict:
    node, err = resolve_symbol(store, symbol, pick=pick)
    if node is None:
        return {"op": "has_tests", "error": err}
    cov = test_coverage_for(store, node["id"], max_hops=3)
    return {
        "op": "has_tests",
        "resolved_id": node["id"],
        # Back-compat: `result` = any coverage (direct or transitive).
        "result": cov["direct_count"] > 0 or cov["transitive"],
        "direct": cov["direct_count"] > 0,
        "direct_count": cov["direct_count"],
        "transitive": cov["transitive"],
        "transitive_via": cov["transitive_via"],
        "transitive_hops": cov["transitive_hops"],
        # Legacy field kept for any callers parsing the old shape.
        "test_count": cov["direct_count"],
    }


# ── dispatch ─────────────────────────────────────────────────────────


HANDLERS = {
    "exists": ask_exists,
    "signature": ask_signature,
    "calls": ask_calls,
    "imports": ask_imports,
    "is-async": ask_is_async,
    "returns-type": ask_returns_type,
    "callers-count": ask_callers_count,
    "has-tests": ask_has_tests,
}


def run_ask(
    store: GraphStore,
    subcommand: str,
    args: list[str],
    pick: int | None = None,
) -> dict:
    """Dispatch a subcommand. ``args`` is the positional argument list.

    ``pick`` forwards to ``resolve_symbol`` so users can select the Nth
    fuzzy candidate without retyping the fully-qualified id. ``imports``
    takes a file path directly and ignores ``pick``.
    """
    handler = HANDLERS.get(subcommand)
    if handler is None:
        return {"op": subcommand, "error": f"Unknown ask subcommand: {subcommand}"}

    if subcommand == "calls":
        if len(args) < 2:
            return {"op": "calls", "error": "Usage: nervx ask calls <A> <B>"}
        return handler(store, args[0], args[1], pick=pick)
    if len(args) < 1:
        return {"op": subcommand, "error": f"Usage: nervx ask {subcommand} <symbol>"}
    if subcommand == "imports":
        # imports uses a file path, not a symbol id — fuzzy-pick doesn't apply.
        return handler(store, args[0])
    return handler(store, args[0], pick=pick)


# ── formatting ───────────────────────────────────────────────────────


def format_ask(result: dict) -> str:
    if "error" in result:
        return result["error"]
    op = result.get("op", "")

    if op == "exists":
        return "yes" if result["result"] else "no"
    if op == "signature":
        return result.get("signature") or "(no signature)"
    if op == "calls":
        if result.get("direct"):
            line = result.get("line")
            return f"yes (line {line})" if line else "yes"
        return "no"
    if op == "imports":
        if not result.get("imports"):
            return "(no imports indexed)"
        return "\n".join(result["imports"])
    if op == "is_async":
        return "yes" if result["result"] else "no"
    if op == "returns_type":
        rt = result.get("return_type") or ""
        return rt or "(no return annotation)"
    if op == "callers_count":
        return str(result.get("count", 0))
    if op == "has_tests":
        if result.get("direct"):
            return f"direct: yes ({result['direct_count']} refs)"
        if result.get("transitive"):
            return (
                f"direct: no; transitive: yes via `{result['transitive_via']}` "
                f"({result['transitive_hops']} hops)"
            )
        return "direct: no; transitive: no (3-hop search)"
    return json.dumps(result)


def _parse_tags(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []
