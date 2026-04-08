"""`nervx trace <from> <to>` — shortest call-path between two symbols."""

from __future__ import annotations

from pathlib import Path

from nervx.attention.fuzzy import resolve_symbol
from nervx.attention.graph_paths import bfs_path
from nervx.memory.store import GraphStore


def trace_path(
    store: GraphStore,
    source_symbol: str,
    target_symbol: str,
    include_source: bool = False,
    repo_root: str = ".",
) -> dict | str:
    """Compute the shortest call path from source to target."""
    src, err_s = resolve_symbol(store, source_symbol)
    if src is None:
        return err_s
    dst, err_t = resolve_symbol(store, target_symbol)
    if dst is None:
        return err_t

    path = bfs_path(store, src["id"], dst["id"], edge_type="calls", max_depth=8)
    if not path:
        return {
            "found": False,
            "source": src["id"],
            "target": dst["id"],
            "path": [],
        }

    hops: list[dict] = []
    for nid in path:
        n = store.get_node(nid)
        if n is None:
            continue
        hop = {
            "id": nid,
            "name": n["name"],
            "file_path": n["file_path"],
            "line_start": n.get("line_start") or 0,
            "line_end": n.get("line_end") or 0,
            "signature": (n.get("signature") or "").strip(),
        }
        if include_source:
            hop["source"] = _read_source(n, repo_root) or ""
        hops.append(hop)

    return {
        "found": True,
        "source": src["id"],
        "target": dst["id"],
        "hops": hops,
        "length": len(hops),
    }


def format_trace(result: dict | str) -> str:
    if isinstance(result, str):
        return result
    if not result["found"]:
        return f"No call path from {result['source']} to {result['target']}."

    lines: list[str] = [
        f"## trace: {result['source']} → {result['target']}  ({result['length']} hops)",
        "",
    ]
    for i, hop in enumerate(result["hops"]):
        arrow = "  " if i == 0 else "  ↓ "
        loc = f"{hop['file_path']}:{hop['line_start']}"
        sig = hop["signature"] or hop["name"]
        lines.append(f"{arrow}{loc}  {sig}")
        if "source" in hop and hop["source"]:
            lines.append("")
            lines.append(hop["source"])
            lines.append("")
    return "\n".join(lines)


def _read_source(node: dict, repo_root: str) -> str | None:
    fp = node.get("file_path")
    if not fp:
        return None
    full = Path(repo_root) / fp
    if not full.exists():
        return None
    try:
        text = full.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    ls = node.get("line_start") or 0
    le = node.get("line_end") or 0
    if not ls or not le:
        return None
    return "\n".join(lines[max(0, ls - 1):le])
